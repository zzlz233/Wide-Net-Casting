import csv
import os
import re
import warnings
from collections import defaultdict
from copy import copy
from datetime import datetime

import hydra
import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import setproctitle
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tqdm import tqdm

from src.advprompteropt_group import advPrompterOpt, evaluate_prompt
from src.llm import LLM
from src.remissopt import reMissOpt
from src.sequence import MergedSeq, Seq, group_collate_fn
from src.utils import (
    Metrics,
    check_jailbroken,
    column_names,
    dotdict,
    get_affirmative_prefixes,
    get_dataloader,
    get_test_prefixes,
    hit_rate_at_n,
    log_data,
)

setproctitle.setproctitle("main_group")


def init_distributed():
    """Initialize the distributed environment."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def should_print():
    """Return True if this process should print to the console."""
    return not dist.is_initialized() or dist.get_rank() == 0


def safe_print(msg):
    """Print only from the main process."""
    if should_print():
        tqdm.write(msg)


def apply_branch_config(cfg: DictConfig, rank: int, local_rank: int):
    """Load the per-rank branch config and apply it."""
    
    # Get available branch configurations
    branches = cfg.train_group.branches
    
    if rank >= len(branches):
        raise ValueError(f"Rank {rank} exceeds available branch configurations ({len(branches)})")
    
    # Select the branch for the current rank
    branch = branches[rank]
    print(f"Rank {rank} using branch: {branch.name}")
    
    from omegaconf import OmegaConf
    
    # Load prompter config
    prompter_config_path = f"conf/prompter/{branch.prompter}.yaml"
    if os.path.exists(prompter_config_path):
        prompter_cfg = OmegaConf.load(prompter_config_path)
        # Handle inheritance (if base_prompter is in defaults)
        if 'defaults' in prompter_cfg and 'base_prompter' in str(prompter_cfg.defaults):
            base_prompter_cfg = OmegaConf.load("conf/prompter/base_prompter.yaml")
            prompter_cfg = OmegaConf.merge(base_prompter_cfg, prompter_cfg)
        
        with omegaconf.open_dict(cfg):
            cfg.prompter = prompter_cfg
        print(f"[OK] Loaded prompter config: {branch.prompter}")
    else:
        print(f"[ERR] Prompter config file missing: {prompter_config_path}")
    
    # Load target_llm config  
    target_llm_config_path = f"conf/target_llm/{branch.target_llm}.yaml"
    if os.path.exists(target_llm_config_path):
        target_llm_cfg = OmegaConf.load(target_llm_config_path)
        # Handle inheritance (if base_target_llm is in defaults)
        if 'defaults' in target_llm_cfg and 'base_target_llm' in str(target_llm_cfg.defaults):
            base_target_llm_cfg = OmegaConf.load("conf/target_llm/base_target_llm.yaml")
            target_llm_cfg = OmegaConf.merge(base_target_llm_cfg, target_llm_cfg)
        
        with omegaconf.open_dict(cfg):
            cfg.target_llm = target_llm_cfg
        print(f"[OK] Loaded target_llm config: {branch.target_llm}")
    else:
        print(f"[ERR] Target_llm config file missing: {target_llm_config_path}")
    
    
    
    # Set device. cfg.prompter and cfg.target_llm should already be the right config objects
    device = f"cuda:{local_rank}"
    if hasattr(cfg, 'prompter') and hasattr(cfg.prompter, 'llm_params'):
        cfg.prompter.llm_params.device = device
    else:
        print(f"Warning: prompter config not properly loaded")
        
    if hasattr(cfg, 'target_llm') and hasattr(cfg.target_llm, 'llm_params'):
        cfg.target_llm.llm_params.device = device
    else:
        print(f"Warning: target_llm config not properly loaded")
    
    # Use a per-rank output directory
    original_output_dir = cfg.output_dir
    cfg.output_dir = f"{original_output_dir}/{branch.name}_rank{rank}"
    
    # Use a per-rank wandb run id
    if cfg.wandb_params.id:
        cfg.wandb_params.id = f"{cfg.wandb_params.id}_{branch.name}_rank{rank}"
    else:
        cfg.wandb_params.id = f"{branch.name}_rank{rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Update derived paths
    cfg.train.suffix_opt_dataset_dir = f"{cfg.output_dir}/suffix_opt_dataset"
    cfg.train.model_save_dir = f"{cfg.output_dir}/checkpoints"
    cfg.eval.data.suffix_dataset_dir = f"{cfg.output_dir}/suffix_dataset"
    
    print(f"Rank {rank} config: {branch.name} -> {branch.target_llm} + {branch.prompter} on {device}")
    
    return cfg

def temperature_scheduler(init_temperature, cur_epoch, exp_base):
    """Temperature annealing schedule (high -> low)."""
    return init_temperature * (exp_base ** cur_epoch)

class GroupWorkspace:
    """Distributed Workspace; inherits the original functionality."""
    
    def __init__(self, cfg):
        pl.seed_everything(cfg.seed)
        self.step = 0
        self.cfg = cfg
        self.verbose = cfg.verbose and should_print()  # Only main process is verbose
        self.enable_wandb = cfg.wandb_params.enable_wandb
        self.starttime = datetime.now()
        
        # Distributed bookkeeping
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        if self.enable_wandb:
            self.init_wandb()

        safe_print("Initializing Prompter...")
        self.prompter = LLM(cfg.prompter, verbose=self.verbose)
        safe_print("Initializing TargetLLM...")
        self.target_llm = LLM(cfg.target_llm, verbose=self.verbose)

        self.test_prefixes = get_test_prefixes()
        self.affirmative_prefixes = get_affirmative_prefixes()

        self.train_table = wandb.Table(columns=column_names)
        self.eval_table = wandb.Table(columns=column_names)

        # aWTA hyperparameters
        self.awta_init_temperature = getattr(cfg.train, 'awta_init_temperature', 2.0)
        self.awta_exp_base = getattr(cfg.train, 'awta_exp_base', 0.95)

    @torch.no_grad()
    def init_wandb(self):
        safe_print("Initializing Wandb...")
        wandb_id = (
            wandb.util.generate_id()
            if self.cfg.wandb_params.id is None
            else self.cfg.wandb_params.id
        )
        config = omegaconf.OmegaConf.to_container(
            self.cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            entity=self.cfg.wandb_params.entity,
            project=self.cfg.wandb_params.project,
            config=config,
            id=wandb_id,
            resume="allow",
        )

    @torch.no_grad()
    def save_prompter(self):
        save_path = os.path.join(self.cfg.train.model_save_dir, f"step_{self.step}")
        safe_print(f" Saving prompter to {save_path}...")
        self.prompter.save_pretrained(save_path=save_path)

    def train(self):
        self.prompter_optimizer = torch.optim.Adam(
            self.prompter.parameters(), **self.cfg.train.prompter_optim_params
        )
        sampler = PrioritizedSampler(
            max_capacity=self.cfg.train.replay_buffer.size,
            alpha=self.cfg.train.replay_buffer.priority_alpha,
            beta=1.0,
        )
        self.replay_buffer = ReplayBuffer(
            storage=ListStorage(self.cfg.train.replay_buffer.size),
            batch_size=self.cfg.train.batch_size,
            sampler=sampler,
            collate_fn=group_collate_fn,
        )

        if self.cfg.train.do_initial_eval:
            self.eval()

        safe_print("Starting training...")
        pbar = tqdm(range(self.cfg.train.epochs), disable=not should_print())
        pbar.set_description(f"Training (epochs) - Rank {self.rank}")

        final_selected_logprobs = self.cfg.train.q_params.selected_logprobs
        for self.epoch in pbar:
            # linear warmup
            if self.cfg.train.q_params.init_selected_logprobs is not None:
                self.cfg.train.q_params.selected_logprobs = (
                    self.cfg.train.q_params.init_selected_logprobs
                    + (
                        final_selected_logprobs
                        - self.cfg.train.q_params.init_selected_logprobs
                    )
                    * (self.epoch / self.cfg.train.epochs)
                )

            self.train_epoch()
            
            # Optional epoch-level sync barrier
            # if dist.is_initialized():
            #     dist.barrier()  # all ranks sync at end of epoch
            
            if (
                self.cfg.train.eval_every is not None
                and (self.epoch + 1) % self.cfg.train.eval_every == 0
                and (self.epoch + 1) < self.cfg.train.epochs
            ):
                if self.cfg.train.model_save_dir is not None:
                    self.save_prompter()
                self.eval()

        if self.cfg.train.model_save_dir is not None:
            self.save_prompter()
        self.eval()

    def train_epoch(self):
        self.prompter.train()
        self.target_llm.eval()
        train_metrics = Metrics(prefix="train/")
        train_loader = get_dataloader(
            data_pth=self.cfg.train.dataset_pth,
            shuffle=True,
            augment_target=self.cfg.train.augment_target,
            batch_size=self.cfg.train.batch_size,
        )
        data = []

        pbar_batches = tqdm(train_loader, disable=not should_print())
        pbar_batches.set_description(f"Training epoch {self.epoch} - Rank {self.rank}")
        for batch_idx, batch in enumerate(pbar_batches):
            if self.cfg.train.add_target_whitespace:
                batch["target"] = [" " + t for t in batch["target"]]
            context = self.batch_to_context(batch)
            instruct = context.instruct
            target = context.target
            log_sequences = (
                batch_idx % self.cfg.wandb_params.log_sequences_every.train == 0
            )
            with torch.no_grad():

                # Generate initial suffix
                prompter_ar = self.prompter.generate_autoregressive(
                    key="suffix",
                    max_new_tokens=self.cfg.train.q_params.max_new_tokens,
                    instruct=instruct,
                )
                suffix = prompter_ar.response_sample 

                # Merge into full instruction
                full_instruct_text = (
                    MergedSeq(seqs=[instruct, suffix]).to_seq(merge_dtype="ids").text
                )
                full_instruct = Seq(
                    text=full_instruct_text,
                    tokenizer=self.target_llm.tokenizer,
                    device=self.target_llm.device,
                )

                # Evaluate initial suffix
                if self.verbose:
                    tqdm.write(f"\nStep: {self.step} | Evaluating initial suffix...")
                target_llm_tf, target_llm_ar, basemodel_tf = evaluate_prompt(
                    rank=self.rank,
                    cfg=self.cfg,
                    instruct=instruct,
                    suffix=suffix,
                    full_instruct=full_instruct,
                    target=target,
                    prompter=self.prompter,
                    target_llm=self.target_llm,
                    generate_target_llm_response=log_sequences,
                )

                # Generate optimized suffix
                if self.cfg.train.opt_type == "remiss":
                    suffix = reMissOpt( 
                        cfg=self.cfg,
                        instruct=instruct,
                        target=target,
                        prompter=self.prompter,
                        target_llm=self.target_llm,
                        rank=self.rank,
                    )
                elif self.cfg.train.opt_type == "advprompter":
                    suffix = advPrompterOpt(
                        cfg=self.cfg,
                        instruct=instruct,
                        target=target,
                        prompter=self.prompter,
                        target_llm=self.target_llm,
                    )
                else:
                    raise ValueError(
                        f"Opt type {self.cfg.train.opt_type} not recognized."
                    )

                # Merge optimized suffix
                full_instruct_text = MergedSeq(seqs=[instruct, suffix]).to_seq(
                    merge_dtype="ids"
                )
                full_instruct = Seq(
                    text=full_instruct_text.text,
                    tokenizer=self.target_llm.tokenizer,
                    device=self.target_llm.device,
                )

                # evaluate optimized suffix
                if self.verbose:
                    tqdm.write(f"\nStep: {self.step} | Evaluating optimized suffix...")
                target_llm_tf_opt, target_llm_ar_opt, basemodel_tf_opt = (
                    evaluate_prompt(
                        rank=self.rank,
                        cfg=self.cfg,
                        instruct=instruct,
                        suffix=suffix,
                        full_instruct=full_instruct,
                        target=target,
                        prompter=self.prompter,
                        target_llm=self.target_llm,
                        generate_target_llm_response=True,
                    )
                )

                # Store optimized suffix
                for i in range(instruct.bs):
                    data.append(
                        (
                            instruct.text[i],
                            target.text[i],
                            suffix.text[i],
                            full_instruct.text[i],
                        )
                    )

            self.add_to_replay_buffer(
                instruct=instruct,
                suffix=suffix,
                target=target,
                target_llm_tf=target_llm_tf,
                target_llm_tf_opt=target_llm_tf_opt,
                target_llm_ar_opt=target_llm_ar_opt,
            )

            prompter_tf_opt = self.finetune_prompter()

            log_data(
                log_table=self.train_table,
                metrics=train_metrics,
                step=self.step,
                split=self.cfg.train.dataset_key,
                batch_idx=batch_idx,
                test_prefixes=self.test_prefixes,
                affirmative_prefixes=self.affirmative_prefixes,
                log_sequences_to_wandb=log_sequences and self.enable_wandb,
                log_metrics_to_wandb=self.enable_wandb,
                prompter_ar=prompter_ar,
                target_llm_tf=target_llm_tf,
                target_llm_ar=target_llm_ar,
                basemodel_tf=basemodel_tf,
                prompter_tf_opt=prompter_tf_opt,
            )

            self.step += instruct.bs

        suffix_dataset_key = f"{self.cfg.train.dataset_key}_opt_{self.step}"
        fields = ["instruct", "target", "suffix", "full_instruct"]
        suffix_dataset = dotdict(
            data=data,
            fields=fields,
            suffix_dataset_key=suffix_dataset_key,
        )
        self.save_suffix_dataset(
            suffix_dataset, dir=self.cfg.train.suffix_opt_dataset_dir
        )

        if self.enable_wandb:
            wandb.log(dict(train_examples=copy(self.train_table)), step=self.step)

        avg_metrics = train_metrics.get_avg(
            step=self.step, log_to_wandb=self.enable_wandb
        )
        if self.verbose:
            tqdm.write(
                f" Train loss epoch {self.epoch}: {avg_metrics['avg/train/target_llm/tf/loss']:.2f}"
            )

    def batch_to_context(self, batch):
        model_map = dict(
            instruct=self.prompter,
            suffix=self.prompter,
            target=self.target_llm,
            full_instruct=self.target_llm,
        )
        context = dotdict()
        for key, model in model_map.items():
            if key in batch.keys():
                seq = Seq(
                    text=batch[key],
                    tokenizer=model.tokenizer,
                    device=model.device,
                )
            else:
                seq = None
            context[key] = seq
        return context

    def add_to_replay_buffer(
        self,
        instruct,
        suffix,
        target,
        target_llm_tf,
        target_llm_tf_opt,
        target_llm_ar_opt,
    ):
        loss_batch = target_llm_tf.loss_batch
        loss_opt_batch = target_llm_tf_opt.loss_batch

        # dist.barrier()
        # tqdm.write(f"\nRank {self.rank} - Instruct 0: {instruct.text[0]}")
        # dist.barrier()
        # tqdm.write(f"\nRank {self.rank} - Instruct 1: {instruct.text[1]}")

        # priority = priority_factor.loss_delta * relu(loss_delta) + priority_factor.jailbreaking * jailbreaking
        priority = (
            torch.relu(loss_batch - loss_opt_batch)
            * self.cfg.train.replay_buffer.priority_factor.loss_delta
        )
        if self.cfg.train.replay_buffer.priority_factor.jailbreaking > 0:
            _, target_llm_ar_opt_jailbroken_list = check_jailbroken(
                seq=target_llm_ar_opt.response_sample,
                test_prefixes=self.test_prefixes,
            )
            jailbroken = torch.tensor(
                target_llm_ar_opt_jailbroken_list, device=loss_batch.device
            )
            priority += (
                jailbroken * self.cfg.train.replay_buffer.priority_factor.jailbreaking
            )

        if dist.is_initialized() and self.world_size > 1:
            averaged_priority = self.sync_priority_across_ranks(priority)
            if self.verbose:
                safe_print(f"Priority sync - Local avg: {priority.mean().item():.3f}, Global avg: {averaged_priority.mean().item():.3f}")
        else:
            averaged_priority = priority

        for i, prio in enumerate(averaged_priority):
            if prio > 0:
                datapoint = (
                    instruct[i],
                    target[i],
                    suffix[i],
                    loss_opt_batch[i].item(),
                    prio,
                )
                idx = self.replay_buffer.add(datapoint)
                self.replay_buffer.update_priority(index=idx, priority=prio.item())
        



    def sync_priority_across_ranks(self, local_priority):
        """Synchronize priority across ranks and average it."""
        
        # Ensure tensor on the correct device
        if not isinstance(local_priority, torch.Tensor):
            local_priority = torch.tensor(local_priority, device=self.prompter.device)
        
        # Make a copy for the all_reduce operation
        averaged_priority = local_priority.clone().float()  # Ensure float dtype
        
        # Sum across all ranks
        dist.all_reduce(averaged_priority, op=dist.ReduceOp.SUM)
        
        # Compute the mean
        averaged_priority = averaged_priority / self.world_size
        
        # Optional: add verbose debug info
        if self.verbose and self.rank == 0:
            safe_print(f"\n Priority statistics - Min: {averaged_priority.min().item():.3f}, "
                    f"Max: {averaged_priority.max().item():.3f}, "
                    f"Mean: {averaged_priority.mean().item():.3f} \n")
    
        return averaged_priority

    
    def finetune_prompter(self):
        prompter_tf_opt = None
        if len(self.replay_buffer) < self.cfg.train.batch_size:
            return None

        if self.verbose:
            tqdm.write(
                f"Step: {self.step} | Sampling from replay buffer and finetuning prompter..."
            )
        num_updates = min(
            self.cfg.train.replay_buffer.num_updates,
            len(self.replay_buffer) // self.cfg.train.batch_size,
        )
        # for _ in range(num_updates):
        #     context, priority_batch = self.replay_buffer.sample(
        #         batch_size=self.cfg.train.batch_size
        #     )
        for update_idx in range(num_updates):
            sample_seed = self.cfg.seed + self.step * 10000 + update_idx
            
            # Save all random states
            torch_state = torch.get_rng_state()
            numpy_state = np.random.get_state()
            if torch.cuda.is_available():
                cuda_state = torch.cuda.get_rng_state_all()
            
            try:
                # Set deterministic seed
                torch.manual_seed(sample_seed)
                np.random.seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                
                if hasattr(self.replay_buffer._sampler, 'np_random'):
                    self.replay_buffer._sampler.np_random = np.random.RandomState(sample_seed)
                
                # Sample
                context, priority_batch = self.replay_buffer.sample(
                    batch_size=self.cfg.train.batch_size
                )

                # dist.barrier()
                # tqdm.write(f"\nRank {self.rank}-Instruct 0: {context.instruct.text[0]}-{[p.item() for p in priority_batch]}")
                # dist.barrier()
                # tqdm.write(f"\nRank {self.rank}-Instruct 0: {context.instruct.text[0]}-{[p.item() for p in priority_batch]}")
                
            finally:
                # Restore random state
                torch.set_rng_state(torch_state)
                np.random.set_state(numpy_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(cuda_state)

            prompter_tf_opt = self.finetune_prompter_step(
                instruct=context.instruct, suffix=context.suffix, loss_opt=context.loss_opt
            )
            if self.verbose:
                tqdm.write(
                    f"Step: {self.step} | Regressing Prompter to sampled target suffixes: Loss {prompter_tf_opt.loss:.3f}, Sample priorities {[p.item() for p in priority_batch]}"
                )
        return prompter_tf_opt



    # Actual parameter update
    def finetune_prompter_step(self, instruct, suffix, loss_opt):
        self.prompter_optimizer.zero_grad()
        prompter_tf_opt = self.prompter.compute_pred_loss_teacher_forced(
            key="suffix",
            instruct=instruct,
            suffix=suffix,
            loss_params=dict(hard_labels=True),
        )
        
        # if self.epoch != 0: # TODO
        #     awta_weights = self.sync_loss_opt_across_ranks(loss_opt)
        # else:
        #     awta_weights = torch.ones_like(prompter_tf_opt.loss_batch, device=self.prompter.device)

        awta_weights = self.sync_loss_opt_across_ranks(loss_opt)

        loss_batch = prompter_tf_opt.loss_batch  # [batch_size]
        awta_weights = awta_weights.to(loss_batch.device)
        weighted_loss = (loss_batch * awta_weights).mean()


        # Backprop and step
        weighted_loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.prompter.parameters(), max_norm=1.0)

        self.prompter_optimizer.step()

        # Log raw and weighted loss
        if self.enable_wandb:
            wandb.log({
                "regression_loss": prompter_tf_opt.loss.item(),  # Raw scalar loss
                "regression_loss_batch_mean": loss_batch.mean().item(),  # Mean of per-sample loss
                "awta_weighted_loss": weighted_loss.item(),
                "awta_weight_mean": awta_weights.mean().item(),
                "awta_weight_std": awta_weights.std().item(),
                "loss_batch_std": loss_batch.std().item(),  # Std of per-sample loss
            }, step=self.step)
        
        # Return a new object to keep the interface consistent
        from copy import copy
        weighted_tf_opt = copy(prompter_tf_opt)
        weighted_tf_opt.loss = weighted_loss
        
        return weighted_tf_opt
        
        # loss = prompter_tf_opt.loss
        # loss.backward()
        # self.prompter_optimizer.step()
        # if self.enable_wandb:
        #     wandb.log({"regression_loss": loss.item()}, step=self.step)
        # return prompter_tf_opt
        
    def sync_loss_opt_across_ranks(self, loss_opt_batch):
        """
        Gather loss_opt across branches and compute aWTA weights.
        
        Args:
            loss_opt_batch: this branch's loss_opt [batch_size] (may live on CPU)
        
        Returns:
            awta_weights: aWTA weights [batch_size] on the original device
        """
        if not dist.is_initialized() or self.world_size <= 1:
            # Single-process case: weight is 1
            if isinstance(loss_opt_batch, torch.Tensor):
                return torch.ones_like(loss_opt_batch)
            else:
                return torch.ones(len(loss_opt_batch), device=self.prompter.device)
        
        # Normalize input tensor device
        if not isinstance(loss_opt_batch, torch.Tensor):
            # Not a tensor yet: convert and move to GPU
            loss_opt_batch = torch.tensor(loss_opt_batch, device=self.prompter.device)
            original_device = self.prompter.device
        else:
            # Already a tensor: remember its original device
            original_device = loss_opt_batch.device
            if loss_opt_batch.device.type == 'cpu':
                loss_opt_batch = loss_opt_batch.to(self.prompter.device)
        
        # Gather loss_opt from all ranks (now all on GPU)
        all_loss_opt = [torch.zeros_like(loss_opt_batch) for _ in range(self.world_size)]
        dist.all_gather(all_loss_opt, loss_opt_batch)
        
        # Stack into a [batch_size, world_size] matrix
        loss_opt_matrix = torch.stack(all_loss_opt, dim=1)  # [batch_size, world_size]
        
        # Compute current temperature
        current_temp = temperature_scheduler(
            self.awta_init_temperature, 
            self.step, 
            self.awta_exp_base
        )
        
        # aWTA weights: smaller loss_opt -> larger weight
        awta_weights = torch.softmax(-loss_opt_matrix / current_temp, dim=1)  # [batch_size, world_size]
        
        # Extract this rank's weights
        current_rank_weights = awta_weights[:, self.rank]  # [batch_size]
        
        if original_device != current_rank_weights.device:
            current_rank_weights = current_rank_weights.to(original_device)
        
        # Debug info
        if self.verbose and self.rank == 0:
            safe_print(f"aWTA Temperature: {current_temp:.3f}")
            safe_print(f"Loss_opt across ranks: {loss_opt_matrix[0].cpu().numpy()}")
            safe_print(f"aWTA weights across ranks: {awta_weights[0].cpu().numpy()}")
            # safe_print(f"Original device: {original_device}, Final device: {current_rank_weights.device}")
        
        return current_rank_weights.detach()

    @torch.no_grad()
    def eval(self):
        suffix_dataset_pth_dct = self.generate_suffix_datasets()
        self.eval_suffix_datasets(suffix_dataset_pth_dct)

    @torch.no_grad()
    def generate_suffix_datasets(self):
        suffix_dataset_pth_dct = {}
        for dataset_key, dataset_pth in self.cfg.eval.data.dataset_pth_dct.items():
            suffix_dataset = self.generate_suffix_dataset(
                dataset_key=dataset_key, dataset_pth=dataset_pth
            )
            suffix_dataset_pth = self.save_suffix_dataset(
                suffix_dataset, dir=self.cfg.eval.data.suffix_dataset_dir
            )
            suffix_dataset_pth_dct[suffix_dataset.suffix_dataset_key] = (
                suffix_dataset_pth
            )
        return suffix_dataset_pth_dct

    @torch.no_grad()
    def generate_suffix_dataset(self, dataset_key, dataset_pth):
        self.prompter.eval()
        self.target_llm.eval()

        if self.cfg.prompter.gen_params.do_sample:
            num_trials = self.cfg.eval.num_trials
        else:
            if self.cfg.eval.num_trials != 1:
                warnings.warn(
                    "Prompter generation is deterministic, but num_trials > 1. Setting num_trials to 1."
                )
            num_trials = 1

        data = []

        suffix_dataset_key = f"{dataset_key}_{self.step}"
        eval_loader = get_dataloader(
            data_pth=dataset_pth,
            shuffle=False,
            augment_target=False,
            batch_size=self.cfg.eval.batch_size,
        )
        pbar_batches = tqdm(eval_loader, disable=not should_print())
        pbar_batches.set_description(f"Generating suffix dataset {suffix_dataset_key}")
        for batch in pbar_batches:
            context = self.batch_to_context(batch)
            instruct = context.instruct
            target = context.target
            batch_data = []
            for max_new_tokens in self.cfg.eval.prompter.max_new_tokens_list:
                trial_data = []
                for trial in range(num_trials):
                    prompter_ar = self.prompter.generate_autoregressive(
                        key="suffix",
                        max_new_tokens=max_new_tokens,
                        instruct=instruct,
                    )
                    suffix = prompter_ar.response_sample
                    full_instruct = MergedSeq(seqs=[instruct, suffix]).to_seq(
                        merge_dtype="ids"
                    )

                    basemodel_tf = self.prompter.compute_pred_loss_teacher_forced(
                        key="suffix",
                        instruct=instruct,
                        suffix=suffix,
                        use_basemodel=True,
                        loss_params=dict(hard_labels=True),
                    )
                    if should_print():
                        tqdm.write(f"Perplexity: {basemodel_tf.perplexity}")

                    assert instruct.bs == target.bs == suffix.bs
                    datapoint = []
                    for i in range(instruct.bs):
                        datapoint.append(
                            (
                                instruct.text[i],
                                target.text[i],
                                suffix.text[i],
                                full_instruct.text[i],
                            )
                        )
                    trial_data.append(datapoint)
                batch_data.append(trial_data)

            for i in range(instruct.bs):
                for j in range(len(self.cfg.eval.prompter.max_new_tokens_list)):
                    for k in range(num_trials):
                        data.append(batch_data[j][k][i])

        suffix_dataset = dotdict(
            data=data,
            fields=["instruct", "target", "suffix", "full_instruct"],
            suffix_dataset_key=suffix_dataset_key,
        )

        return suffix_dataset

    @torch.no_grad()
    def save_suffix_dataset(self, suffix_dataset, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        suffix_dataset_pth = os.path.join(
            dir,
            suffix_dataset.suffix_dataset_key + ".csv",
        )
        safe_print(
            f" Saving {suffix_dataset.suffix_dataset_key} to {suffix_dataset_pth}"
        )
        with open(suffix_dataset_pth, "w") as csvfile:
            csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            csvwriter.writerow(suffix_dataset.fields)
            csvwriter.writerows(suffix_dataset.data)
        return suffix_dataset_pth

    @torch.no_grad()
    def eval_suffix_datasets(self, suffix_dataset_pth_dct):
        for suffix_dataset_key, suffix_dataset_pth in suffix_dataset_pth_dct.items():
            self.eval_suffix_dataset(
                suffix_dataset_key=suffix_dataset_key,
                suffix_dataset_pth=suffix_dataset_pth,
            )

    @torch.no_grad()
    def eval_suffix_dataset(self, suffix_dataset_key, suffix_dataset_pth):
        self.prompter.eval()
        self.target_llm.eval()

        # split = suffix_dataset_key
        split = re.sub("[^a-zA-Z]", "", suffix_dataset_key)

        eval_loader = get_dataloader(
            suffix_dataset_pth,
            shuffle=False,
            augment_target=False,
            batch_size=self.cfg.eval.batch_size,
        )
        eval_metrics = Metrics(prefix=split + "_eval/")

        instruct_jb_dict = defaultdict(list)
        processed_samples, ppl_sum = 0, 0
        pbar = tqdm(eval_loader, disable=not should_print())
        pbar.set_description(
            f"Evaluating suffix dataset {suffix_dataset_key} | Jailbroken 0/0 | Success 0/0"
        )

        all_results = defaultdict(list)
        for batch_idx, batch in enumerate(pbar):
            context = self.batch_to_context(batch)
            instruct = context.instruct
            suffix = context.suffix
            full_instruct = context.full_instruct
            target = context.target
            target_llm_tf, target_llm_ar, basemodel_tf = evaluate_prompt(
                rank=self.rank,
                cfg=self.cfg,
                instruct=instruct,
                suffix=suffix,
                full_instruct=full_instruct,
                target=target,
                prompter=self.prompter,
                target_llm=self.target_llm,
                generate_target_llm_response=True,
            )

            # --------- check jb for each trial
            _, jailbroken_list = check_jailbroken(
                seq=target_llm_ar.response_sample, test_prefixes=self.test_prefixes
            )
            instruct = instruct
            assert instruct.bs == len(jailbroken_list)
            instruct_text = instruct.text
            for i in range(instruct.bs):
                instruct_jb_dict[instruct_text[i]].append(jailbroken_list[i])
            # -----------

            log_data(
                log_table=None,
                metrics=eval_metrics,
                step=self.step,
                split=split,
                batch_idx=batch_idx,
                test_prefixes=self.test_prefixes,
                affirmative_prefixes=self.affirmative_prefixes,
                batch_size=self.cfg.eval.batch_size,
                log_sequences_to_wandb=False,
                log_metrics_to_wandb=False,
                target_llm_tf=target_llm_tf,
                target_llm_ar=target_llm_ar,
                basemodel_tf=basemodel_tf,
            )
            processed_samples += instruct.bs
            if basemodel_tf is not None:
                ppl_sum += basemodel_tf.perplexity.sum().item()

            total_jailbroken = sum(
                eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"]
            )
            if should_print():
                pbar.set_description(
                    f"Evaluating {suffix_dataset_key} | Jailbroken {total_jailbroken}/{processed_samples}"
                )

            # log results
            all_results["forbidden_prompt"].extend(instruct.text)
            all_results["response"].extend(target_llm_ar.response_sample.text)
            all_results["suffix"].extend(suffix.text)

        # save
        response_df = pd.DataFrame(dict(all_results))
        save_dirname = os.path.dirname(suffix_dataset_pth)
        save_basename = os.path.basename(suffix_dataset_pth).split(".")[0]
        save_path = os.path.join(save_dirname, f"{save_basename}_response.csv")
        response_df.to_csv(save_path)
        safe_print(f"Saved responses to {save_path}")

        avg_metrics = eval_metrics.get_avg(step=self.step, log_to_wandb=False)
        avg_metrics["avg/" + split + "_eval/target_llm/ar/jailbroken_sum"] = (
            float(
                sum(eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"])
            )
            / processed_samples
        )

        safe_print(
            f" Loss: {avg_metrics['avg/' + split + '_eval/target_llm/tf/loss']:.2f}"
        )
        safe_print(
            f" Jailbroken: {avg_metrics['avg/' + split + '_eval/target_llm/ar/jailbroken_sum']:.2f}"
        )
        safe_print(f" PPL: {float(ppl_sum) / processed_samples:.2f}")
        jb_all = [jb_list for (instruct, jb_list) in instruct_jb_dict.items()]
        max_length = max(len(sublist) for sublist in jb_all)
        padded_list = [
            np.pad(sublist, (0, max_length - len(sublist)), "constant")
            for sublist in jb_all
        ]
        jb_stat_np = np.array(padded_list)
        for ti in range(1, jb_stat_np.shape[1] + 1):
            safe_print(
                f"{suffix_dataset_key} | hit rate @ {ti}: {hit_rate_at_n(jb_stat_np, ti)}"
            )
        if self.enable_wandb:
            wandb.log(avg_metrics, step=self.step)
            wandb.log(dict(eval_examples=copy(self.eval_table)), step=self.step)


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig):
    # Initialize the distributed environment
    rank, world_size, local_rank = init_distributed()

    # if rank != 0:
    #     import sys
    #     sys.stdout = open(f"/dev/null", 'w')
    
    safe_print("Starting distributed run...")
    safe_print(f"Rank {rank}/{world_size}, Local rank {local_rank}")
    
    # Apply branch-specific configuration
    cfg = apply_branch_config(cfg, rank, local_rank)
    
    # if should_print():
    #     safe_print(f"Using parameters: \n{OmegaConf.to_yaml(cfg)}")
    
    workspace = GroupWorkspace(cfg)
    if cfg.mode == "train":
        workspace.train()
    elif cfg.mode == "eval":
        workspace.eval()
    elif cfg.mode == "eval_suffix_dataset":
        workspace.eval_suffix_datasets(cfg.eval.suffix_dataset_pth_dct)
    else:
        raise ValueError(f"Mode {cfg.mode} not recognized.")
    
    safe_print("Finished!")


if __name__ == "__main__":
    main()