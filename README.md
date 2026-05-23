# New Wide-Net-Casting Jailbreak Attacks Risk Large Models

🎉 **Good news — our paper has been accepted to ICML 2026!** This is the official implementation. Visit our [project page](https://zzlz233.github.io/Wide-net-casting/) for paper, results, and more details.

<p align="center">
  <img src="Fig1.png" alt="Figure 1: Illustration of the single-model jailbreak scenario and the wide-net-casting jailbreak scenario." width="80%">
</p>

<p align="center">
  <em><b>Figure 1.</b> Illustration of the single-model jailbreak scenario and the wide-net-casting jailbreak scenario.</em>
</p>

### ⚙️ Requirements

To install requirements:

```
conda create -n widenet python=3.11 -y
conda activate widenet
pip install -r requirements.txt
```

### 🚀 Training

To train on 4 GPUs:

```
bash scripts/train_group.sh
```

A 2-GPU variant is also provided:

```
bash scripts/train_group_2gpu_test.sh
```

The default configuration attacks four target LLMs (`Llama-2-7b-chat`, `Vicuna-7b-v1.5`, `Mistral-7B-Instruct-v0.2`, `Vicuna-13b-v1.5`) with `Llama-2-7b` as the prompter base. To attack a subset, edit `conf/train_group.yaml` and remove the corresponding entries from `branches:`.

### 📂 Outputs

Each run writes to `res_group/<jobname>/<branch>_rank<N>/`:

- `checkpoints/` — saved prompter LoRA adapters
- `suffix_opt_dataset/` — adversarial suffixes optimized during training
- `suffix_dataset/` — adversarial suffix dataset generated on the evaluation splits

### 🙏 Acknowledgements

Our code is based on [AdvPrompter](https://github.com/facebookresearch/advprompter) ([license](https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hans)) and the [ReMiss](https://arxiv.org/abs/2406.14393) codebase.

### 📖 Citation

```bibtex
@inproceedings{xiang2026widenet,
  title     = {New Wide-Net-Casting Jailbreak Attacks Risk Large Models},
  author    = {Xiang, Qiuchi and Qu, Haoxuan and Rahmani, Hossein and Liu, Jun},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```
