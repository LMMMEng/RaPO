# [[NeurIPS 2026] Overcoming Catastrophic Forgetting in Visual Continual Learning with Reinforcement Fine-Tuning](https://arxiv.org/abs/2605.09640)

This is an official PyTorch implementation of "[**Overcoming Catastrophic Forgetting in Visual Continual Learning with Reinforcement Fine-Tuning**](https://arxiv.org/abs/2605.09640)".

## Overview

**TL;DR: A new reinforcement fine-tuning method for rehearsal-free continual learning with MLLMs.**

Reinforcement Fine-Tuning (RFT) with verifiable rewards has emerged as a powerful learning paradigm for eliciting reasoning capabilities in Multi-modal Large Language Models (MLLMs). Recent studies suggest that RFT is inherently more resilient to catastrophic forgetting than Supervised Fine-Tuning (SFT). However, whether RFT (e.g., GRPO) can effectively overcome forgetting in challenging visual continual learning settings, such as class-incremental learning (CIL) and domain-incremental learning (DIL), remains an open problem. Through a pilot study, we confirm that while RFT consistently outperforms SFT, it still suffers from non-negligible forgetting. We empirically trace this bottleneck to Trajectory-level Drift Agnosticism: among candidate rollouts achieving identical task rewards, the KL divergence from the preceding-task policy varies substantially, which strongly correlates with catastrophic forgetting across sequential tasks. Motivated by this insight, we propose Retention-aware Policy Optimization (RaPO), a simple yet effective RFT method that explicitly mitigates forgetting through trajectory-level reward shaping. Specifically, RaPO comprises two core components: (1) Retention Reward that converts trajectory-level distribution drift into a continuous reward signal, preferentially reinforcing knowledge-preserving rollouts within each group; (2) Cross-Task Advantage Normalization (CTAN), which maintains a persistent exponential moving average of reward statistics across task boundaries to stabilize the optimization progress during continual learning. Leveraging the free-form textual generalization of MLLMs, we comprehensively evaluate RaPO across five visual continual learning settings. Extensive experiments demonstrate that RaPO achieves leading performance, substantially reducing catastrophic forgetting while preserving strong plasticity. To the best of our knowledge, this work represents the first systematic exploration of RFT in visual continual learning, offering insights that we hope will inspire future research.

## Experiments

### 1. Dataset Preparation

- Please refer to [DATASET.md](data/DATASET.md) for the detailed dataset construction guidelines.

- Please also prepare the checkpoints of [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) and [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

### 2. Requirements

```bash
conda create -n rapo python=3.11
conda activate rapo
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Training

#### Image Classification

```bash
# Example on ImageNet-R
bash scripts/img_cls/inr/train_10task.sh
bash scripts/img_cls/inr/train_20task.sh
```

The same launcher layout is provided for ImageNet-A, TinyImageNet, and CUB-200 under [scripts/img_cls](scripts/img_cls).

#### Video Classification

```bash
bash scripts/video_cls/ucf101/train_5task.sh
bash scripts/video_cls/ucf101/train_10task.sh
```

#### Object Detection

```bash
COCO_IMAGE_ROOT=/path/to/coco bash scripts/det/coco/train_5task.sh
COCO_IMAGE_ROOT=/path/to/coco bash scripts/det/coco/train_10task.sh
```

## Citation

If you find this project useful for your research, please cite:

```bibtex
@inproceedings{lou2026overcoming,
  title={Overcoming Catastrophic Forgetting in Visual Continual Learning with Reinforcement Fine-Tuning},
  author={Lou, Meng and Guo, Hanzhong and Chen, Linwei and Yu, Yizhou},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
}
```

## Acknowledgments

This project builds on [EasyR1](https://github.com/hiyouga/EasyR1) and [verl](https://github.com/volcengine/verl). We gratefully thank the authors for their wonderful works.

## Contact

If you have any questions, please feel free to create an issue or contact me at `lmzmm.0921@gmail.com`.
