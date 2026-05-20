# VLA Portfolio: SmolVLA × LIBERO Reproduction

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+cu128-EE4C2C)
![LeRobot](https://img.shields.io/badge/LeRobot-v0.5+-yellow)
![GPU](https://img.shields.io/badge/GPU-RTX_4090-76B900)
![License](https://img.shields.io/badge/license-MIT-green)

> 个人 Vision-Language-Action 模型复现项目。基于 Hugging Face SmolVLA-450M，在 LeRobot 框架下完成从环境搭建、模型推理、数据集微调到仿真评估的完整流程。

## 🎯 项目目标

复现并理解 SmolVLA 这一轻量级 VLA 模型的完整工作流，通过实践掌握：
- VLA 模型的端到端推理与微调
- LeRobot 框架的数据集 / 训练 / 评估管线
- LIBERO 仿真 benchmark 的评估协议
- 跨本体 (cross-embodiment) 迁移的工程挑战

## 📊 当前进度

- [x] **M1: 环境搭建** — LeRobot v0.5+ / PyTorch 2.8 + CUDA 12.8 / SmolVLA 依赖
- [x] **M2: SmolVLA Hello World** — 加载预训练模型，端到端推理验证
- [x] **M2.5: Fine-tuning Pipeline Validation** — Smoke test 通过，所有 v0.5 兼容性问题已解决
- [x] **M3: SO-101 数据集微调** — 20k 步完成，loss 0.5+ → 0.086，验证 SmolVLA 微调能力
- [ ] M4: LIBERO 仿真环境搭建
- [ ] M5: LIBERO 4 套件微调与评估
- [ ] M6: 跨本体迁移实验

## 🧠 已掌握的核心概念

| 概念 | 关键理解 |
|---|---|
| **VLA 模型** | 从多视角图像 + 本体状态 + 语言指令直接生成连续动作 |
| **SmolVLA 架构** | SmolVLM2 backbone (16 层裁剪) + Flow Matching action expert |
| **Flow Matching** | 替代 Diffusion 的连续动作生成方法，推理快 3-5 倍 |
| **Action Chunking** | 一次预测 N 步动作（默认 50），减少推理频率提高控制率 |
| **VLA 输入约定** | OBS_IMAGE_1 (top) + OBS_IMAGE_2 (wrist) + OBS_IMAGE_3 (side) + state |
| **Camera Schema Mismatch** | 社区数据集 camera 命名不一致，需 `rename_map` + `empty_cameras` 适配 |
| **Pretraining Transfer** | SmolVLA-base 起步 loss 已经很低，社区数据预训练对 SO-100 系列覆盖充分 |
| **Marginal Returns in Training** | 微调任务 80% 的 loss 下降发生在前 25% 训练步内 |

## 🛠 技术栈

- **VLA 模型**: SmolVLA-450M ([lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base))
- **框架**: LeRobot (main branch, ≥ 0.5.2)
- **GPU**: NVIDIA RTX 4090 D (24GB)
- **环境**: AutoDL 云服务器，PyTorch 2.8.0 + CUDA 12.8 + Python 3.12

## 📁 仓库结构

    .
    ├── scripts/
    │   ├── 01_test_smolvla_inference.py     # M2: pretrained inference
    │   ├── 02a_smoke_test.sh                # M2.5: 500-step pipeline validation
    │   ├── 02_train_smolvla_so101.sh        # M3: 20k-step fine-tuning
    │   ├── 03_plot_loss_curve.py            # M3: loss curve visualization
    │   ├── 04_test_finetuned_inference.py   # M3: inference with finetuned weights
    │   └── 05_generate_summary.py           # M3: auto-generate result summary
    ├── results/
    │   ├── 01_inference_output.txt          # M2 output
    │   ├── 02a_smoke_test_log.txt           # M2.5 smoke test log
    │   ├── 02a_smoke_test_summary.md        # M2.5 summary
    │   ├── 03_loss_curve.png                # M3 training curve plot
    │   ├── 03_loss_data.csv                 # M3 raw loss data
    │   ├── 03_m3_training_log.txt           # M3 condensed log
    │   ├── 03_m3_training_summary.md        # M3 full results
    │   └── 04_finetuned_inference_output.txt # M3 inference verification
    ├── notes/
    │   ├── installation.md                  # Setup guide
    │   ├── 03_finetuning_setup.md           # M2.5: 7 v0.5 compatibility issues
    │   └── 04_m3_training_results.md        # M3: engineering reflections
    ├── constraints.txt                       # PyTorch version lock
    └── README.md

## 🚀 快速开始

详见 [notes/installation.md](notes/installation.md)。

## 📈 实验结果

### M2: SmolVLA 推理验证

| 指标 | 值 |
|---|---|
| 模型参数 | 450.0M |
| 推理设备 | RTX 4090 D (CUDA 12.8) |
| 输入 | state(6) + 3× image(3,256,256) + language |
| 输出 | action(6) |

### M2.5: Fine-tuning Pipeline Validation

| 指标 | 值 |
|---|---|
| 训练步数 | 500 |
| Loss | 0.549 → 0.196 |
| 总耗时 | 2 minutes |

### M3: SO-101 Full Fine-tuning

| 指标 | 值 |
|---|---|
| 训练步数 | **20,000** |
| Dataset | SO-101 PickPlace (50 ep, 11,939 frames) |
| Initial loss | ~0.5 |
| **Final loss** | **0.086** |
| Final grad norm | ~1.4 |
| 总耗时 | **1h 11min** (RTX 4090 D) |
| Throughput | ~4-5 steps/sec |

![Loss Curve](results/03_loss_curve.png)

📄 完整摘要: [results/03_m3_training_summary.md](results/03_m3_training_summary.md)
🔧 工程反思: [notes/04_m3_training_results.md](notes/04_m3_training_results.md)
🐛 兼容性踩坑: [notes/03_finetuning_setup.md](notes/03_finetuning_setup.md)

## 📚 参考资料

- [SmolVLA Paper (arXiv:2506.01844)](https://arxiv.org/abs/2506.01844)
- [SmolVLA Blog](https://huggingface.co/blog/smolvla)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LIBERO Benchmark](https://libero-project.github.io/)

## 👤 About the Author

**Background**: B.S. & M.S. at Southern University of Science and Technology (SUSTech)
- **B.S.**: Robotics Engineering
- **M.S.**: Intelligent Manufacturing and Robotics

**Research interests**: modern robot control theory · hybrid force-position / impedance control · 6-DoF force sensor parameter identification and external force estimation · hand-eye calibration · multi-modal visual perception · 6-DoF visual grasping · machine learning / deep learning modeling and optimization · LLM engineering · end-to-end robot system deployment

## 📮 Contact

- **Email**: 1820191867@qq.com
- **GitHub**: [@BugLoverNaN](https://github.com/BugLoverNaN)

## 📝 License

MIT
