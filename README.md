# VLA Portfolio: SmolVLA × LIBERO Reproduction

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+cu128-EE4C2C)
![LeRobot](https://img.shields.io/badge/LeRobot-v0.5+-yellow)
![GPU](https://img.shields.io/badge/GPU-RTX_4090-76B900)
![License](https://img.shields.io/badge/license-MIT-green)

> 个人 Vision-Language-Action 模型复现项目。基于 Hugging Face SmolVLA-450M，在 LeRobot 框架下完成从环境搭建、模型推理、数据集微调到 LIBERO 仿真评估的完整工程链路，并通过单任务 vs 多任务训练对比揭示多任务干扰现象。

## 🎯 项目目标

复现并理解 SmolVLA 这一轻量级 VLA 模型的完整工作流，掌握：
- VLA 模型端到端推理与微调
- LeRobot 数据集 / 训练 / 评估管线
- LIBERO 仿真 benchmark 评估协议
- 跨本体迁移与多任务干扰的工程洞察

## 📊 当前进度

- [x] **M1: 环境搭建** — LeRobot v0.5+ / PyTorch 2.8 + CUDA 12.8
- [x] **M2: SmolVLA Hello World** — 加载预训练模型推理
- [x] **M2.5: Pipeline Validation** — Smoke test 通过
- [x] **M3: SO-101 微调** — 20K 步，loss 0.55 → 0.086
- [x] **M4: LIBERO 环境与数据** — Mujoco EGL + 273K 帧
- [x] **M5a: LIBERO-Spatial 专项训练** — 20K 步，60.0% (n=500)
- [x] **M5b: LIBERO 4-suite 混合训练** — 40K 步，4-suite avg 50.9% (n=2000)
- [ ] M6: 数据效率消融实验

## 🏆 关键实验结果

### M3: SmolVLA on SO-101 PickPlace

| 指标 | 值 |
|---|---|
| 训练步数 | 20,000 |
| Final loss | **0.086** |
| 训练时长 | 1h 11min |

### M5a: LIBERO-Spatial 专项 (单 suite)

| 指标 | 值 |
|---|---|
| 训练步数 | 20,000 |
| Final loss | 0.42 |
| **Eval success (n=500)** | **60.0%** |

### M5b: LIBERO 4-Suite 混合 (40K 步)

| Suite | Success (n=500) | Paper (450M) |
|---|---|---|
| LIBERO-Spatial | 53.8% | 95.4% |
| LIBERO-Object | 50.6% | 96.6% |
| LIBERO-Goal | **66.0%** | 93.4% |
| LIBERO-Long | 33.0% | 80.0% |
| **Average** | **50.9%** | 91.4% |

训练: Final loss 0.367, 5h 41min, batch=16, num_workers=8

### 🔑 核心发现: 多任务干扰 (Multi-Task Interference)

| 策略 | LIBERO-Spatial | 训练 |
|---|---|---|
| A (单 suite 专训) | **60.0%** | Spatial only, 20K |
| B (4 suite 混训) | 53.8% | All 4 suites, 40K |

**尽管混合训练用了 2 倍步数 + 4 倍数据，Spatial 反而低了 6.2pp。** 这是经典的多任务干扰——模型容量在 4 个任务分布间被稀释，单 suite 专精度下降。揭示了 generalist vs specialist 策略的权衡。

![M3 Loss](results/03_loss_curve.png)
![M5 Loss](results/05_m5_loss_curve.png)
![M5b Loss](results/06_m5b_loss_curve.png)

## 🧠 核心概念

| 概念 | 关键理解 |
|---|---|
| **VLA 模型** | 从多视角图像 + state + 语言指令直接生成连续动作 |
| **SmolVLA 架构** | SmolVLM2 (16 层裁剪) + Flow Matching action expert |
| **Flow Matching** | 替代 Diffusion，推理快 3-5 倍 |
| **Action Chunking** | 一次预测 50 步，减少推理频率 |
| **Pretraining Transfer** | M3 起步 loss 0.55 vs M5 1.83，量化跨本体差距 |
| **多任务干扰** | 单 suite 60% vs 4 suite 混训 53.8%，generalist 稀释 specialist |
| **训练吞吐优化** | num_workers 4→8，data_s 0.74→0.36s，吞吐 +67% |
| **GPU vs CPU 瓶颈诊断** | data_s > updt_s 即 CPU-bound，加 workers |
| **Train vs Eval 优化哲学** | 训练优化吞吐，评估保持可复现性 |
| **统计方差** | Smoke n=30 vs Full n=500，n 必须 >=50 |

## 🛠 技术栈

- **VLA 模型**: SmolVLA-450M
- **框架**: LeRobot main branch
- **仿真**: Mujoco 3.8 + LIBERO + robosuite (EGL headless)
- **GPU**: NVIDIA RTX 4090 D (24GB), 192-core Xeon host
- **环境**: AutoDL，PyTorch 2.8.0 + CUDA 12.8 + Python 3.12

## 📁 仓库结构

    .
    ├── scripts/
    │   ├── 01-05    # M2-M3: inference + SO-101 training
    │   ├── 06       # M4: LIBERO env validation
    │   ├── 07a-08   # M5a: Spatial training + eval
    │   ├── 09-10    # M5a: visualization + summary
    │   ├── 11-12    # M5b: 4-suite pipeline + A/B/C smoke
    │   └── 13       # M5b: visualization
    ├── results/     # loss curves, summaries, CSV
    ├── notes/       # installation, 17 pitfalls, M3/M5/M5b reflections
    └── README.md

## 📈 详细结果

- M3: [results/03_m3_training_summary.md](results/03_m3_training_summary.md)
- M5a: [results/05_m5_training_summary.md](results/05_m5_training_summary.md)
- M5b: [results/06_m5b_training_summary.md](results/06_m5b_training_summary.md)
- 反思: notes/04 (M3), notes/05 (M5a), notes/06 (M5b)
- 17 个工程踩坑: [notes/03_finetuning_setup.md](notes/03_finetuning_setup.md)

## 📚 参考

- [SmolVLA Paper](https://arxiv.org/abs/2506.01844)
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
