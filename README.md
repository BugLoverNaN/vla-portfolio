# VLA Portfolio: SmolVLA × LIBERO Reproduction

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+cu128-EE4C2C)
![LeRobot](https://img.shields.io/badge/LeRobot-v0.5+-yellow)
![GPU](https://img.shields.io/badge/GPU-RTX_4090-76B900)
![License](https://img.shields.io/badge/license-MIT-green)

> 个人 Vision-Language-Action 模型复现项目。基于 Hugging Face SmolVLA-450M，在 LeRobot 框架下完成从环境搭建、模型推理、数据集微调到 LIBERO 仿真评估的完整工程链路。

## 🎯 项目目标

复现并理解 SmolVLA 这一轻量级 VLA 模型的完整工作流，掌握：
- VLA 模型的端到端推理与微调
- LeRobot 框架的数据集 / 训练 / 评估管线
- LIBERO 仿真 benchmark 的评估协议
- 跨本体 (cross-embodiment) 迁移的工程挑战

## �� 当前进度

- [x] **M1: 环境搭建** — LeRobot v0.5+ / PyTorch 2.8 + CUDA 12.8
- [x] **M2: SmolVLA Hello World** — 加载预训练模型，端到端推理
- [x] **M2.5: Pipeline Validation** — Smoke test 通过
- [x] **M3: SO-101 微调** — 20k 步，loss 0.55 → 0.086
- [x] **M4: LIBERO 环境与数据** — Mujoco EGL + 273K 帧数据
- [x] **M5 策略 A: LIBERO-Spatial 微调与评估** — 20K 步训练 + 500 episodes 评估
- [ ] M5 策略 B: 4 suites 混合训练（进行中）
- [ ] M6: 数据效率消融实验

## 🏆 关键实验结果

### M3: SmolVLA on SO-101 PickPlace

| 指标 | 值 |
|---|---|
| 训练步数 | 20,000 |
| Initial loss | 0.55 |
| **Final loss** | **0.086** |
| 训练时长 | 1h 11min |

### M5 策略 A: SmolVLA on LIBERO-Spatial

| 指标 | 值 |
|---|---|
| 训练步数 | 20,000 |
| Dataset | LIBERO-Spatial (432 ep, 52,970 帧) |
| Initial loss | 1.83 |
| **Final loss** | **0.42** |
| 训练时长 | 2h 51min |
| **Full eval success rate (n=500)** | **60.0%** |

### Per-task breakdown (LIBERO-Spatial)

| Task | Rate | | Task | Rate |
|------|------|-|------|------|
| Task 0 | 68% | | Task 5 | 30% |
| Task 1 | 70% | | Task 6 | 70% |
| Task 2 | 56% | | Task 7 | 64% |
| Task 3 | 78% | | Task 8 | 68% |
| Task 4 | 42% | | Task 9 | 54% |

**对比参考**：

| System | LIBERO-Spatial | 备注 |
|--------|----------------|-------|
| **This work (20K steps)** | **60.0%** | Our reproduction |
| Octo baseline | ~60% | Comparable |
| Diffusion Policy | ~78% | Paper |
| OpenVLA-OFT | ~88% | 7B params |
| SmolVLA paper | 95.4% | 60K-100K steps |

**核心观察**：M5 起步 loss 是 M3 的 3.3 倍，定量度量了跨本体迁移差距（SO-101 6-DoF joint → Franka 7-DoF EEF delta）。

![M3 Loss Curve](results/03_loss_curve.png)
![M5 Loss Curve](results/05_m5_loss_curve.png)

## 🧠 核心概念

| 概念 | 关键理解 |
|---|---|
| **VLA 模型** | 从多视角图像 + state + 语言指令直接生成连续动作 |
| **SmolVLA 架构** | SmolVLM2 (16 层裁剪) + Flow Matching action expert |
| **Flow Matching** | 替代 Diffusion，推理快 3-5 倍 |
| **Action Chunking** | 一次预测 50 步，减少推理频率提高控制率 |
| **Pretraining Transfer** | M3 起步 loss 0.55 vs M5 起步 1.83，定量度量跨本体差距 |
| **Action Space 差异** | SO-101 6-DoF joint pos vs LIBERO 7-DoF EEF delta — 物理含义不同 |
| **Step / Epoch / Episode** | Episode 是数据段落 / Epoch 是训练进度 / Step 是梯度更新 |
| **磁盘满诊断** | ML pipeline 中磁盘满会伪装成各种网络错误，第一反应 `df -h` |
| **LIBERO Suite 设计** | 4 个 suite (Spatial/Object/Goal/Long) 测试不同维度能力 |
| **统计方差** | Smoke (n=30) 50% vs Full (n=500) 60%，n 必须 >=50 才稳定 |

## 🛠 技术栈

- **VLA 模型**: SmolVLA-450M
- **框架**: LeRobot main branch
- **仿真**: Mujoco 3.8 + LIBERO + robosuite (EGL headless)
- **GPU**: NVIDIA RTX 4090 D (24GB)
- **环境**: AutoDL，PyTorch 2.8.0 + CUDA 12.8 + Python 3.12

## 📁 仓库结构

    .
    ├── scripts/
    │   ├── 01-05    # M2-M3: SmolVLA inference + SO-101 training
    │   ├── 06       # M4: LIBERO env validation
    │   ├── 07a-07   # M5: LIBERO smoke + training
    │   ├── 08a-08   # M5: LIBERO smoke + full eval
    │   ├── 09-10    # M5: visualization + summary
    │   └── 11-11a   # M5b: 4 suites training (planned)
    ├── results/     # loss curves, summaries, CSV data
    ├── notes/       # installation, 14 pitfalls, M3/M5 reflections
    └── README.md

## 📈 详细结果

- M3 摘要: [results/03_m3_training_summary.md](results/03_m3_training_summary.md)
- M5 摘要: [results/05_m5_training_summary.md](results/05_m5_training_summary.md)
- M3 反思: [notes/04_m3_training_results.md](notes/04_m3_training_results.md)
- M5 反思: [notes/05_m5_libero_training_reflections.md](notes/05_m5_libero_training_reflections.md)
- 14 个工程踩坑: [notes/03_finetuning_setup.md](notes/03_finetuning_setup.md)

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
