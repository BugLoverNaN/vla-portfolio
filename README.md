# VLA Portfolio: SmolVLA × LIBERO Reproduction

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+cu128-EE4C2C)
![LeRobot](https://img.shields.io/badge/LeRobot-v0.5+-yellow)
![GPU](https://img.shields.io/badge/GPU-RTX_4090-76B900)
![License](https://img.shields.io/badge/license-MIT-green)

> 个人 Vision-Language-Action 模型复现项目。基于 Hugging Face SmolVLA-450M，在 LeRobot 框架下完成从环境搭建、模型推理、数据集微调到 LIBERO 仿真评估的完整流程。

## 🎯 项目目标

复现并理解 SmolVLA 这一轻量级 VLA 模型的完整工作流，通过实践掌握：
- VLA 模型的端到端推理与微调
- LeRobot 框架的数据集 / 训练 / 评估管线
- LIBERO 仿真 benchmark 的评估协议
- 跨本体 (cross-embodiment) 迁移的工程挑战

## 📊 当前进度

- [x] **M1: 环境搭建** — LeRobot v0.5+ / PyTorch 2.8 + CUDA 12.8 / SmolVLA 依赖
- [x] **M2: SmolVLA Hello World** — 加载预训练模型，端到端推理验证
- [x] **M2.5: Fine-tuning Pipeline Validation** — Smoke test 通过
- [x] **M3: SO-101 数据集微调** — 20k 步完成，loss 0.55 → 0.086
- [x] **M4: LIBERO 仿真环境与数据集准备** — Mujoco EGL 渲染 + 273K 帧数据
- [x] **M5: LIBERO-Spatial 微调与评估** — 20k 步训练 (loss 1.83 → 0.42)，smoke eval 50%
- [ ] M5 策略 B: 4 suites 完整训练（计划周末）
- [ ] M6: 数据效率消融实验

## 🏆 关键实验结果

### M3: SmolVLA on SO-101 PickPlace

| 指标 | 值 |
|---|---|
| 训练步数 | 20,000 |
| Initial loss | ~0.55 |
| **Final loss** | **0.086** |
| 训练时长 | 1h 11min |

### M5: SmolVLA on LIBERO-Spatial

| 指标 | 值 |
|---|---|
| 训练步数 | 20,000 |
| Dataset | 432 episodes / 52,970 frames (LIBERO-Spatial) |
| Initial loss | ~1.83 |
| **Final loss** | **0.42** |
| 训练时长 | ~2h 51min |
| **Smoke eval success rate** | **50.0% (n=30)** |
| Full eval success rate (n=500) | Running... |

**对比观察**：M5 起步 loss 是 M3 的 ~3.3 倍，证实了跨本体迁移（SO-101 6-DoF joint pos → Franka 7-DoF EEF delta）对预训练迁移的削弱效应。

![M3 Loss Curve](results/03_loss_curve.png)

## 🧠 已掌握的核心概念

| 概念 | 关键理解 |
|---|---|
| **VLA 模型** | 从多视角图像 + 本体状态 + 语言指令直接生成连续动作 |
| **SmolVLA 架构** | SmolVLM2 backbone (16 层裁剪) + Flow Matching action expert |
| **Flow Matching** | 替代 Diffusion 的连续动作生成方法，推理快 3-5 倍 |
| **Action Chunking** | 一次预测 N 步动作（默认 50），减少推理频率提高控制率 |
| **Pretraining Transfer** | M3 起步 loss 0.55 vs M5 起步 1.83，定量度量跨本体迁移差距 |
| **Action Space 差异** | SO-101 是 6-DoF joint position，LIBERO 是 7-DoF EEF delta — 物理含义不同 |
| **Step vs Epoch vs Episode** | Episode 是数据段落 / Epoch 是训练进度 / Step 是梯度更新次数 |
| **磁盘满诊断** | ML pipeline 中磁盘满会伪装成各种网络错误，第一反应应该是 `df -h` |
| **LIBERO Suite 设计** | 4 个 suite (Spatial/Object/Goal/Long) 测试不同维度的能力 |

## 🛠 技术栈

- **VLA 模型**: SmolVLA-450M ([lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base))
- **框架**: LeRobot (main branch, ≥ 0.5.2)
- **仿真**: Mujoco 3.8 + LIBERO + robosuite (EGL headless rendering)
- **GPU**: NVIDIA RTX 4090 D (24GB)
- **环境**: AutoDL 云服务器，PyTorch 2.8.0 + CUDA 12.8 + Python 3.12

## 📁 仓库结构

    .
    ├── scripts/
    │   ├── 01_test_smolvla_inference.py       # M2: pretrained inference
    │   ├── 02a_smoke_test.sh                  # M2.5: SO-101 smoke test
    │   ├── 02_train_smolvla_so101.sh          # M3: SO-101 20k training
    │   ├── 03_plot_loss_curve.py              # M3: loss visualization
    │   ├── 04_test_finetuned_inference.py     # M3: post-training inference
    │   ├── 05_generate_summary.py             # M3: result summary
    │   ├── 06_test_libero_env.py              # M4: LIBERO env validation
    │   ├── 07a_libero_spatial_smoke_test.sh   # M5: 50-step schema check
    │   ├── 07b_libero_spatial_smoke_500.sh    # M5: 500-step smoke
    │   ├── 07_libero_spatial_train.sh         # M5: 20k full training
    │   ├── 08a_libero_spatial_smoke_eval.sh   # M5: 30-episode smoke eval
    │   ├── 08_libero_spatial_full_eval.sh     # M5: 500-episode full eval
    │   ├── 09_plot_m5_loss.py                 # M5: loss visualization
    │   └── 10_generate_m5_summary.py          # M5: result summary
    ├── results/
    │   ├── 03_loss_curve.png                  # M3 loss curve
    │   ├── 03_m3_training_summary.md          # M3 results
    │   ├── 05_m5_loss_curve.png               # M5 loss curve
    │   ├── 05_m5_loss_data.csv                # M5 raw loss
    │   └── 05_m5_training_summary.md          # M5 results
    ├── notes/
    │   ├── installation.md                    # Setup guide
    │   ├── 03_finetuning_setup.md             # 12 engineering pitfalls
    │   ├── 04_m3_training_results.md          # M3 reflections
    │   └── 05_m5_libero_training_reflections.md  # M5 reflections
    └── README.md

## 📈 完整实验结果

完整摘要:
- M3 (SO-101): [results/03_m3_training_summary.md](results/03_m3_training_summary.md)
- M5 (LIBERO): [results/05_m5_training_summary.md](results/05_m5_training_summary.md)

工程反思:
- M3 reflections: [notes/04_m3_training_results.md](notes/04_m3_training_results.md)
- M5 reflections: [notes/05_m5_libero_training_reflections.md](notes/05_m5_libero_training_reflections.md)

踩坑记录 (12+ engineering pitfalls): [notes/03_finetuning_setup.md](notes/03_finetuning_setup.md)

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
