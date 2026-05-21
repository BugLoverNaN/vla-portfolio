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
- [x] **M3: SO-101 数据集微调** — 20k 步完成，loss 0.5+ → 0.086
- [x] **M4: LIBERO 仿真环境与数据集准备** — Mujoco EGL 渲染验证 + 273K 帧数据就绪
- [ ] M5: LIBERO 微调与评估
- [ ] M6: 跨本体迁移 / 数据效率实验

## 🧠 已掌握的核心概念

| 概念 | 关键理解 |
|---|---|
| **VLA 模型** | 从多视角图像 + 本体状态 + 语言指令直接生成连续动作 |
| **SmolVLA 架构** | SmolVLM2 backbone (16 层裁剪) + Flow Matching action expert |
| **Flow Matching** | 替代 Diffusion 的连续动作生成方法，推理快 3-5 倍 |
| **Action Chunking** | 一次预测 N 步动作（默认 50），减少推理频率提高控制率 |
| **Pretraining Transfer** | 起步 loss ≈ 0.55，证实预训练对 SO-100 系列覆盖充分 |
| **Marginal Returns** | 微调任务 80% 的 loss 下降发生在前 25% 训练步内 |
| **Action Space 差异** | SO-101 是 6-DoF joint position，LIBERO 是 7-DoF EEF delta — 物理含义不同，跨本体迁移本质上是 action head 重训问题 |
| **Step vs Epoch vs Episode** | Episode 是数据层面的段落 / Epoch 是训练进度 / Step 是梯度更新次数 |
| **磁盘满诊断** | ML pipeline 中磁盘满会伪装成各种网络错误，第一反应应该是 `df -h` |

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
    │   ├── 02a_smoke_test.sh                  # M2.5: 500-step pipeline validation
    │   ├── 02_train_smolvla_so101.sh          # M3: 20k-step fine-tuning
    │   ├── 03_plot_loss_curve.py              # M3: loss curve visualization
    │   ├── 04_test_finetuned_inference.py     # M3: inference with finetuned weights
    │   ├── 05_generate_summary.py             # M3: auto-generate result summary
    │   └── 06_test_libero_env.py              # M4: LIBERO sim env validation
    ├── results/
    │   ├── 01_inference_output.txt            # M2
    │   ├── 02a_smoke_test_log.txt + summary   # M2.5
    │   ├── 03_loss_curve.png                  # M3 loss curve
    │   ├── 03_loss_data.csv                   # M3 raw loss data
    │   ├── 03_m3_training_summary.md          # M3 full results
    │   └── 04_finetuned_inference_output.txt  # M3 inference verification
    ├── notes/
    │   ├── installation.md                    # Setup guide
    │   ├── 03_finetuning_setup.md             # 12 engineering pitfalls postmortem
    │   └── 04_m3_training_results.md          # M3 engineering reflections
    ├── constraints.txt
    └── README.md

## 📈 实验结果

### M3: SO-101 Full Fine-tuning

| 指标 | 值 |
|---|---|
| 训练步数 | **20,000** |
| Dataset | SO-101 PickPlace (50 ep, 11,939 frames) |
| Initial loss | ~0.55 |
| **Final loss** | **0.086** |
| Final grad norm | ~1.4 |
| 总耗时 | **1h 11min** (RTX 4090 D) |
| Throughput | ~4-5 steps/sec |

![Loss Curve](results/03_loss_curve.png)

### M4: LIBERO Environment & Dataset Preparation

| 指标 | 值 |
|---|---|
| 仿真环境 | Mujoco 3.8 + robosuite + LIBERO via EGL headless |
| 数据集 | HuggingFaceVLA/libero (LeRobot v3 format) |
| Total episodes | **1,693** |
| Total frames | **273,465** |
| Total tasks | **40** (4 suites × 10) |
| Robot | Franka Panda (7-DoF EEF delta action) |
| Image resolution | 256×256 RGB × 2 cameras (agentview + eye_in_hand) |
| 数据盘扩容 | 50GB → 145GB（适配 ~100GB Arrow cache） |
| 解决的工程问题 | **5 个新坑**：CMake 4.x、hf-mirror 限流、xethub S3 503、nested dir 污染、磁盘满伪装多种错误 |

📄 完整摘要: [results/03_m3_training_summary.md](results/03_m3_training_summary.md)
🔧 工程反思: [notes/04_m3_training_results.md](notes/04_m3_training_results.md)
🐛 全部 12 个踩坑记录: [notes/03_finetuning_setup.md](notes/03_finetuning_setup.md)

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
