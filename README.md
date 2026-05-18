# VLA Portfolio: SmolVLA × LIBERO Reproduction

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
- [ ] **M3: SO-101 数据集微调** — 20k 步正式训练（in progress）
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

## 🛠 技术栈

- **VLA 模型**: SmolVLA-450M ([lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base))
- **框架**: LeRobot (main branch, ≥ 0.5.2)
- **GPU**: NVIDIA RTX 4090 D (24GB)
- **环境**: AutoDL 云服务器，PyTorch 2.8.0 + CUDA 12.8 + Python 3.12

## 📁 仓库结构

    .
    ├── scripts/                                  # 训练 / 推理 / 评估脚本
    │   ├── 01_test_smolvla_inference.py         # M2: 推理验证
    │   └── 02a_smoke_test.sh                    # M2.5: Fine-tuning pipeline smoke test
    ├── results/                                  # 实验输出
    │   ├── 01_inference_output.txt              # M2: 推理输出快照
    │   ├── 02a_smoke_test_log.txt               # M2.5: smoke test 完整日志
    │   └── 02a_smoke_test_summary.md            # M2.5: 训练结果摘要
    ├── notes/                                    # 学习笔记 / 工程实践
    │   ├── installation.md                      # 详细安装流程 + 常见问题
    │   └── 03_finetuning_setup.md               # M3: Fine-tuning 7 大踩坑详解
    ├── constraints.txt                           # 关键依赖版本锁定
    └── README.md

## 🚀 快速开始

### 环境要求
- NVIDIA GPU，Driver ≥ 580 (CUDA 12.8+)
- Python 3.12
- 50GB 磁盘 (模型 + 数据集 + checkpoint)

### 安装

详见 [notes/installation.md](notes/installation.md)，核心步骤：

    # 1. 创建 conda 环境
    conda create -y -n lerobot python=3.12
    conda activate lerobot
    conda install -y ffmpeg=7.1.1 -c conda-forge

    # 2. 安装 PyTorch 2.8 + CUDA 12.8
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128

    # 3. 安装 LeRobot + SmolVLA + datasets extras
    git clone --depth 1 https://github.com/huggingface/lerobot.git
    cd lerobot
    pip install -e ".[smolvla,dataset]" -c ../constraints.txt

    # 4. 装视频解码器(pyav 替代 torchcodec)
    pip install av

### 运行推理 Hello World

    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/01_test_smolvla_inference.py

### 运行 Smoke Test 微调

    bash scripts/02a_smoke_test.sh

## 📈 实验结果

### M2: SmolVLA 推理验证

| 指标 | 值 |
|---|---|
| 模型参数 | 450.0M |
| 输入形状 | state(6) + 3× image(3,256,256) + language |
| 输出动作维度 | 6 |
| 推理设备 | RTX 4090 D (CUDA 12.8) |

完整输出: [results/01_inference_output.txt](results/01_inference_output.txt)

### M2.5: Fine-tuning Pipeline Validation

| 指标 | 值 |
|---|---|
| 数据集 | SO-101 PickPlace (50 ep, 11,939 frames) |
| 训练步数 | 500 (smoke test) |
| 初始 loss | 0.549 |
| 最终 loss | **0.196** |
| Grad norm 趋势 | 8.08 → 2.74 (稳定收敛) |
| 单步耗时 | ~0.15s (4090 D) |
| Throughput | ~4-5 steps/sec |
| 总耗时 | 2 minutes |

完整摘要: [results/02a_smoke_test_summary.md](results/02a_smoke_test_summary.md)
踩坑详解: [notes/03_finetuning_setup.md](notes/03_finetuning_setup.md)

## 📚 参考资料

- [SmolVLA Paper (arXiv:2506.01844)](https://arxiv.org/abs/2506.01844)
- [SmolVLA Blog](https://huggingface.co/blog/smolvla)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LIBERO Benchmark](https://libero-project.github.io/)

## 📝 License

MIT
