# Qwen2.5-VL Fakeddit 评论生成微调

该目录与 MFCL baseline 完全隔离。模型输入为新闻标题、真实图片和 Fakeddit 标签条件，监督目标是一条真实 Reddit 评论。

默认模型为 `Qwen/Qwen2.5-VL-3B-Instruct`，训练方式为 4-bit QLoRA。视觉编码器保持冻结，LoRA 只适配语言模型的注意力和 MLP 投影层。Prompt 明确把真实性标签视为数据集条件，不把标签等同于评论立场。

## 当前数据限制

`sample_check/` 中的 126 条可用新闻全部属于官方 train。程序严格保留原始 split，不会把同一新闻的评论随机拆到验证集。因此当前能完成数据检查、单步反向传播和小规模训练，但不会产生官方验证损失。正式实验前应补充官方 validate 图片并重新运行样本筛选和数据准备。

## 目录

```text
comment_generator/
├── config.yaml
├── common.py
├── prepare_data.py
├── assemble_local_model.py
├── train.py
├── generate.py
├── requirements.txt
├── data/                  # prepare_data.py 生成
└── outputs/               # 训练后生成
```

## 1. 准备监督微调数据

在项目根目录、`multimodal-fakenews` 环境中运行：

```powershell
conda activate multimodal-fakenews
python .\comment_generator\prepare_data.py --config .\comment_generator\config.yaml
```

默认按新闻 ID 固定随机采样，每条新闻最多保留 5 条评论。输出为：

- `data/train.jsonl`
- `data/validate.jsonl`
- `data/test.jsonl`
- `data/stats.json`

## 2. 不加载模型的数据检查

```powershell
python .\comment_generator\train.py `
  --config .\comment_generator\config.yaml `
  --check-data-only
```

该命令会检查字段、图片、重复样本以及 train/validate 新闻 ID 泄漏，不会联网或加载 Qwen。

## 3. 组装网页手动下载的模型分片

如果已经从模型网页下载两个 `model-0000*-of-00002.safetensors`，运行：

```powershell
python .\comment_generator\assemble_local_model.py `
  --weights-dir D:\Downloads
```

程序会核对全部张量，并根据分片的实际内容重建索引。默认优先在同一磁盘上创建硬链接；如果当前 Windows 权限不允许，则自动复制约 7GB 权重。其他配置和 Tokenizer 从已有 Hugging Face 缓存复制到 `comment_generator/models/Qwen2.5-VL-3B-Instruct/`。

## 4. 单步 QLoRA 冒烟测试

当前配置默认读取已经组装好的本地模型，不再联网：

```powershell
python .\comment_generator\train.py `
  --config .\comment_generator\config.yaml `
  --smoke-test
```

也可以显式指定其他本地模型目录：

```powershell
python .\comment_generator\train.py `
  --config .\comment_generator\config.yaml `
  --model-path D:\models\Qwen2.5-VL-3B-Instruct `
  --local-files-only `
  --smoke-test
```

冒烟测试只使用最多两条样本，运行一次前向、反向和优化器更新，结果保存在 `outputs/smoke_test/`。

## 5. 正式微调

```powershell
python .\comment_generator\train.py --config .\comment_generator\config.yaml
```

默认配置：LoRA rank 16、alpha 32、dropout 0.05、学习率 `2e-4`、2 epochs、batch size 1、梯度累积 8。RTX A5000 支持 BF16，程序会自动选择 BF16；其他显卡不支持时使用 FP16。

断点恢复：

```powershell
python .\comment_generator\train.py `
  --config .\comment_generator\config.yaml `
  --resume-from-checkpoint .\comment_generator\outputs\qwen2_5_vl_3b_comment_lora\checkpoint-50
```

## 6. 枚举候选标签生成评论

推理不使用真实标签来选择条件。程序会对同一新闻依次枚举配置中的候选标签 `0, 1, 2`，每个条件默认生成 5 条评论。

```powershell
python .\comment_generator\generate.py `
  --config .\comment_generator\config.yaml `
  --adapter-dir .\comment_generator\outputs\qwen2_5_vl_3b_comment_lora `
  --output-jsonl .\comment_generator\outputs\generated_comments.jsonl `
  --limit 10
```

由于当前没有官方 validate 样本，推理脚本会明确提示并回退到 train，只用于定性检查。

## 显存调整

当前机器的 RTX A5000 有 24GB 显存，默认 3B QLoRA 配置应可运行。如果显存不足，优先降低 `model.max_pixels` 或 `data.max_sequence_length`，不要先取消 4-bit。`attn_implementation` 默认使用环境自带的 PyTorch SDPA，不要求额外安装 FlashAttention。
