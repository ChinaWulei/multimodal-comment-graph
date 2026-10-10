# Partial Comment Observation Generator

该目录实现以下任务：

```text
新闻图片 + 新闻文本 + 部分随机观察到的真实顶层评论
                         ↓
              Qwen2.5-VL Comment Generator
                         ↓
                 一条潜在用户评论
```

评论文件没有时间戳。本实验只称为 `Partial Comment Observation`，不解释为 early-stage、未来评论预测或时间演化。

## Label leakage 边界

标签只保存在每条记录的 `metadata.labels`，供后续检测器监督和统计使用。模型输入位于独立的 `model_input`，只允许：

- `image_path`
- `news_text`
- `observed_comments`

训练 Collator 与生成脚本均拒绝其他 `model_input` 字段。Prompt 渲染函数只接收 `model_input`，不接收 metadata。旧目录 `comment_generator/outputs/qwen2_5_vl_3b_comment_lora` 是此前的标签条件实验，存在本任务定义下的标签泄漏，不能作为本实验 Adapter 使用。

## 实际数据结果

- 本地图片：172张，Pillow验证全部有效。
- 图片与新闻ID匹配：172条，全部属于官方 train。
- 有任意有效评论：126条新闻。
- 排除7条明显的 AutoModerator / bot 自动回复后，有有效顶层评论：119条新闻。
- 至少2条有效去重顶层评论：82条新闻。
- 按 `observed_ratio=0.3`、最多3条 observed comments 构建800条 SFT 样本。
- validate/test：当前均为0，因为本地没有对应官方划分图片。

## 目录

```text
partial_observation/
├── config/generator.yaml
├── prompts/comment_generation.txt
├── scripts/
│   ├── inspect_data.py
│   ├── build_manifest.py
│   ├── build_comment_trees.py
│   ├── validate_comment_trees.py
│   ├── distill_comment_trees.py
│   ├── build_sft_dataset.py
│   ├── validate_dataset.py
│   ├── train_generator.py
│   ├── generate_comments.py
│   ├── evaluate_generator.py
│   └── build_detector_input.py
├── data/
├── outputs/
└── reports/
```

## 图文相关性过滤与评论树提炼

新版正式数据入口不再直接使用杂乱的顶层评论。处理单位是一棵顶层回复子树：

```text
原始评论 + parent_id
→ JSON 评论树
→ 图片 + 新闻 + 评论树送入冻结的多模态 Teacher
→ 过滤无关线程
→ 每棵相关线程提炼为一句可独立发布的评论
→ 划分 observed / target
→ 构建 Generator SFT 数据
```

标签不会进入评论树过滤器或提炼 Prompt。完整处理使用 `generator_tree.yaml`：

```powershell
python .\comment_generator\partial_observation\scripts\build_comment_trees.py `
  --config .\comment_generator\partial_observation\config\generator_tree.yaml

python .\comment_generator\partial_observation\scripts\validate_comment_trees.py `
  --config .\comment_generator\partial_observation\config\generator_tree.yaml

python .\comment_generator\partial_observation\scripts\distill_comment_trees.py `
  --config .\comment_generator\partial_observation\config\generator_tree.yaml `
  --local-files-only

python .\comment_generator\partial_observation\scripts\build_sft_dataset.py `
  --config .\comment_generator\partial_observation\config\generator_tree.yaml

python .\comment_generator\partial_observation\scripts\validate_dataset.py `
  --config .\comment_generator\partial_observation\config\generator_tree.yaml `
  --smoke-limit 20
```

`distill_comment_trees.py` 支持断点继续：

```powershell
python .\comment_generator\partial_observation\scripts\distill_comment_trees.py `
  --config .\comment_generator\partial_observation\config\generator_tree.yaml `
  --resume `
  --local-files-only
```

完整输出包括：

- `raw_comment_trees.jsonl`：保留父子关系的原始树。
- `thread_distillation_results.jsonl`：每棵树的相关性、分数、类型、提炼评论和来源 ID。
- `distilled_comment_pool.jsonl`：通过阈值并去重后的新评论池。
- `tree_train.jsonl`、`tree_val.jsonl`、`tree_test.jsonl`：新版 SFT 数据。

本地 Qwen2.5-VL-3B 当前被配置为可替换的 Teacher。这适合流程验证，但正式研究应记录自蒸馏偏差，条件允许时改用独立、更强的 Teacher。

完整树数据目前只有64条SFT样本，因此新版训练流程 smoke 使用10个 optimizer steps，并在第5/10步验证：

```powershell
python .\comment_generator\partial_observation\scripts\train_generator.py `
  --config .\comment_generator\partial_observation\config\generator_tree.yaml `
  --smoke-train `
  --local-files-only
```

## 从原始数据重新构建

在项目根目录运行：

```powershell
conda activate multimodal-fakenews

python .\comment_generator\partial_observation\scripts\inspect_data.py `
  --config .\comment_generator\partial_observation\config\generator.yaml

python .\comment_generator\partial_observation\scripts\build_manifest.py `
  --config .\comment_generator\partial_observation\config\generator.yaml

python .\comment_generator\partial_observation\scripts\build_sft_dataset.py `
  --config .\comment_generator\partial_observation\config\generator.yaml

python .\comment_generator\partial_observation\scripts\validate_dataset.py `
  --config .\comment_generator\partial_observation\config\generator.yaml `
  --smoke-limit 20 `
  --preview-count 10
```

主要输出：

- `data/manifest.csv`：172张图片逐条筛选结果。
- `data/comment_pool.jsonl`：清洗、去重后的顶层评论及原始元数据。
- `data/news_contexts.jsonl`：新闻级生成输入，不含 target 正文。
- `data/gold_targets.jsonl`：独立评价用真实 target，不供生成脚本读取。
- `data/train.jsonl`、`val.jsonl`、`test.jsonl`：一条 target comment 一个样本。
- `reports/data_report.md`：统计报告。
- `reports/sample_preview.md`：10条实际 Prompt 和 Target。

## 20条数据读取 smoke test

该命令不加载模型、不训练：

```powershell
python .\comment_generator\partial_observation\scripts\train_generator.py `
  --config .\comment_generator\partial_observation\config\generator.yaml `
  --check-data-only `
  --processor-smoke-test `
  --limit 20
```

## 50-step QLoRA smoke training

由于当前官方 validate 没有本地图片，脚本会固定按 `news_id` 从官方 train 中留出 8 条新闻，作为内部流程验证集。这个分数不能作为官方验证性能。

```powershell
python .\comment_generator\partial_observation\scripts\train_generator.py `
  --config .\comment_generator\partial_observation\config\generator.yaml `
  --smoke-train `
  --local-files-only
```

Smoke Adapter 保存到：

```text
comment_generator/partial_observation/outputs/qwen2_5_vl_3b_partial_comment_smoke_lora/
```

训练报告位于 `reports/smoke_train_report.md`。内部留出数据记录为 `data/smoke_train.jsonl` 和 `data/smoke_val.jsonl`，不会覆盖正式数据文件。

## 正式 QLoRA 训练

必须先完整生成 `distilled_comment_pool.jsonl` 和 `tree_train.jsonl`。训练脚本会校验数据来源，拒绝把旧版顶层评论数据当成新版正式数据。当前没有官方 validate 样本，运行后只有训练损失，不能报告验证性能。确认后使用：

```powershell
python .\comment_generator\partial_observation\scripts\train_generator.py `
  --config .\comment_generator\partial_observation\config\generator_full_train.yaml `
  --local-files-only
```

输出 Adapter 默认保存到：

```text
comment_generator/partial_observation/outputs/qwen2_5_vl_3b_partial_comment_full_lora/
```

## 生成评论

由于目前只有 train，定性检查必须显式写 `--split train`，程序不会把 train 静默伪装成 validation：

```powershell
python .\comment_generator\partial_observation\scripts\generate_comments.py `
  --config .\comment_generator\partial_observation\config\generator_full_train.yaml `
  --adapter-dir .\comment_generator\partial_observation\outputs\qwen2_5_vl_3b_partial_comment_full_lora `
  --split train `
  --limit 10 `
  --comments-per-news 5 `
  --local-files-only
```

## 评价与 Detector 输入

```powershell
python .\comment_generator\partial_observation\scripts\evaluate_generator.py `
  --config .\comment_generator\partial_observation\config\generator.yaml `
  --generated-jsonl .\comment_generator\partial_observation\outputs\generated_comments.jsonl

python .\comment_generator\partial_observation\scripts\build_detector_input.py `
  --config .\comment_generator\partial_observation\config\generator.yaml `
  --generated-jsonl .\comment_generator\partial_observation\outputs\generated_comments.jsonl
```

`--compute-bertscore` 是可选项，需要另行安装 BERTScore 依赖并准备对应模型。语义相似度仅作为辅助指标。

## 结构感知评论和传播图生成

`generator_structure.yaml` 使用已恢复的 Fakeddit `parent_id` 作为真实回复边监督。每个 SFT target 都是严格 JSON：

```json
{"parent_node_index": 2, "content": "A natural reply comment."}
```

模型输入只有新闻图片、新闻文本和可见评论节点；真实性标签不会传入模型。评论文件无时间戳，节点的父先于子顺序只用于保证结构可见，不能解释为真实时间传播。

```powershell
python .\comment_generator\partial_observation\scripts\build_structured_sft.py `
  --config .\comment_generator\partial_observation\config\generator_structure.yaml

python .\comment_generator\partial_observation\scripts\validate_structured_sft.py `
  --config .\comment_generator\partial_observation\config\generator_structure.yaml

python .\comment_generator\partial_observation\scripts\train_structured_generator.py `
  --config .\comment_generator\partial_observation\config\generator_structure.yaml `
  --check-data-only --processor-smoke-test --local-files-only

python .\comment_generator\partial_observation\scripts\train_structured_generator.py `
  --config .\comment_generator\partial_observation\config\generator_structure.yaml `
  --smoke-train --local-files-only
```

训练 smoke 的 Adapter 用于流程验证。生成时，模型逐条输出 JSON；通过 JSON 格式、父节点范围、重复评论和最大树深度校验后，新评论和 `edge_type=1` 的生成边才会被加入图：

```powershell
python .\comment_generator\partial_observation\scripts\generate_structured_comments.py `
  --config .\comment_generator\partial_observation\config\generator_structure.yaml `
  --adapter-dir .\comment_generator\partial_observation\outputs\qwen2_5_vl_3b_structure_comment_smoke_lora `
  --limit 3 --nodes-per-graph 5 `
  --output-jsonl .\comment_generator\partial_observation\outputs\generated_structured_graphs.jsonl `
  --local-files-only
```

在当前本地图片全部属于 Fakeddit 官方训练划分的前提下，该流程只能验证代码和生成格式，不能报告官方验证或测试性能。待补齐官方 validate/test 图片后，再把生成图接入检测器并比较 root-only 与 structure-aware 图。

## 官方划分的结构图实验子集

`plan_official_structure_subset.py` 从 Fakeddit 官方 train / validate / test TSV
中选择同时具有有效文本、图片 URL、顶层评论和回复评论的新闻；不会重划分新闻。
当前配置从每个原始划分选取 `300 / 80 / 80` 条，并按 `3_way_label` 均衡抽样，
以避免很小的少数类让 Macro-F1 失真。标签只用于抽样和最终检测评价，绝不传给评论生成器。

先生成或更新候选清单（该步骤会流式读取一次 `all_comments.tsv`）：

```powershell
python .\comment_generator\partial_observation\scripts\plan_official_structure_subset.py `
  --config .\comment_generator\partial_observation\config\generator_structure.yaml
```

图片下载默认不会启动；下面的命令仅对每个指定划分的前 10 条执行限速测试。下载器
会校验图片，写入 SQLite 状态，遇到 404/403 跳过，遇到 429 按 `Retry-After`
暂停，且可以安全续跑：

```powershell
python .\comment_generator\partial_observation\scripts\download_structure_subset_images.py `
  --config .\comment_generator\partial_observation\config\generator_structure.yaml `
  --splits validate test --limit 10 --download
```

成功图片会保存到 `data/official_subset_images/<split>/`。随后构建与 pilot 数据隔离的
官方 manifest 和真实评论树：

```powershell
python .\comment_generator\partial_observation\scripts\build_official_subset_manifest.py `
  --config .\comment_generator\partial_observation\config\generator_structure.yaml
```

`official_structure_candidate_comment_trees.jsonl` 已是一次性扫描原始评论文件得到的
候选缓存。它含真实树，但尚未证明图片存在，因此不能直接给模型。图片下载完成后，
用下面的 join 命令仅保留 `image_valid=true` 的新闻：

```powershell
python .\comment_generator\partial_observation\scripts\join_official_candidate_trees.py
```

之后再用独立的官方配置做图文相关性过滤。`--max-threads-per-news 2` 是固定协议：
每条新闻使用按赞同票排序的前两个真实根讨论；它不是按标签选择，且三个划分使用相同规则。

```powershell
python .\comment_generator\partial_observation\scripts\distill_comment_trees.py `
  --config .\comment_generator\partial_observation\config\generator_structure_official.yaml `
  --max-threads-per-news 2 `
  --local-files-only
```

此处的 Qwen-VL 只根据图文和评论树决定“是否相关”，不接收真实性标签。它支持
`--resume`，因此可安全中断和续跑。后续结构化 SFT 数据也必须使用
`generator_structure_official.yaml`，以保证 pilot 的 172 张训练图不混入正式结果。

完成图片下载与筛选后，按官方划分分别构建并验证结构化 SFT 文件：

```powershell
python .\comment_generator\partial_observation\scripts\build_structured_sft.py `
  --config .\comment_generator\partial_observation\config\generator_structure_official.yaml `
  --split train `
  --output-jsonl .\comment_generator\partial_observation\data\official_subset\structure_train.jsonl

python .\comment_generator\partial_observation\scripts\build_structured_sft.py `
  --config .\comment_generator\partial_observation\config\generator_structure_official.yaml `
  --split validate `
  --output-jsonl .\comment_generator\partial_observation\data\official_subset\structure_validate.jsonl

python .\comment_generator\partial_observation\scripts\build_structured_sft.py `
  --config .\comment_generator\partial_observation\config\generator_structure_official.yaml `
  --split test `
  --output-jsonl .\comment_generator\partial_observation\data\official_subset\structure_test.jsonl

python .\comment_generator\partial_observation\scripts\validate_structured_sft.py `
  --config .\comment_generator\partial_observation\config\generator_structure_official.yaml `
  --input-jsonl .\comment_generator\partial_observation\data\official_subset\structure_train.jsonl `
  --expected-official-split train
```

正式 LoRA 仅优化 `structure_train.jsonl`，以 `structure_validate.jsonl` 选择 checkpoint；
`structure_test.jsonl` 只用于训练完成后的离线生成与最终检测评估，绝不参与反向传播：

```powershell
python .\comment_generator\partial_observation\scripts\train_structured_generator.py `
  --config .\comment_generator\partial_observation\config\generator_structure_official.yaml `
  --formal-train --local-files-only
```

验证损失选出的正式 Adapter 保存为：
`outputs/qwen2_5_vl_3b_structure_comment_official_bestval_lora/`。

训练完成后，使用下列脚本对三个官方划分分别生成每新闻最多五个带父节点的评论图。
生成器输入只包含图片、新闻文本和可见真实评论，不含真实性标签；该脚本不会进行训练：

```powershell
.\comment_generator\partial_observation\scripts\generate_official_structured_graphs.ps1 `
  -Python python `
  -NodesPerGraph 5
```
