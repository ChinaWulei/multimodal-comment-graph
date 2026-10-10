# Qwen 直接分类对照

检验评论微调后的 Qwen 是否可以直接完成 Fakeddit 三分类，并与现有检测管线比较。
不训练任何模型，不改动原 MFCL、生成器权重或已经冻结的 A–E 实验。

## 四个入口

| 名称 | 权重 | 输入 |
|---|---|---|
| base_mm | 本地原始 Qwen2.5-VL-3B-Instruct | 新闻图文 |
| base_real | 同上 | 新闻图文＋真实评论 |
| adapter_mm | 同一基础模型＋结构评论 LoRA | 新闻图文 |
| adapter_real | 同上 | 新闻图文＋真实评论 |

每组只允许输出 0/1/2；不向模型传入当前新闻标签、subreddit 字段或数据文件路径。
类别定义：0=True，1=Fake with true text，2=Fake with false text。
依据已有 `sample_check/report.md` 的 Fakeddit 论文与原始数据交叉审计，准备阶段再次验证本地 2/3/6-way 标签对应关系。
Fakeddit 标签来自远程监督，类别定义不等于每条样本均经过人工事实核查。
原论文 3.3 节明确以“准确引用宣传海报文字”解释第二类。核对本地完整训练 TSV 后，数字 1 对应 propagandaposters；
六分类的 5 同时出现于三分类的 1 和 2，因此不能从六分类直接推导三分类。
定义来源：https://arxiv.org/html/1911.03854v2#S3.SS3 。源社区字段只用于标签定义审计，不进入模型输入。

## 固定数据与公平性

- 复用 `grounded_structure_20261010` 的 77 条验证新闻，44 条作为主比较。
- 排除的 33 条曾参与生成器检查点选择；全部验证集都已有探索历史，不称为盲测。
- 8 条训练新闻只用于格式、真实图片加载和运行检查；不依据其准确率改提示词。
- 直接复用 MFCL 数据集类的真实节点选择逻辑（最多 16 条），保留同样的顺序、父节点和原始正文。
- 文字裁剪到 MFCL BERT 实际可见的源文本范围（标题 128、评论 96 token，包含特殊 token 预算），记录裁剪情况。
- Qwen 使用同一图片的完整画面和自身视觉预处理，MFCL 使用 224 中心裁剪。二者不是完全等输入分辨率、等计算量的架构消融。
- Qwen 基础版与 Adapter 版使用完全相同的 Processor、提示词、4-bit NF4 量化和贪心解码。

## 两个预先定义的指标口径

主口径：严格解析模型生成的单个数字。非法输出保留为 -1 并计错，不丢弃样本、不人工改答案。

辅助口径：同一次前向计算中，取首输出位置对单 token `0`、`1`、`2` 的得分，选择最大者。
这是受限类别选择诊断，不是经过校准的置信度，不替代主指标，也不因看到结果才增加。
它用于区分评论 Adapter 的格式惯性与分类信息。四组统一报告两种口径。

模型权重、提示词、代码、参考结果、输入图像和样本集合均通过 SHA-256 固定。
完成一条新闻即保存，可以断点恢复。恢复时来源变化会拒绝继续；修改实验请换输出目录。

## 运行

在项目根目录及 `multimodal-fakenews` 环境：

```powershell
python .\comment_graph\qwen_direct\test_protocol.py
python .\comment_graph\qwen_direct\run.py --stage prepare
python .\comment_graph\qwen_direct\run.py --stage smoke
python .\comment_graph\qwen_direct\run.py --stage full
python .\comment_graph\qwen_direct\audit_results.py
```

`full` 会校验/复用 smoke，再串行运行四组各 77 条预测并汇总；默认只准备，不意外启动推理。
也可以执行单组（先 prepare）：

```powershell
python .\comment_graph\qwen_direct\run.py --stage evaluate --arm base_mm
python .\comment_graph\qwen_direct\run.py --stage evaluate --arm base_real
python .\comment_graph\qwen_direct\run.py --stage evaluate --arm adapter_mm
python .\comment_graph\qwen_direct\run.py --stage evaluate --arm adapter_real
python .\comment_graph\qwen_direct\run.py --stage summarize
```

## 结果

默认目录 `comment_graph/qwen_direct/outputs/validation_20261010_v2/`。
初版 `validation_20261010/` 仅保留开发 smoke 记录；修复 JSON 协议重载时的 tuple/list 比较后，在 v2 重跑，未修改任何评估提示词。

- `protocol.json`：预先固定的方案、参数和来源哈希。
- `validation_inputs.jsonl`、`validation_targets.jsonl`：无标签输入与独立评估答案。
- `input_audit.json`：图像哈希、所选评论 ID、文字裁剪情况。
- `validation/<arm>.jsonl`：每条新闻的原始输出、预测、辅助得分、耗时和图像 token 网格。
- `comparison.json`：44/77 条的 Accuracy、Macro-F1、Weighted-F1、每类指标及含非法输出列的混淆矩阵。
- `report.md`：四组与既有 A_real、D_observed_root、E_observed_parent 的验证比较。
- `status.json`、`logs/`：进度与错误。

Qwen 为确定性单次评估；检测管线为三个训练种子的均值和标准差，不制造伪独立重复。
这不是分类专用 Qwen 微调实验，也不能据此证明分类微调后的性能上限。
MFCL 检测管线使用了训练集类别标签，Qwen 的评论 Adapter 不以类别标签为训练目标；本次不是等监督量比较。
公开 Fakeddit 与 Qwen 预训练数据是否重叠未知。
没有测试集入口，不绕过原 A–E 实验门槛。
