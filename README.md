# MFCL：生成评论与结构增强的多模态假新闻检测

本仓库在 MFCL 代码基础上开展研究：根据新闻图文及已有真实评论生成补充评论，再将真实评论和生成评论组成图，用于新闻真实性分类。

当前主线是 **Fakeddit 三分类任务**，重点回答：

1. 在已有真实评论之外，补充生成评论是否有帮助？
2. 对相同的生成评论，预测连接位置是否优于全部连接新闻根节点？

这是持续开发中的科研仓库，不是已经完成的论文方法发布。原 MFCL、当前扩展方法和历史实验不能混称为同一个模型。

> 最近核对：2026-10-10。当前实验尚未证明预测连接稳定优于根节点连接。以下指标属于探索性验证结果，不是最终测试结果。

## 1. 新加入项目从哪里开始

建议先理解流程、核对数据和环境，再运行小规模检查，不要直接启动完整训练。

| 内容 | 入口 |
|---|---|
| 当前五组公平对照及测试纪律 | [GROUNDED_STRUCTURE.md](comment_graph/GROUNDED_STRUCTURE.md) |
| 分类训练入口 | [train.py](comment_graph/train.py) |
| 检测模型与融合 | [model.py](comment_graph/model.py) |
| Fakeddit 新闻、真实树和生成图加载 | [fakeddit_structure_dataset.py](comment_graph/fakeddit_structure_dataset.py) |
| 区分真实边与生成边的 GNN | [typed_graph.py](comment_graph/typed_graph.py) |
| 结构化评论生成 | [generate_structured_comments.py](comment_generator/partial_observation/scripts/generate_structured_comments.py) |
| 连接打乱与顺序敏感性检查 | [connection_diagnostics/README.md](comment_graph/connection_diagnostics/README.md) |
| 固定正文后的父节点诊断 | [attachment_consistency/probe.py](comment_graph/attachment_consistency/probe.py) |
| 原版说明及早期独立在线编码入口 | [MFCL_LEGACY_README.md](docs/MFCL_LEGACY_README.md) |

部分子目录 README 保留了“仅172张图片”“没有验证图片”等历史信息，不代表当前主线。具体数据规模和配置以对应实验协议为准。

## 2. 算法流程：先生成讨论，再判断新闻

整个流程包含两个分工不同的模型：生成器补充讨论，检测器结合新闻和讨论做分类。

```text
新闻标题、真实图片、已有真实评论及回复关系
                        │
                        ▼
             已微调的 Qwen2.5-VL-3B
                        │
               生成评论正文＋父节点编号
                        │
                        ▼
       真实评论及真实边 ＋ 生成评论及预测增强边
                        │
              共享文本编码器 → GNN → 图向量
                                             │
新闻文本、图片 → 图文编码与特征处理 → 图文向量 ─┤
                                             ▼
                                  拼接 → 分类器 → 三类分数
```

### 评论生成器

- 基础模型： [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。
- 输入：实际新闻图片、新闻文本、可见真实评论及节点关系。
- 输出：一条评论和它应该连接的父节点；编号0代表新闻根节点。
- 微调监督来自真实评论及真实 `parent_id`，只对助手目标 JSON 计算自回归损失。
- 当前主线的生成输入**不包含新闻真实类别**。早期标签条件生成器不能直接用于这条主线。

输出格式示意，不代表真实样本：

```json
{"parent_node_index": 2, "content": "A plausible reply to the discussion."}
```

当前比较两种生成策略：

- `iterative`：已生成评论进入后续输入，允许回复生成评论。
- `observed_only`：每次只看原始新闻图文和真实评论，父节点只能是新闻或真实评论，不把生成内容反馈给下一次生成。

每条新闻目标生成5条，每个位置最多尝试3次。格式错误、重复和非法父节点如实记录，不伪造补齐。生成结果先保存，再由检测器读取；两者不联合训练。

### 真假新闻检测器

新闻、真实评论和生成评论共享文本编码器。当前主线使用 BERT、ResNet-50、两层关系型 GNN（`typed_rgcn`）、均值池化和拼接分类。

- 节点类型：`0=news`、`1=real_comment`、`2=generated_comment`。
- 边类型：`0=真实回复边`、`1=模型预测增强边`。
- 消息传递使用双向边，但不表示真实世界发生了反向回复。

当前配置下：

```text
h_mm    = 1536维新闻图文向量
h_graph =  768维评论图向量
h_final = concat(h_mm, h_graph) = 2304维
logits  = classifier(h_final)   = 3类分数
loss    = CrossEntropyLoss(logits, 新闻真实类别)
```

图文编码在运行时执行，但当前冻结 BERT 和 ResNet 主干，不等于全参数微调。GNN 和分类相关可训练参数通过训练集分类损失更新，Qwen 不参与此阶段反向传播。

这条主线复用了 MFCL 的部分组件，**不是 MFCL 论文全部训练目标的完整复现**，不能仅凭名称与论文原始指标直接比较。

## 3. 数据集与划分

| 数据集 | 本项目中的用途 | 注意事项 |
|---|---|---|
| Fakeddit | 当前英文图文、评论、三分类主线 | 有真实回复关系；保留官方划分 |
| CHECKED | 历史中文图文＋评论实验 | 本地评论格式未提供真实父节点，不声称有真实回复树 |
| Weibo / PHEME | 原 MFCL 代码及历史复现实验 | 数据接口、标签和特征不同，不与Fakeddit混用 |

新闻使用 `id`、`clean_title`、图片和原始标签。评论通过 `submission_id` 匹配新闻，通过 `parent_id` 建树；处理 Reddit 的 `t1_`、`t3_` 前缀。

保留原始二、三、六分类字段，当前使用 `3_way_label`：

- `0`：True。
- `1`：Fake with true text。
- `2`：Fake with false text。

不要自行把三分类改成二分类，或从六分类推导三分类。语义见 [Fakeddit 论文](https://arxiv.org/abs/1911.03854)；数据集标签不等于每条新闻都经过独立人工事实核查。

下载来源：[Fakeddit 官方仓库](https://github.com/entitize/Fakeddit)、[CHECKED 官方仓库](https://github.com/cyang03/CHECKED)。使用与再分发需遵守来源要求。

### 当前实验规模

| 用途 | 新闻数 | 说明 |
|---|---:|---|
| 检测器训练 | 292 | 来自官方 train |
| 分类主要验证 | 44 | 从77条中排除参与生成器检查点选择的33条 |
| 完整辅助验证 | 77 | 用冻结检查点评估，不再次选模型 |
| 本轮测试 | 未执行 | 尚未通过预设验证门槛 |

生成器与检测器是两个训练阶段。当前结构化生成器使用167条训练新闻构造1,829条 SFT 样本，使用33条验证新闻选择检查点。不要把1,829条评论样本写成独立新闻数。

44条主要验证新闻已参与方法探索，不是新盲测。历史上看过的测试结果也不能重新称为完全未见的最终证据。

## 4. 目录结构

```text
KBS-MFCL/
├── Weibo/                         # 原MFCL相关代码、早期在线编码入口
├── comment_generator/
│   ├── partial_observation/
│   │   ├── config/                # 当前结构化生成配置
│   │   ├── prompts/               # 生成与预处理提示
│   │   ├── scripts/               # 树构建、SFT、生成及校验
│   │   ├── data/                  # 本地处理数据
│   │   └── outputs/               # 本地Adapter等产物
│   ├── checked/                   # 历史CHECKED生成实验
│   └── models/                    # 本地Qwen权重
├── comment_graph/
│   ├── train.py                   # 检测器训练与评估
│   ├── model.py                   # 图文、评论图与分类器
│   ├── run_grounded_structure.py  # 当前A–E对照编排
│   ├── connection_diagnostics/    # 连接打乱、顺序诊断
│   ├── attachment_consistency/    # 固定正文诊断、数据增强预备实验
│   ├── connection_followup/       # 额外根连接对照
│   ├── qwen_direct/               # Qwen直接分类对照
│   ├── outputs/                   # 检查点、预测、指标、日志
│   └── reports/                   # 历史报告与审核材料
├── checked_data/                  # CHECKED工具及本地数据
├── sample_check/                  # 早期图文评论匹配检查
├── emotion_analysis/              # 历史分支，当前主线未启用
├── third_party/                   # 外部项目，注意来源和许可
├── multimodal_only_samples/       # 本地Fakeddit新闻TSV
├── all_comments.tsv/              # 本地评论文件所在目录
├── images/                        # 本地图片，另有正式子集图片目录
└── docs/                          # 项目文档
```

历史的 MIMoE、检索证据、局部修正、动态跨模态图和质量门控不是本轮主线默认组件。代码存在不代表当前全部开启。

## 5. 环境准备

实际验证平台：Windows、PowerShell、NVIDIA RTX A5000 24GB。以下是本机环境记录，不是跨平台安装锁文件：

| 组件 | 本机版本 |
|---|---|
| Python | 3.10.21 |
| PyTorch / CUDA runtime | 2.11.0+cu128 / 12.8 |
| torchvision | 0.26.0+cu128 |
| torch-geometric | 2.8.0.post1 |
| transformers | 4.57.6 |
| peft / accelerate | 0.14.0 / 1.14.0 |
| bitsandbytes | 0.50.2 |

新机器先安装适配硬件的 PyTorch、torchvision 和 PyG，再安装其余依赖；不要把本机版本盲目复制到不兼容系统。

```powershell
# 已有项目环境时
conda activate multimodal-fakenews

# 已准备好适配本机的PyTorch、torchvision和PyG后
python -m pip install -r .\comment_generator\requirements.txt
python -m pip install munch networkx scikit-learn einops
python -m pip check

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

根目录 `requirements-fakeddit.txt` 属于早期在线编码入口，包含torch 1.13.1、transformers 4.36.2等旧约束。**不要直接安装进当前Qwen环境**；旧入口使用独立环境。

小规模实测：4-bit Qwen推理约3.5GiB峰值分配显存，冻结图文主干的分类训练约1.4GiB。微调、长上下文、大图片或解冻主干会使用更多显存，不把上述数字当作通用最低配置。

## 6. 数据和模型交接清单

只有GitHub代码不能完整复现实验。运行前应取得或按原流程重建：

1. 新闻TSV、评论、验证完整的图片，保留官方划分。
2. `comment_generator/partial_observation/data/official_subset/` 中的 `manifest.csv`、`raw_comment_trees.jsonl`、结构化SFT train/validate文件。
3. 同目录下 `generated_train_all_valid_images_observed_boundary.jsonl`、`generated_validate_all_valid_images_observed_boundary.jsonl`，用于当前主线初始化。
4. 固定smoke排除名单所用的历史记录：
   - `comment_graph/reports/generated_quality_annotation_train60.jsonl`
   - `comment_graph/reports/quality_expansion_20261008/fit.jsonl`
   - `comment_graph/reports/quality_expansion_20261008/reserved_audit.jsonl`

   这些只用于名单溯源，不代表启用了质量过滤器。含评论内容的文件不要直接公开上传。
5. 基础权重 `comment_generator/models/Qwen2.5-VL-3B-Instruct/`；结构化Adapter `comment_generator/partial_observation/outputs/qwen2_5_vl_3b_structure_comment_official_expanded_lora/`。
6. BERT tokenizer/model及ResNet-50预训练权重的本地缓存；当前配置使用本地模型加载。

主要配置：[生成器](comment_generator/partial_observation/config/generator_structure_official_expanded.yaml)、[检测器](comment_graph/config_fakeddit_structure_expanded_generateddepth.yaml)。

部分manifest、JSONL、检查点和协议包含绝对本地路径，诊断脚本也绑定历史输出目录。换机器时需要核对路径、模型来源和完整性，在新目录建立新协议；不要修改旧哈希绕过检查。

主线脚本校验固定的292/77/44条队列，不是任意数据集的一键入口。扩大规模需要更新协议及相关检查，不能直接删除断言。

## 7. 运行顺序

以下PowerShell命令均在项目根目录执行。数据和模型未齐全时不要启动完整训练。

### 第一步：代码级检查，不加载大模型

```powershell
python .\comment_graph\connection_diagnostics\test_protocol.py
python .\comment_graph\attachment_consistency\test_probe.py
python .\comment_graph\attachment_consistency\test_augmentation.py
```

这些检查通过不代表真实数据或正式训练已经通过。

### 第二步：当前A–E主线

```powershell
# 准备清单并固定协议；需要第6节的本地资产
python .\comment_graph\run_grounded_structure.py --stage prepare

# 8条实际新闻：生成、图校验、GNN前向/反向
python .\comment_graph\run_grounded_structure.py --stage smoke

# 确认smoke与资源需求后：完整生成、三种子的五组训练及辅助验证
python .\comment_graph\run_grounded_structure.py --stage full

# 仅整理已完成的实验
python .\comment_graph\run_grounded_structure.py --stage summarize
```

默认输出 `comment_graph/outputs/grounded_structure_20261010/`。主线入口支持 `--output-dir`；改变方案时使用新目录，不覆盖旧结果。

恢复会检查输入、代码、参数和模型来源。中断的某次分类训练可能从该种子重新开始，不等于任意优化器状态都能继续。

### 第三步：结构对照与最新诊断

以下入口依赖已完成的主线输出，不是原始数据准备入口。

```powershell
# 父节点打乱组：三种子分类训练及验证
python .\comment_graph\connection_diagnostics\experiment.py --stage full

# 固定评论正文，比较两种模型与三种展示顺序
python .\comment_graph\attachment_consistency\probe.py --stage run

# 24条未参与生成器SFT的训练新闻，做确认性诊断
python .\comment_graph\attachment_consistency\heldout_probe.py --stage run

# 原SFT分布审计及新数据准备，不启动训练
python .\comment_graph\attachment_consistency\audit_sft.py
python .\comment_graph\attachment_consistency\augmented_sft.py --stage prepare
```

`augmented_sft.py --stage smoke` 是独立的4步训练管线检查入口，不保存可用的新Adapter。截至本README核对时尚未执行，不能作为已完成改进报告。

### 测试集纪律

```powershell
# 仅在验证门槛通过、方案和检查点冻结后运行
python .\comment_graph\run_grounded_structure.py --stage test
```

门槛：E相比A和D，平均主要验证Macro-F1均提高至少1个百分点，且各有至少2/3种子提升。目前未通过。

不要删除门槛、反复用测试结果选择方案，或把旧测试队列重新称为全新盲测。

## 8. 对照设计与已有结果

| 组别 | 评论来源 | 生成评论连接 |
|---|---|---|
| A | 仅真实评论 | 无生成边 |
| B | 真实＋迭代生成 | 全部连接新闻 |
| C | 与B完全相同 | 模型预测 |
| D | 真实＋仅真实上下文生成 | 全部连接新闻 |
| E | 与D完全相同 | 模型预测新闻或真实评论 |
| H | 与D/E完全相同 | 新闻内部打乱父节点，不增加边 |

B/C、D/E分别复用相同内容，仅改变连接。H保持每个节点度数；如果所有生成评论原本指向同一父节点，该新闻就无法通过这种打乱改变，需报告此限制。

固定设置：种子42、13、24；最多20 epoch；早停5；batch size 8；学习率1e-4；真实评论上限16、生成评论上限5；`typed_rgcn + mean + concat`。无质量过滤、检索、额外注意力或新损失。

以下是44条主要验证新闻的三种子均值±样本标准差，标准差单位为百分点，**不是测试集指标**：

| 方法 | Accuracy | Macro-F1 |
|---|---:|---:|
| A：真实评论 | 81.82% ± 4.55 | 81.74% ± 4.18 |
| D：生成评论挂根节点 | 85.61% ± 1.31 | 85.39% ± 1.13 |
| E：生成评论预测连接 | 85.61% ± 3.47 | 84.94% ± 3.69 |
| H：生成评论打乱连接 | 84.09% ± 0.00 | 83.47% ± 0.20 |

E对H平均高1.47个百分点，但配对区间仍跨0；E没有优于D。当前可报告生成内容的初步收益，不能宣称结构连接的稳定优势。

本地来源：

- `comment_graph/outputs/grounded_structure_20261010/comparison.json`：A–E完整汇总。
- `comment_graph/connection_diagnostics/outputs/round1/comparison.json`：H对照汇总。
- 同目录 `result_audit.json`：来源、检查点、预测ID和指标重算检查。

本地产物不一定包含在GitHub代码中。共享结果时应单独整理可公开汇总，不上传含原始评论和私有路径的整个输出目录。不要与不同数据划分下的论文原始指标直接比较。

## 9. 当前进度与研究边界

已完成：图文评论匹配、真实回复树与生成边区分、结构化Qwen微调、A–E/H多种子对照、Qwen直接分类对照，以及生成顺序和固定正文的父节点诊断。

最新发现：

- 原SFT的1,614条非空上下文中，有1,184条目标父节点是最后一个节点；所有215条根回复目标原先都没有评论上下文。这可能形成位置捷径，但不是因果证明。
- 24条未参与生成器SFT的训练新闻上，固定正文选父节点的新提示任务中，微调模型原顺序答对2/24，多顺序投票仍为2/24。这个诊断任务不同于原来的联合生成任务，不能用它概括原任务完整准确率。
- 简单顺序投票未通过本轮原型门槛。已准备保持真实关系的编号/顺序增强数据，尚未完成改进模型训练或分类收益验证。

后续需要分别验证连接正确性、顺序稳定性和最终分类收益，并扩大数据及独立评估范围。三者不能互相代替，现有小规模探索也不等于“论文级效果已获证实”。

当前Fakeddit评论材料没有可用时间戳，不把结构遍历和生成顺序解释为真实时间传播，也不据此宣称完成早期检测。生成评论是模拟讨论，不是外部事实证据。

## 10. 协作方式

建议数据与评估、模型与训练分别推进，再在同一固定协议下汇合：

- 数据与评估：检查新闻/图片/评论ID、真实父节点、重复与划分泄漏，整理错误案例和每类指标。
- 模型与训练：维护提示、父节点选择和图加载，新建独立实验入口，不悄悄替换旧实现。
- 共同复核：先确定比较组、种子、主要指标，再运行；保留失败和无提升结果。

每次实验至少保存：代码版本、配置、数据清单及哈希、生成器版本、种子、验证选模规则、逐新闻预测、检查点、日志和资源消耗。

修改前建立独立分支。不要两个人同时修改同一个冻结实验目录，也不要同时启动使用同一输出目录的任务。先审查差异和小规模检查，再合并。修改冻结源码需要新协议，不继续沿用旧结果标签。

AI生成的质量标注要注明来源，不能当作人工审核。曾失败的质量过滤器默认不启用。

## 11. 上传GitHub前必须检查

**当前 `.gitignore` 已补充本地数据、模型、缓存、生成报告及第三方下载目录的排除规则。仍需逐项审查待提交文件，不要直接 `git add .`。** 忽略规则不等于敏感信息扫描或完整发布审核已经完成。

可以提交代码、配置、提示模板和审核过的文档。以下内容应排除，按许可另行共享：

- 原始新闻、评论、图片及数据集归档。
- 完整Qwen权重、LoRA权重、分类检查点、下载缓存与临时文件。
- 含原始评论、审核内容、个人路径的报告和预测文件。
- 访问令牌、`.env`、IDE配置，以及外部仓库的嵌套Git记录。

以下是已配置规则的部分示例，完整清单见根目录 [`.gitignore`](.gitignore)：

```gitignore
/all_comments.tsv/
/multimodal_only_samples/
/images/
/checked_data/official_repo/
/checked_data/images/
/comment_generator/models/
**/data/
**/outputs/
**/checkpoints/
**/__pycache__/
*.safetensors
*.pt
*.pth
*.bin
*.sqlite3
*.part
.env
.env.*
/tmp/
.idea/
```

各级 `reports/`、`sample_check/` 的现有生成结果、CHECKED根目录数据清单及下载的 `third_party/MIMoE-FND/` 已排除；数据工具脚本仍可提交。不要用全局 `*.json`、`*.yaml` 误排配置，也不要把外部完整仓库连同其内部Git记录直接加入。若需发布汇总结果，先脱敏和审核，再放入专门的文档目录；历史MIMoE分支所需外部代码仍须按其来源另行准备。

```powershell
git status --short
git diff --stat
# 逐项选择代码和文档暂存后，检查实际提交内容
git diff --cached --stat
git diff --cached
```

忽略规则不会移除已跟踪文件或历史提交中的敏感信息。发现这类内容应先解决再推送。

根目录目前未发现 `LICENSE`。公开前请核对MFCL原代码、第三方项目、数据集和模型的许可，保留作者与来源；不要自行替原作者声明新的开源许可。

## 12. 原项目与相关工作

原README将MFCL描述为KBS接收论文的源代码。原数据链接、依赖和运行方式已完整保留在 [历史说明](docs/MFCL_LEGACY_README.md)，本扩展仓库不代表原作者官方更新。正式引用MFCL时应核对出版版本的作者、年份和书目信息。

原入口：

```powershell
python .\Weibo\test.py -c .\Weibo\configs\weibo.yaml
```

历史独立Fakeddit在线编码入口：

```powershell
# 仅在独立旧环境中安装，不覆盖当前Qwen环境
python -m pip install -r .\requirements-fakeddit.txt
python .\Weibo\train_fakeddit.py -c .\Weibo\configs\fakeddit.yaml
```

这一旧入口使用RoBERTa和ResNet-50，保留二分类 `0=fake, 1=true`，不与当前三分类主线混用。

相关工作：

- [Fakeddit](https://github.com/entitize/Fakeddit)：英文图文与评论数据。
- [CHECKED](https://github.com/cyang03/CHECKED)：历史中文实验数据。
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)：视觉语言基础模型。
- [StruSP，Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.714/)：结构感知合成传播，当前实现不声称完整复现。
- [Found in the Middle，NAACL 2024](https://aclanthology.org/2024.naacl-long.129/)：通过排列一致性缓解列表排序中的位置偏好。
- [Unveiling Selection Biases，Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.333/)：选项顺序和标识符偏好研究。

生成评论、结构化连接、多排列一致性本身均不能直接声明为全新贡献。论文贡献需要清晰的问题定义、区别于既有工作的机制，以及公平、可复核的实验证据。
