# 近期论文启发与 CHECKED 小规模实验（2026-10-04）

## 结论先说

新增了可关闭的 `mismatch_fallback` 训练正则和评论错配诊断。
代码、反向传播、小规模对照均已运行，但本次 **没有证明性能提升**。
默认 `robustness.mode: none`，已有正式模型和正式实验结果没有被覆盖（技术 smoke 报告已重新运行更新）。
这不是论文复现，也不能作为已经成立的论文创新点。

## 参考论文与取舍

| 论文 | 核心参考点 | 本项目如何处理 |
|---|---|---|
| [CogniDir](https://arxiv.org/abs/2510.09712v4)，2026-09-08 更新；初稿为 2025 年 AdComment | LLM 评论可能误导检测器；按攻击类型调整训练采样 | 借鉴“必须测试评论干扰”的问题设置，实现低成本错配正则；没有复现其攻击生成器或 InfoDirichlet |
| [StruSP，Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.714/) | 生成评论还需要考虑传播结构 | 当前 CHECKED 入口没有真实回复树监督，不伪造 parent，也不把动态语义边叫真实传播边 |
| [Let the Bullets Fly / DM-FEND](https://arxiv.org/abs/2608.22832)，2026-08-24 | 生成弹幕与视频、音频、文本关联，抑制语义噪声 | 本次只阅读其摘要/条目；当前是静态图文，不能声称复现时间对齐弹幕模型 |
| [Generate First, Then Sample，ACL 2025](https://aclanthology.org/2025.acl-long.1182/) | 用生成和自适应采样处理类别不平衡 | 它生成的是新闻，不是本项目的评论；本次不生成真假标签或混入合成新闻 |

CogniDir 查阅了 [v4 方法部分](https://arxiv.org/html/2510.09712v4)。该版本用事实扭曲、逻辑混淆、情绪操纵三类生成评论进行鲁棒学习；我们的随机跨新闻错配比这些攻击简单很多。不同新闻的评论仍可能讨论同一事件，因此这里只能叫“弱错配扰动”，不是已证实的无关评论、恶意评论或因果干预。

## 新方法怎么做

沿用已有 `source_isolated` 架构：

1. 图文 + 真实评论形成基础预测 `base_logits`。
2. 生成评论走独立 GNN，产生有界修正量；门控决定修正幅度。
3. 正常样本仍按真实新闻标签计算分类损失。
4. 训练时，从同一 batch 的其他新闻借用生成评论的文本向量；保留当前新闻根、图文、真实评论、节点数量和类型。
5. 对错配样本，要求预测接近基础预测，而不是让错误匹配的评论大幅改变判断。

设 `p_base` 为基础预测，`p_clean` 为正常评论下预测，`p_swap` 为错配预测：

```text
L = CE(p_clean, y) + 0.5 * CE(p_base, y)
    + lambda * KL(stop_gradient(p_base) || p_swap)
```

`lambda` 默认 0.5，仅在 `--robustness-mode mismatch_fallback` 时启用。
基础分支不是事实核查器，也可能预测错误；回退损失不保证正确性。
正则项的梯度只更新生成评论分支，包括其 GNN、动态边和门控。
正常分类损失仍更新基础分支。共享的 BERT / ResNet50 按当前配置冻结。

实现复用已编码向量，不再次运行图片编码、基础分支或其 BatchNorm。
错配分支关闭整分支随机丢弃，使用独立随机数生成器；不按真实标签选取供体。
没有可用的不同新闻供体时跳过该项；不会跨 train / validate 借用评论。
推理无须增加步骤，不重新训练或调用 Qwen。

## 改动文件

新增：

- `robustness.py`：错配、KL 回退目标。
- `smoke_robustness.py`：真实 8 条新闻与边界情况检查。
- `run_robustness_pilot.py`：小规模配对实验，默认 128 条、2 轮，硬限制最多 256 条、3 轮。
- 本文与 `reports/smoke_robustness.json`、时间戳目录下的实验结果。

修改：

- `model.py`：可选返回已编码生成评论的上下文；默认输出路径不变。
- `isolated_fusion.py`：错配分支可关闭 branch dropout。
- `train.py`：新开关、训练损失、错配诊断、结果记录和独立 checkpoint 文件名。
- `config.yaml`：增加默认关闭的 `robustness` 配置。
- `README.md`：增加入口说明。

本轮没有修改 `Weibo/`、新闻/评论文件、Generator、数据划分或历史 checkpoint。
仓库内已有的其他未提交改动没有回退。

## 实际验证

### 技术检查

- 8 条真实训练新闻；使用本地图片、真实评论和已生成评论。
- 1 次 AMP 参数更新，所有参与反传的梯度有限。
- 错配辅助项单独反传：基础分支无梯度；生成 GNN、动态边、门控有非零梯度。
- 根节点不变、新闻 ID 不自借、不同评论数适配、同 ID 无供体时跳过、空生成集合跳过。
- 交换操作不查看标签；权重 0 保持原损失。
- 可选缓存前后的 logits 最大数值差为 1.49e-8（GPU 归约浮点误差）。
- 不额外更新基础分支 BatchNorm，不消耗主随机数流。
- 原 `smoke_isolated_fusion.py` 回归检查再次通过：两步 AMP、基础预测隔离、缺失评论回退、四种数据模式、星型/动态图、分支参数序列化。

### 配对实验

结果文件：`outputs/robustness_pilot_20261004_111923/results.json`。

- 原训练集 939 条；按原比例、固定种子 42 抽取 128 条：124 real / 4 fake。
- 验证集完整保留 118 条：114 real / 4 fake。
- 没有加载测试集；训练与验证新闻 ID 无交集。
- 两组初始可训练参数哈希一致，batch 顺序哈希一致。
- 每组 2 epoch × 16 step = 32 次更新；batch 8，学习率 1e-4，AMP。
- 类别权重都从完整训练集计算；不是人为平衡子集。
- 都以正常验证集 Macro-F1 选轮次，严格改善才更新；同分保留早轮。
- 小实验只保存指标与逐新闻诊断，不保存新的正式模型权重。

| 方法 | 最佳轮次 | 验证 CE | 验证 Macro-F1 | 错配后 Macro-F1 | 峰值 allocated 显存 | 训练及验证耗时 |
|---|---:|---:|---:|---:|---:|---:|
| source_isolated 对照 | 2 | 0.305474 | 0.566176 | 0.566176 | 0.899 GiB | 27.66 秒 |
| + mismatch_fallback | 1 | 0.501600 | 0.560872 | 0.560872 | 0.905 GiB | 29.42 秒 |

不隐藏末轮：新方法第 2 轮验证 CE 0.393566、Macro-F1 0.4920；不是训练越久就一定更好。
正常/错配评论之间的预测概率 MAE：对照 0.00003383，新方法 0.00008107；两组均有 0/118 条 argmax 改变。
正常/错配的平均门控也几乎一样（对照约 0.1120，新方法约 0.1029）。

解释：本轮没有观察到新方法的分类或回退鲁棒性收益。短训练中的生成分支可能主要依赖新闻上下文，尚未明显区别评论内容；也可能是评论向量相似、扰动太弱或样本太少。这些是待检验解释，不是已证明原因。不能把“错配后也没掉分”直接解释成强鲁棒性。
只训练 128 条、2 轮，不能与之前完整 939 条实验的指标横比。验证集中只有 4 条假新闻，也不适合据单次结果宣称显著提升。

## PowerShell 命令

在项目根目录、`multimodal-fakenews` 环境下：

```powershell
# 8 条真实新闻检查
python .\comment_graph\smoke_robustness.py

# 重复本轮配对实验；新建时间戳目录，不覆盖旧结果
python .\comment_graph\run_robustness_pilot.py --train-size 128 --epochs 2 --seed 42
```

下面是完整训练入口，**本轮没有执行**。默认使用完整 939 条训练集、最多 20 轮、验证早停 5 轮，不跑测试集。该实验仍然需要同时跑对照，不能只报告表现更好的一组。

```powershell
# 同架构对照
python .\comment_graph\train.py `
  --config .\comment_graph\config.yaml `
  --mode real_generated `
  --graph-structure dynamic_cross_modal `
  --pooling-mode mean `
  --decision-mode source_isolated `
  --robustness-mode none `
  --audit-mismatch `
  --seed 42 `
  --allow-formal-training `
  --summary-json .\comment_graph\outputs\isolated_control_seed42.json

# 开启新正则
python .\comment_graph\train.py `
  --config .\comment_graph\config.yaml `
  --mode real_generated `
  --graph-structure dynamic_cross_modal `
  --pooling-mode mean `
  --decision-mode source_isolated `
  --robustness-mode mismatch_fallback `
  --audit-mismatch `
  --seed 42 `
  --allow-formal-training `
  --summary-json .\comment_graph\outputs\mismatch_fallback_seed42.json
```

正式入口会保存完整 `.pt` checkpoint；开启正则的文件名包含 `_mismatch_fallback`，不会覆盖对照。重复运行同模式同 seed 会更新该模式自己的 checkpoint；需要长期保留时先使用新 checkpoint_dir 配置。
配置的 `robustness.mismatch_weight` 可调整；不要使用测试分数选择权重。
`--audit-mismatch` 输出同划分、batch 内的供体 ID、门控、概率变化、纠正/引入错误数；供体选择依赖 batch 大小与顺序，对照时须一致。

正式完整训练按本机 A5000 小实验估计为每组数分钟，20 轮无早停大致 15 分钟量级（不是保证）；batch=8 的峰值 allocated 约 1 GiB，实际预留显存会更高。完整 checkpoint 还包含冻结编码器，需额外磁盘空间。

## 标签泄漏与后续方向

当前生成入口 `comment_generator/partial_observation/scripts/generate_comments.py` 白名单只允许图片路径、新闻正文、observed 评论，不允许真实性标签进入输入。本轮错配也不读取标签决定供体。这不能替代完整历史数据泄漏审计，例如事件近重复、Generator 的训练划分仍需在最终论文实验前核实。

下一步优先做评论贡献诊断：在固定验证集比较同一模型的正常生成评论、删除生成评论、跨新闻错配和重复评论；再决定是否需要让生成分支更依赖评论内容，而不是立刻堆叠注意力层。完整训练的多种子结果确认后，冻结配置再做测试评估；此前已反复查看的测试集不能再当作完全未接触的最终评测集。
