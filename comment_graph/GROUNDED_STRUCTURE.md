# 生成评论内容与连接位置对照实验

本实验仅使用现有 Fakeddit 三分类数据，不启用未通过预留审核的 AI 质量过滤器。
Qwen 生成器固定，不联合训练；新闻真实性标签不进入生成提示。

## 两种生成方式

- `iterative`：新评论加入下一轮输入，允许后续评论回复生成评论，保持原对照策略。
- `observed_only`：每次只输入新闻图文和最初真实评论，父节点只能是新闻或真实评论。
  图中仍保存全部新节点，但不把其内容反馈给生成器。去重器可检查已生成文本，不向模型展示文本。

每条新闻目标为 5 条，每个位置最多尝试 3 次。格式错误、重复、非法父节点和上下文超长均记录；
不截断图片 token，不伪造成功，不剔除生成失败的新闻。真实回复边保持原样，生成边是预测增强边，不是真实传播事实。

## 五组比较

| 组别 | 评论来源 | 生成评论连接 |
|---|---|---|
| A_real | 真实评论 | 无生成评论 |
| B_iterative_root | 真实＋迭代生成 | 全部连接新闻 |
| C_iterative_parent | 与 B 相同 | 模型预测 |
| D_observed_root | 真实＋仅真实上下文生成 | 全部连接新闻 |
| E_observed_parent | 与 D 相同 | 模型预测新闻或真实评论 |

B/C、D/E 分别复用同一个 JSONL，只在数据加载时改变增强边。真实节点预算为 16，生成节点预算为 5。
分类模型固定使用 MFCL 图文特征、两层 typed R-GCN、mean pooling 和 concat 分类，无质量门控或额外损失。

训练 292 条新闻。77 条验证新闻中，排除参与生成器检查点选择的 33 条，剩余 44 条用于分类器早停和主要比较。
完整 77 条只用冻结后的检查点评估，结果属于辅助分析，不重新选模型。这些验证新闻曾参与探索，不是全新盲测。

## PowerShell 命令

在项目根目录与 `multimodal-fakenews` 环境下执行：

```powershell
# 默认只准备数据和冻结实验协议
python .\comment_graph\run_grounded_structure.py --stage prepare

# 8 条未参与旧质量标注的真实新闻，生成、图校验、五组 GNN 前向反向测试
python .\comment_graph\run_grounded_structure.py --stage smoke

# Smoke 通过后：两种完整生成、15 次分类训练、辅助验证、汇总
python .\comment_graph\run_grounded_structure.py --stage full

# 仅在验证提升门槛通过、检查点冻结后执行测试
python .\comment_graph\run_grounded_structure.py --stage test
```

也可分别执行 `--stage generate`、`train`、`summarize`。`--output-dir` 可指定新实验目录。
同一目录禁止同时启动两份任务。重复执行会校验已完成结果和来源哈希，不覆盖旧结果。
生成在每条完整新闻后原子保存；中断的当前新闻会重新生成，已完成新闻不会重复下载、生成。
恢复要求输入、提示、模型 adapter、配置和相关代码保持一致；修改方案应使用新目录并重跑 smoke。

固定种子 42、13、24；最多 20 epoch，早停 5。每次训练只在 44 条验证新闻上选检查点。
如果已存在完整训练结果会验证后跳过；只有检查点而没有完整结果的中断训练，将从固定种子重新训练该组。

## 结果与测试门槛

默认输出：`comment_graph/outputs/grounded_structure_20261010/`。

- `protocol.json`：固定样本名单、来源哈希、对照和通过门槛。
- `smoke_report.json`：匹配、节点边数量、向量维度、真实边一致性、GNN 梯度。
- `resource_estimate.json`：小规模实测耗时与显存；完整生成时间估计不含分类训练。
- `*_train.jsonl`、`*_validate.jsonl`：两种生成图；`.run.json` 记录生成契约与资源统计。
- `classification/`：三种种子各五组的验证指标、每类指标、混淆矩阵、逐新闻预测和训练曲线。
- `comparison.json`：44 条主要验证与 77 条辅助验证的均值、标准差及 E 组配对差值。
- `status.json`、`logs/`：当前阶段、错误信息、完整日志。
- `frozen_selection.json`：仅在通过门槛后生成，固定所有待评估检查点哈希。

进入测试的必要条件是 E 相比 A、D 的平均主要验证 Macro-F1 均提高至少 0.01，且每项比较至少 2/3 种子提升。
不满足则停止在验证结果，不能把评论质量或 loss 改善当作分类提升。满足后测试阶段对同一批既有测试新闻评估全部五组，
测试图同样不使用真实类别生成，保存 `test_comparison.json`，不根据测试结果改参数。
这一既有测试队列曾被查看，报告必须保留探索性结果的限定，不能重新称为全新独立盲测。

旧报告与原 MFCL 训练流程不被覆盖。不保证新方法必然提高指标。
