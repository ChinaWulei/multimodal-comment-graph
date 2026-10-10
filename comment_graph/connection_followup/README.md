# 根节点补充连接：结构后续实验

上一轮 E（预测连接）比 A（真实评论）有提升，但没有优于 D（同内容根连接）。
训练图审计显示 907/1458 条生成评论选择最后一个可见真实节点，且不少父节点深于一层。
这只是位置偏好与图距离的观察，不证明父节点错误，也不证明它是指标变化原因。

## 本轮假设与对照

- F_root_parent：保留预测父节点连接，再补一条生成节点与新闻根节点之间的双向边。
- G_root_permuted_parent：同样补根节点边，但在每条新闻内部打乱生成评论与非根父节点的对应。

G 保持每个父节点接收的生成评论数不变，因此 F/G 的每个节点度数、边数和真实边都相同。
打乱使用固定新闻 ID 哈希和种子，不读正文、标签或分类结果。若一条新闻的生成评论都指向同一父节点，
保持度数的打乱不能改变连接；该限制与实际变动数量记录在 smoke_report.json。

额外边是模型实验的增强连接；结果不再是一棵严格回复树，不声称恢复真实传播过程。

## 保持不变

直接复用上一轮 observed_only 的全部生成评论，不重新生成、不删新闻。
训练292条、主验证44条、辅助验证77条，真实评论上限16、生成上限5。
同一图文编码器、typed R-GCN、mean pooling、concat分类器、交叉熵损失和训练配置。
随机种子42/13/24，最多20 epoch、早停5、batch8、lr1e-4。
无额外注意力、质量过滤器、检索或新损失。

F 需同时比 A、D、G 平均 Macro-F1 高至少1个百分点，且每项比较至少2/3种子提升，才通过本轮验证门槛。
这是检查过旧验证结果之后的探索性后续实验，不能称为新盲测。入口没有测试集阶段，也不会自动绕过前轮测试纪律。

## 实现说明

所有新代码只在本目录。`experiment.py` 在独立进程中给原训练器提供包装后的数据集，复用原训练循环和模型，
不编辑或覆盖已经冻结的 MFCL / comment_graph 源文件。包装只改变 edge_index 与 edge_type。
自定义连接策略与协议哈希保存到检查点 config.graph.connection_followup。

**这些检查点必须通过本目录的 auxiliary 入口加载。** 原通用评估器不了解补充连接规则，不能直接用它评估这些检查点。
目录、结果名中的 F/G 才是实际连接策略；旧模型字段 graph_structure=observed_tree 只表示底层加载原始真实树。

## PowerShell

```powershell
python .\comment_graph\connection_followup\audit.py
python .\comment_graph\connection_followup\test_edges.py
python .\comment_graph\connection_followup\experiment.py --stage prepare
python .\comment_graph\connection_followup\experiment.py --stage smoke
python .\comment_graph\connection_followup\experiment.py --stage full
```

full 串行执行6次训练、冻结检查点的77条辅助验证，并汇总。
只跑一个种子时（需先 prepare 和 smoke）：

```powershell
python .\comment_graph\connection_followup\experiment.py --stage train-one --arm F_root_parent --seed 42
python .\comment_graph\connection_followup\experiment.py --stage auxiliary --arm F_root_parent --seed 42
```

输出在 `comment_graph/connection_followup/outputs/round1/`：
protocol.json 固定来源与成功标准；previous_round_audit.json / cases.jsonl 记录旧轮诊断；
smoke_report.json 记录真实数据校验与梯度；classification/ 保存逐新闻预测、每类指标、混淆矩阵及训练曲线；
checkpoints/ 保存各组检查点；comparison.json / report.md 汇总三种种子；status.json / logs/ 记录进度。
只有完整且来源哈希匹配的结果可复用；中断的单次训练会从固定种子重新运行，不覆盖其他组。
