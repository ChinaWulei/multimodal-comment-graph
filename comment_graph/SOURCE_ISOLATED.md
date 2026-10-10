# 来源隔离的评论图增强

本次只实现“真实评论与生成评论分开传播、受控融合”。不同时增加干扰训练、
缺失角度生成等其他方案，也不修改 `Weibo/`、CHECKED 原始数据、生成器或已有生成评论。
这是待验证的方法，不是事实核查器，也不保证指标上升。

## 流程

1. 输入现有新闻文本、实际图片、真实评论及已经生成的评论。
2. 基础图只保留新闻根节点和真实评论；删除生成节点后重新编号并筛选边。
3. 使用现有共享 BERT、ResNet 图文编码器及 MFCL 基础 GNN，得到
   `h_mm=1536`、`h_base_graph=768`、`h_base=2304`，输出 `z_base`。
4. 另建新闻根节点和生成评论组成的图，使用**参数独立**的 MFCL 两层 GCN，
   得到 `h_generated_graph=768`。真实/生成评论使用同一个 BERT 和文本投影，
   不另下载文本模型；生成图不重复编码图片。
5. 生成图向量结合新闻图文向量产生修正量和逐新闻 gate：

   `delta = gate * max_logit_delta * tanh(correction)`

   `z_final = z_base + delta`

两张图没有跨来源边，即使是新闻根节点也各自保留独立的传播状态。
支持 `star` 和 `dynamic_cross_modal`。动态图仍是特征驱动的邻接选择，
不是恢复的真实回复树或按真实时间演化的传播树。为隔离实验因素，新模式暂时只支持 `mean` pooling。

基础图先编码，避免生成评论数量改变基础文本编码分块。当前编码器默认冻结；
若未来解冻共享 BERT，两个任务会在训练过程中通过共享参数相互影响，
不能宣称整个训练过程完全独立。这里保证的是给定模型的前向数据路径隔离。

`detach_base_context=true` 阻止增强分支通过输入的基础图文/基础特征反传；
最终 CE 仍会正常更新基础分类器。真实评论不被假定为正确事实。

## 损失与开关

`loss = CE(z_final, y) + base_loss_weight * CE(z_base, y)`

两项使用现有训练集类别权重；验证/测试 `loss` 仍只报告最终分类 CE。
日志分别记录 `train_loss`（总目标）、`train_final_ce`、`train_base_ce`，
不能直接将总目标数值与旧版单项 CE 比较。最佳检查点仍由验证集最终 Macro-F1 选择。

配置 `config.yaml -> isolated_fusion`：

- `hidden_dim: 128`
- `base_loss_weight: 0.5`：防止基础路径完全依赖修正分支。
- `gate_init: 0.1`
- `branch_dropout: 0.2`：训练时逐新闻随机关闭整个生成修正，不做反向缩放；推理不随机关闭。
- `max_logit_delta: 2.0`：限制每类修正的绝对幅度；并不保证不误判。
- `detach_base_context: true`
- `amp_init_scale: 1024.0`：初次 smoke test 在默认 65536 缩放下出现 FP16 梯度溢出；降低初值后通过。

没有生成评论的新闻（包括混合 batch 中个别缺失）修正严格为零。
`model(batch, disable_generated=True)` 可以关闭新模式的增强分支。
四种数据模式仍支持：baseline、real_only、generated_only、real_generated。
基础路径不含生成节点，所以 `generated_only` 下基础图只剩新闻根节点。

`global_macro_f1` / `global_confusion_matrix` 在新模式下是基础分支指标，
并非另一模型独立训练的结果。逐新闻记录含基础概率、最终概率、gate 和修正量。
可以检查新分支纠正了谁、又误伤了谁。损失和标签不参与前向门控输入。

## 小规模验证

```powershell
python .\comment_graph\smoke_isolated_fusion.py --config .\comment_graph\config.yaml
```

脚本仅选择训练集中 8 条真实新闻（每类 4 条），使用实际图片和已有评论，执行两次 AMP
优化步骤；不会训练生成器、下载新模型或用测试集反传。报告：
`comment_graph/reports/smoke_isolated_fusion.json`。

检查两条 GNN、两套动态图边评分器、基础分类器和门控均有有限非零梯度；
替换/删除/重复生成评论不改变基础输出；缺失评论与关闭分支可回退；
检查无跨来源/跨新闻边、标签不进入前向、四种数据模式、两种图结构，以及新模块序列化。
梯度覆盖测试临时关闭 branch dropout，随后另行测试 dropout=1 的精确回退。
Smoke test 的 Loss 下降不作为有效性证据。

## 正式训练

项目根目录、已激活 `multimodal-fakenews` 环境：

```powershell
python .\comment_graph\train.py `
  --config .\comment_graph\config.yaml `
  --mode real_generated `
  --graph-structure dynamic_cross_modal `
  --pooling-mode mean `
  --decision-mode source_isolated `
  --seed 42 `
  --allow-formal-training `
  --summary-json .\comment_graph\outputs\source_isolated_seed42_results.json
```

如果已经冻结本次实验配置，希望训练完成后评估测试集，在命令中再加 `--run-test`。
测试发生在加载最佳验证检查点之后，没有反向传播，不参与早停。
训练开始显示 `decision=source_isolated`、`base_loss_weight=0.5` 才是新入口。

新检查点：
`comment_graph/outputs/checked_mfcl_real_generated_dynamic_cross_modal_mean_source_isolated_seed42.pt`。
不会覆盖旧 concat、局部修正或质量门控检查点。同名配置/种子再次训练会覆盖该新模式自己的文件。
本命令从预训练图文编码器初始化后重新训练分类模块，不是继续微调 Qwen 或恢复旧分类器。

## 实验边界

先比较同一数据、划分、种子下的 concat、global_local_correction、source_isolated。
这些是完整方案比较，不能把差异全部归因于图隔离（还改变了辅助监督与分支关闭）。
后续单因素消融可依次设 `base_loss_weight=0`、`branch_dropout=0`；保存不同配置及输出目录，
避免覆盖。进一步使用 real_only 对照和多个种子；通过验证集做选择，不根据测试集调参。

当前验证/测试各只有 4 条假新闻，单次 F1 波动很大；现有测试集已反复查看，结果应作为
探索性结果，需扩大有效样本或增加独立留出数据。来源隔离不消除数据重复或生成器训练泄漏：
既有评论仍需遵守原始新闻划分、不得使用验证/测试真实标签生成。本实现不重审原生成器的全部训练历史。
如果后续做评论稀缺实验，生成器也必须只看到允许的评论，不能从完整评论中提前获知信息。
