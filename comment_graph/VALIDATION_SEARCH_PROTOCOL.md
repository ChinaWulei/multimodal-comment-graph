# 2026-10-04 完整训练集后续实验约定

目的：继续验证改动能否改善 CHECKED 检测，不能靠不断试验直到偶然出现高分。

## 首轮固定设置

- train 全部 939 条（910 real、29 fake），validate 全部 118 条（114 real、4 fake）。
- 不加载 test，不改变已有划分或剔除错分样本；不重新生成评论。
- 架构 source_isolated / dynamic_cross_modal / mean；20 real + 5 generated 上限。
- 先固定 seed 42 比较三组：control、mismatch、balanced。
- control：原随机抽样、原类别加权 CE；mismatch：只增加系数 0.5 的错配回退损失。
- balanced：相同架构，无错配损失；仅训练集有放回按类别逆频率抽样，使两类抽样概率约 1:1；训练 CE 不再加逆频率权重，避免重复补偿。
- 每组最多 20 epoch，batch 8，学习率 1e-4，patience 5。
- 指标计算时验证集保留原分布；各组验证 CE 都使用相同的完整训练集类别权重。
- 主指标固定为正常验证 Macro-F1；相同 F1 不用其他指标另选更晚轮次。
- 全部日志、失败结果、每轮采样次数与最佳 checkpoint 保留在新目录。

## 解释限制

均衡抽样仍只有 29 条不同的假新闻，重复看到它们不是增加了独立样本；有过拟合风险。
各组训练 loss 因加权方式不同不可直接横比，验证 loss 可比。
对照与错配正则使用相同的 batch 顺序；均衡组的顺序变化是该方法本身的一部分。
如果验证 F1 改善，使用相同改动在新的随机种子上配对复查，不只保留最好的种子。
多个随机种子不是多个独立验证数据集。验证只有 4 条假新闻，持续在其上选方法仍会过拟合。
即使平均分上升，也只称为探索性验证改善，不能宣称独立测试提升或统计显著。
历史上其他设置已有单次验证 F1=1.0；本轮比较同架构配对增益与跨种子稳定性，不把胜过来源隔离对照宣称为胜过所有历史方法。
seed 43、44 的 control 在 seed 42 首轮筛选完成前启动，以提前量化对照波动；候选改动仍只能依据 seed 42 首轮结果选定，再原样复查。

## 评论贡献诊断

固定训练后的模型，在验证集比较：正常生成评论、删除生成评论、同一新闻内重复首条生成评论、借用别的新闻的生成评论。
记录 F1、逐新闻概率变化和翻转数。删除不应改变隔离基础分支。
指标不变可能因为模型没用内容、评论表示近似、扰动弱或分类边界未跨越，不能自动当成鲁棒性成功。

## 命令

```powershell
python .\comment_graph\run_validation_search.py --seeds 42 --arms control mismatch balanced --epochs 20 --patience 5
```

每次启动会创建新的 `comment_graph/outputs/validation_search_时间戳/`，含预先保存的 `plan.json`、逐组日志、结果 JSON 和完整 `.pt`。
该脚本没有测试集开关。单独训练入口新增 `--sampling balanced`，默认仍为 `shuffle`。

训练后的诊断：

```powershell
python .\comment_graph\audit_generated_contribution.py --checkpoint 完整checkpoint路径 --output 新的诊断结果路径.json
```

首轮结果和复查结果将另外记录，不修改这里预先约定的比较规则。
