# Fakeddit 评论情绪分析

该模块独立于 MFCL 和 Qwen 评论生成器。它用 GoEmotions 模型分析评论中表达出的情绪，并保留 Fakeddit 的顶层评论、回复关系和赞同票。

注意：这里预测的是文本表达出来的情绪，不代表模型能够读取用户真实的内心状态。

## 流程

```text
原始 comments.tsv
        ↓
抽取 126 条可用新闻的评论及 parent_id
        ↓
GoEmotions 生成每条评论的 28 维多标签概率
        ↓
新闻级情绪分布 + 回复层级情绪转移
```

## 1. 抽取评论结构

```powershell
conda activate multimodal-fakenews
python .\emotion_analysis\extract_comments.py `
  --config .\emotion_analysis\config.yaml
```

输出：

- `data/comments_with_metadata.jsonl`
- `data/extraction_stats.json`

本项目实测结果：126 条新闻匹配到 3,136 条原始评论；清理 94 条空白、`[deleted]` 或 `[removed]` 后保留 3,042 条评论。其中顶层评论 979 条、回复 2,063 条，可解析父子回复边 1,945 条。13 条正文重复记录被保留并标记，因为它们是不同评论节点。

抽取结果同时保留 `comment_id`、`parent_id`、`parent_type`、`is_top_level`、`ups` 和 `submission_id`。正文重复的评论不会从传播图中删除，而是用 `duplicate_body` 标记，因为不同评论节点可能具有相同文本。

## 2. GoEmotions 标注

不要和 Qwen 训练同时占用同一张 GPU。首次执行会下载 `SamLowe/roberta-base-go_emotions`。

先用 20 条评论测试：

```powershell
python .\emotion_analysis\annotate_emotions.py `
  --config .\emotion_analysis\config.yaml `
  --limit 20
```

冒烟测试写入 `outputs/comment_emotions.smoke.jsonl`，不会覆盖完整结果。

全部标注：

```powershell
python .\emotion_analysis\annotate_emotions.py `
  --config .\emotion_analysis\config.yaml
```

如果 Qwen 正在占用 GPU，可以使用 CPU：

```powershell
python .\emotion_analysis\annotate_emotions.py `
  --config .\emotion_analysis\config.yaml `
  --device cpu
```

输出为 `outputs/comment_emotions.jsonl`。每条评论保留完整的 28 维 sigmoid 概率、阈值以上的多标签结果和最高概率情绪。

如果 Hugging Face 下载超时，可以先在网络可用的终端下载模型，或者把模型目录写到 `config.yaml` 的 `emotion_model.name_or_path`。模型完整缓存后可加 `--local-files-only`，确保运行时不再联网。

## 3. 聚合新闻情绪与父子回复情绪

```powershell
python .\emotion_analysis\aggregate_emotions.py `
  --config .\emotion_analysis\config.yaml
```

输出：

- `news_emotion_features.csv`：每条新闻的顶层评论和全部评论情绪分布。
- `label_emotion_summary.csv`：按 Fakeddit 标签聚合的平均情绪分布。
- `reply_emotion_edges.csv`：父评论到回复评论的情绪边。
- `emotion_transition_matrix.csv`：28×28 父子情绪转移计数。
- `report.md`：统计报告。

新闻特征同时保存均匀平均和基于 `ups` 的对数加权平均。因为评论文件没有时间戳，结果只能解释为回复层级上的情绪传播，不能解释为时间情绪演化。
