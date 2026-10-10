from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import load_config, normalize_reddit_id, read_jsonl, resolve_path


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().with_name("config.yaml")
    parser = argparse.ArgumentParser(description="Aggregate comment emotion distributions.")
    parser.add_argument("--config", type=Path, default=default_config)
    return parser.parse_args()


def normalized_distribution(matrix: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if matrix.size == 0:
        return np.array([])
    if weights is None:
        vector = matrix.mean(axis=0)
    else:
        vector = np.average(matrix, axis=0, weights=weights)
    total = vector.sum()
    return vector / total if total > 0 else np.full_like(vector, 1.0 / len(vector))


def entropy(vector: np.ndarray) -> float:
    positive = vector[vector > 0]
    return float(-(positive * np.log(positive)).sum()) if positive.size else 0.0


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    samples_path = resolve_path(config_path, config["source"]["usable_samples_csv"])
    output_dir = resolve_path(config_path, config["output"]["directory"])
    emotion_path = output_dir / config["output"]["comment_emotions"]
    samples = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
    samples["_normalized_id"] = samples["news_id"].map(normalize_reddit_id)
    sample_map = samples.set_index("_normalized_id").to_dict("index")
    comments = read_jsonl(emotion_path)
    if not comments:
        raise RuntimeError(f"No emotion annotations in {emotion_path}")
    labels = list(comments[0]["emotion_probabilities"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in comments:
        grouped.setdefault(record["submission_id"], []).append(record)

    feature_rows = []
    for news_id, records in grouped.items():
        if news_id not in sample_map:
            continue
        metadata = sample_map[news_id]
        all_matrix = np.asarray(
            [[record["emotion_probabilities"][label] for label in labels] for record in records],
            dtype=np.float64,
        )
        top_records = [record for record in records if record["is_top_level"]]
        top_matrix = np.asarray(
            [[record["emotion_probabilities"][label] for label in labels] for record in top_records],
            dtype=np.float64,
        )
        all_uniform = normalized_distribution(all_matrix)
        top_uniform = normalized_distribution(top_matrix) if top_records else np.zeros(len(labels))
        ups_weights = np.asarray(
            [math.log1p(max(float(record.get("ups") or 0.0), 0.0)) + 1.0 for record in records]
        )
        top_ups_weights = np.asarray(
            [math.log1p(max(float(record.get("ups") or 0.0), 0.0)) + 1.0 for record in top_records]
        )
        all_weighted = normalized_distribution(all_matrix, ups_weights)
        top_weighted = (
            normalized_distribution(top_matrix, top_ups_weights)
            if top_records
            else np.zeros(len(labels))
        )
        primary = top_uniform if top_records else all_uniform
        row = {
            "news_id": metadata["news_id"],
            "split": metadata["split"],
            "headline": metadata["headline"],
            "2_way_label": metadata["2_way_label"],
            "3_way_label": metadata["3_way_label"],
            "6_way_label": metadata["6_way_label"],
            "comment_count": len(records),
            "top_level_count": len(top_records),
            "reply_count": len(records) - len(top_records),
            "primary_top_emotion": labels[int(primary.argmax())],
            "primary_emotion_entropy": entropy(primary),
        }
        for index, label in enumerate(labels):
            row[f"all_uniform_{label}"] = all_uniform[index]
            row[f"top_uniform_{label}"] = top_uniform[index]
            row[f"all_logups_{label}"] = all_weighted[index]
            row[f"top_logups_{label}"] = top_weighted[index]
        feature_rows.append(row)

    features = pd.DataFrame(feature_rows).sort_values(["split", "news_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / config["output"]["news_features"]
    features.to_csv(features_path, index=False, encoding="utf-8-sig")

    label_column = str(config["aggregation"]["label_column"])
    emotion_columns = [f"top_uniform_{label}" for label in labels]
    label_summary = features.groupby(label_column, as_index=False)[emotion_columns].mean()
    summary_path = output_dir / config["output"]["label_summary"]
    label_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    comment_by_id = {
        record["comment_id"]: record for record in comments if record.get("comment_id")
    }
    edge_rows = []
    transition_counts: Counter[tuple[str, str]] = Counter()
    for child in comments:
        if child.get("parent_type") != "comment":
            continue
        parent = comment_by_id.get(child.get("parent_id"))
        if parent is None:
            continue
        edge_rows.append(
            {
                "news_id": child["submission_id"],
                "parent_comment_id": parent["comment_id"],
                "child_comment_id": child["comment_id"],
                "parent_top_emotion": parent["top_emotion"],
                "parent_top_probability": parent["top_emotion_probability"],
                "child_top_emotion": child["top_emotion"],
                "child_top_probability": child["top_emotion_probability"],
            }
        )
        transition_counts[(parent["top_emotion"], child["top_emotion"])] += 1
    edges = pd.DataFrame(
        edge_rows,
        columns=[
            "news_id",
            "parent_comment_id",
            "child_comment_id",
            "parent_top_emotion",
            "parent_top_probability",
            "child_top_emotion",
            "child_top_probability",
        ],
    )
    edges_path = output_dir / config["output"]["reply_edges"]
    edges.to_csv(edges_path, index=False, encoding="utf-8-sig")

    transition = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for (parent_emotion, child_emotion), count in transition_counts.items():
        transition.loc[parent_emotion, child_emotion] = count
    transition.index.name = "parent_emotion"
    transition_path = output_dir / config["output"]["transition_matrix"]
    transition.to_csv(transition_path, encoding="utf-8-sig")

    top_by_label = []
    for _, row in label_summary.iterrows():
        values = np.asarray([row[column] for column in emotion_columns])
        order = values.argsort()[::-1][:5]
        top_by_label.append(
            (row[label_column], [(labels[index], float(values[index])) for index in order])
        )
    report = [
        "# Fakeddit 评论情绪分析报告",
        "",
        f"- 评论情绪模型：`{comments[0]['emotion_model']}`",
        f"- 新闻数量：{len(features)}",
        f"- 评论数量：{len(comments)}",
        f"- 顶层评论：{sum(record['is_top_level'] for record in comments)}",
        f"- 回复评论：{sum(not record['is_top_level'] for record in comments)}",
        f"- 可解析父子回复边：{len(edges)}",
        "- 新闻级主分布优先使用顶层评论；没有顶层评论时回退到全部评论。",
        "- `ups` 加权结果使用 `1 + log(1 + max(ups, 0))`，同时保留不加权分布。",
        "- 情绪标签是模型对文本表达的估计，不代表用户真实内心状态。",
        "- 数据没有时间戳，因此父子边只表示回复层级，不表示精确时间演化。",
        "",
        f"## 按 `{label_column}` 的主要情绪",
        "",
    ]
    for label_value, top_values in top_by_label:
        formatted = ", ".join(f"{name}={value:.4f}" for name, value in top_values)
        report.append(f"- 标签 `{label_value}`：{formatted}")
    report.extend(
        [
            "",
            "## 输出",
            "",
            f"- 新闻级情绪特征：`{features_path.name}`",
            f"- 标签情绪汇总：`{summary_path.name}`",
            f"- 父子回复情绪边：`{edges_path.name}`",
            f"- 情绪转移矩阵：`{transition_path.name}`",
        ]
    )
    report_path = output_dir / config["output"]["report"]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "news": len(features),
                "comments": len(comments),
                "reply_edges": len(edges),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
