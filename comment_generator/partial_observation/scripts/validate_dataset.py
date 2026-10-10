from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from common import (
    LABEL_COLUMNS,
    load_config,
    load_prompt_template,
    read_jsonl,
    render_user_prompt,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator.yaml"
    parser = argparse.ArgumentParser(description="Validate Partial Observation data and leakage barriers.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--smoke-limit", type=int, default=20)
    parser.add_argument("--preview-count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data_config = config["data"]
    output_dir = resolve_path(config_path, data_config["output_dir"])
    reports_dir = config_path.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    template_path = resolve_path(config_path, config["prompt"]["template_path"])
    template = load_prompt_template(template_path)
    manifest = pd.read_csv(
        output_dir / data_config["manifest_csv"], dtype=str, keep_default_na=False
    )
    split_paths = {
        "train": output_dir / data_config["train_jsonl"],
        "validate": output_dir / data_config["val_jsonl"],
        "test": output_dir / data_config["test_jsonl"],
    }
    datasets = {split: read_jsonl(path) for split, path in split_paths.items()}
    seen_sample_ids = set()
    news_by_split: dict[str, set[str]] = {split: set() for split in datasets}
    target_ids_by_split: dict[str, set[str]] = {split: set() for split in datasets}
    target_texts_by_split: dict[str, set[str]] = {split: set() for split in datasets}
    checked = 0
    leakage_checks = Counter()

    for split, records in datasets.items():
        for record in records:
            sample_id = record.get("sample_id")
            if not sample_id or sample_id in seen_sample_ids:
                raise ValueError(f"Missing or duplicate sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)
            if set(record) != {"sample_id", "metadata", "model_input", "target_comment"}:
                raise ValueError(f"Unexpected top-level fields in {sample_id}")
            metadata = record["metadata"]
            model_input = record["model_input"]
            if set(model_input) != {"image_path", "news_text", "observed_comments"}:
                raise ValueError(f"Forbidden or missing model_input field in {sample_id}")
            if "labels" not in metadata or set(metadata["labels"]) != set(LABEL_COLUMNS):
                raise ValueError(f"Labels must be isolated under metadata.labels: {sample_id}")
            if metadata["split"] != split:
                raise ValueError(f"Split mismatch in {sample_id}")
            news_by_split[split].add(metadata["news_id"])
            target_ids_by_split[split].add(str(metadata["target_comment_id"]))
            target_texts_by_split[split].add(
                " ".join(str(record["target_comment"]).casefold().split())
            )
            if metadata["target_comment_id"] in set(metadata["observed_comment_ids"]):
                raise ValueError(f"Observed/target comment leakage in {sample_id}")
            prompt = render_user_prompt(template, model_input)
            static_lower = template.lower()
            if any(field in static_lower for field in LABEL_COLUMNS):
                raise ValueError("Prompt template contains a label column")
            if not str(record["target_comment"]).strip():
                raise ValueError(f"Empty target in {sample_id}")
            leakage_checks["schema_separates_metadata_and_model_input"] += 1
            leakage_checks["target_not_in_observed_ids"] += 1
            leakage_checks["prompt_built_only_from_model_input"] += 1
            if checked < args.smoke_limit:
                image_path = Path(model_input["image_path"])
                with Image.open(image_path) as image:
                    image.load()
                if not prompt.strip():
                    raise ValueError(f"Empty prompt in {sample_id}")
                checked += 1

    split_names = list(news_by_split)
    overlap = {}
    target_id_overlap = {}
    target_text_overlap = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            ids = news_by_split[left] & news_by_split[right]
            overlap[f"{left}-{right}"] = len(ids)
            if ids:
                raise ValueError(f"News leakage across {left}/{right}: {sorted(ids)[:5]}")
            pair = f"{left}-{right}"
            target_id_overlap[pair] = len(
                target_ids_by_split[left] & target_ids_by_split[right]
            )
            target_text_overlap[pair] = len(
                target_texts_by_split[left] & target_texts_by_split[right]
            )
            if target_id_overlap[pair] or target_text_overlap[pair]:
                raise ValueError(f"Target comment leakage across {left}/{right}")

    image_news_counts = manifest.groupby("image_path")["news_id"].nunique()
    image_cross_news_count = int((image_news_counts > 1).sum())
    if image_cross_news_count:
        raise ValueError("One local image path maps to multiple news IDs")
    leakage_audit = {
        "news_id_overlap": overlap,
        "target_comment_id_overlap": target_id_overlap,
        "normalized_target_text_overlap": target_text_overlap,
        "test_target_comments_used_for_training": len(
            target_ids_by_split["train"] & target_ids_by_split["test"]
        ),
        "image_paths_mapped_to_multiple_news": image_cross_news_count,
        "news_counts": {split: len(ids) for split, ids in news_by_split.items()},
        "target_comment_counts": {
            split: len(ids) for split, ids in target_ids_by_split.items()
        },
    }
    (reports_dir / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    train_records = datasets["train"]
    rng = random.Random(int(config["seed"]))
    preview = rng.sample(train_records, min(args.preview_count, len(train_records)))
    preview_lines = [
        "# Partial Observation SFT 样本预览",
        "",
        "下面的标签只显示在 metadata 检查行中，不属于模型 Prompt。",
        "",
    ]
    console_preview = []
    for index, record in enumerate(preview, 1):
        metadata = record["metadata"]
        prompt = render_user_prompt(template, record["model_input"])
        preview_lines.extend(
            [
                f"## 样本 {index}: `{record['sample_id']}`",
                "",
                f"- news_id：`{metadata['news_id']}`",
                f"- split：`{metadata['split']}`",
                f"- image：`{record['model_input']['image_path']}`",
                f"- metadata labels（不进入模型）：`{json.dumps(metadata['labels'], ensure_ascii=False)}`",
                "",
                "### 实际模型 Prompt",
                "",
                "```text",
                prompt,
                "```",
                "",
                "### Target comment",
                "",
                record["target_comment"],
                "",
            ]
        )
        console_preview.append(
            {
                "sample_id": record["sample_id"],
                "model_input": record["model_input"],
                "target_comment": record["target_comment"],
            }
        )
    preview_path = reports_dir / "sample_preview.md"
    preview_path.write_text("\n".join(preview_lines), encoding="utf-8")

    eligible = manifest[manifest["eligible"].str.lower().eq("true")]
    manifest_stats = json.loads((output_dir / "manifest_stats.json").read_text(encoding="utf-8"))
    report_lines = [
        "# Partial Comment Observation 数据报告",
        "",
        "## 数据来源",
        "",
        f"- 图片目录：`{resolve_path(config_path, config['source']['images_dir'])}`",
        f"- 评论文件：`{resolve_path(config_path, config['source']['comments_tsv'])}`",
        "- 评论字段：空索引列、id、author、body、isTopLevel、parent_id、submission_id、ups。",
        "- 评论文件没有时间戳。本实验使用固定种子的随机部分观察，不解释为早期评论或未来评论预测。",
        "",
        "## 筛选结果",
        "",
        f"- 本地图片：{manifest_stats['images']} 张",
        f"- Pillow 验证有效：{manifest_stats['valid_images']} 张",
        f"- 匹配新闻表：{manifest_stats['matched_news']} 条",
        f"- 至少一条有效评论：{manifest_stats['news_with_any_valid_comments']} 条新闻",
        f"- 至少一条有效顶层评论：{manifest_stats['news_with_top_level_comments']} 条新闻",
        f"- 排除明显自动回复：{manifest_stats['excluded_automated_top_level_comments']} 条",
        f"- 至少 {config['partial_observation']['min_total_comments']} 条有效去重顶层评论：{manifest_stats['eligible_news']} 条新闻",
        f"- 有效新闻全部有效评论平均数：{manifest_stats['eligible_all_valid_comment_mean']:.2f}",
        f"- 有效新闻全部有效评论最大数：{manifest_stats['eligible_all_valid_comment_max']}",
        f"- 有效新闻顶层评论平均数：{manifest_stats['eligible_top_level_comment_mean']:.2f}",
        f"- 有效新闻顶层评论最大数：{manifest_stats['eligible_top_level_comment_max']}",
        "",
        "## 原始划分与 SFT 数量",
        "",
    ]
    for split in ["train", "validate", "test"]:
        news_count = int((eligible["split"] == split).sum())
        report_lines.append(
            f"- {split}: {news_count} 条新闻，{len(datasets[split])} 条 SFT 样本"
        )
    report_lines.extend(["", "## 标签分布（仅 metadata）", ""])
    for column in LABEL_COLUMNS:
        distribution = eligible[column].value_counts().sort_index().to_dict()
        report_lines.append(f"- `{column}`：{json.dumps(distribution, ensure_ascii=False)}")
    report_lines.extend(
        [
            "",
            "## Label leakage 防护",
            "",
            "- 标签仅存在于 `metadata.labels`。",
            "- `model_input` 只允许 `image_path`、`news_text`、`observed_comments` 三个字段。",
            "- Prompt 渲染函数只接收 `model_input`，不接收 metadata。",
            "- Prompt 模板只允许 `{news_text}` 与 `{observed_comments}` 两个占位符。",
            "- Target comment ID 不得出现在 observed comment IDs 中。",
            f"- train/validate/test 新闻交叉数量：{json.dumps(overlap, ensure_ascii=False)}",
            f"- Target comment ID 交叉数量：{json.dumps(target_id_overlap, ensure_ascii=False)}",
            f"- 规范化 Target 正文交叉数量：{json.dumps(target_text_overlap, ensure_ascii=False)}",
            f"- 一张本地图片映射多个 news_id：{image_cross_news_count}",
            f"- 数据读取 smoke test：成功读取并解码 {checked} 条样本图片。",
            "",
            "## 限制",
            "",
            "- 当前172张图片全部属于官方 train，validate 和 test 没有可用本地图片。",
            "- 因此当前数据足够做数据 smoke test 和小规模训练，但不能报告官方验证或测试性能。",
            "- 个别新闻有大量评论；默认保留全部未观察目标评论，正式训练前应关注新闻级样本权重不均衡。",
        ]
    )
    report_path = reports_dir / "data_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    result = {
        "samples": {split: len(records) for split, records in datasets.items()},
        "news": {split: len(ids) for split, ids in news_by_split.items()},
        "split_overlap": overlap,
        "target_comment_id_overlap": target_id_overlap,
        "target_comment_text_overlap": target_text_overlap,
        "image_paths_mapped_to_multiple_news": image_cross_news_count,
        "smoke_samples_checked": checked,
        "leakage_assertions_per_sample": dict(leakage_checks),
        "preview": str(preview_path),
        "report": str(report_path),
        "preview_records": console_preview,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
