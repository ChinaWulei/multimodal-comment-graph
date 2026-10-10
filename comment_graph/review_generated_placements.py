"""Readable, label-blind review of generated-parent changes on validation news."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform


def compact(value: str, limit: int = 240) -> str:
    return " ".join(str(value).split())[:limit].replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    config.graph.structure_mode = "observed_tree"
    tokenizer, transform = make_tokenizer_and_transform(config)
    dataset = make_dataset(config, config_path, "validate", "real_generated", tokenizer, transform)
    by_id = {row["news_id"]: row for row in dataset.records}
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    reviews = []
    for prediction in summary["validation"]["prediction_records"]:
        news_id = prediction["news_id"]
        if news_id not in by_id:
            raise ValueError(f"Validation news ID missing: {news_id}")
        row = by_id[news_id]
        comments, node_types, _ = dataset._nodes_and_edges(news_id)
        contents = [row["news_text"], *comments]
        types = [0, *node_types]
        for item in prediction.get("generated_reliability", []):
            if not item["rewired"]:
                continue
            original = item["original_parent"]
            selected = item["selected_parent"]
            child = item["node_index"]
            if not all(0 <= index < len(contents) for index in (original, selected, child)):
                raise ValueError(f"Invalid local node index in {news_id}")
            reviews.append({
                "news_id": news_id, "headline": row["news_text"],
                "image_path": row["image_path"],
                "comment": contents[child], "original_parent": contents[original],
                "selected_parent": contents[selected],
                "selected_parent_type": types[selected],
                "original_similarity": item["original_parent_similarity"],
                "selected_similarity": item["selected_parent_similarity"],
                "fused_reliability": item.get("fused_reliability", item["text_reliability"]),
            })
    reviews.sort(key=lambda item: item["selected_similarity"] - (item["original_similarity"] or 0), reverse=True)
    lines = [
        "# Generated-parent review (validation only)", "",
        "These are semantic augmentation edges, **not** verified human reply links.",
        "The heuristic score is an inspection aid, not a correctness label.", "",
        f"Rewired comments: {len(reviews)}. Showing up to {args.limit}.", "",
    ]
    for i, item in enumerate(reviews[: args.limit], start=1):
        original_score = (f"{item['original_similarity']:.3f}"
                          if item["original_similarity"] is not None else "excluded by depth bound")
        lines.extend([
            f"## {i}. {item['news_id']}", "",
            f"- News: {compact(item['headline'])}",
            f"- Image: `{item['image_path']}`",
            f"- Generated: {compact(item['comment'])}",
            f"- Original parent: {compact(item['original_parent'])}",
            f"- Selected parent (type {item['selected_parent_type']}): {compact(item['selected_parent'])}",
            f"- Similarity: {original_score} -> {item['selected_similarity']:.3f}; "
            f"reliability: {item['fused_reliability']:.3f}", "",
        ])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"validation_news": len(dataset), "rewired": len(reviews),
                      "shown": min(len(reviews), args.limit), "output": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
