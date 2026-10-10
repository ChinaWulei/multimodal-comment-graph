"""Inspect conservative root fallback before fitting a classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch_geometric.loader import DataLoader

from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform
from comment_graph.model import CommentGraphMFCL


def compact(value, limit=240):
    return " ".join(str(value).split())[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    config.graph.message_passing = "typed_rgcn"
    config.graph.structure_mode = "observed_tree"
    config.graph.pooling_mode = "mean"
    config.graph.generated_reliability["mode"] = "confidence"
    tokenizer, transform = make_tokenizer_and_transform(config)
    dataset = make_dataset(config, config_path, "validate", "real_generated", tokenizer, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CommentGraphMFCL(config, "real_generated").to(device).eval()
    rows = []
    offset = 0
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=8, shuffle=False):
            news_ids = list(batch.news_id)
            output = model(batch.to(device))
            for item in output["generated_reliability_diagnostics"]:
                graph_index = int(batch.batch[item["node_index"]])
                first = int(batch.ptr[graph_index])
                local = {**item, "node_index": item["node_index"] - first,
                         "original_parent": item["original_parent"] - first,
                         "selected_parent": item["selected_parent"] - first,
                         "news_id": news_ids[graph_index]}
                record = dataset.records[offset + graph_index]
                texts, _, _ = dataset._nodes_and_edges(local["news_id"])
                contents = [record["news_text"], *texts]
                local.update({"headline": record["news_text"],
                              "image_path": record["image_path"],
                              "generated": contents[local["node_index"]],
                              "original_parent_text": contents[local["original_parent"]],
                              "selected_parent_text": contents[local["selected_parent"]]})
                rows.append(local)
            offset += len(news_ids)
    fallbacks = [row for row in rows if row["rewired"]]
    summary = {"split": "validate", "news": len(dataset), "generated_nodes": len(rows),
               "root_fallbacks": len(fallbacks),
               "mean_edge_confidence": sum(row["edge_confidence"] for row in rows) / len(rows),
               "warning": "Fallback is a conservative heuristic; no generated-parent ground truth exists.",
               "records": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Conservative generated-edge fallback review", "",
             f"Validation news: {len(dataset)}; generated nodes: {len(rows)}; root fallbacks: {len(fallbacks)}.",
             "The original parent is generator-predicted, not observed ground truth.", ""]
    for i, row in enumerate(fallbacks, start=1):
        lines.extend([f"## {i}. {row['news_id']}", "",
                      f"- News: {compact(row['headline'])}",
                      f"- Generated: {compact(row['generated'])}",
                      f"- Original parent: {compact(row['original_parent_text'])}",
                      f"- Fallback: news root (percentile {row['original_parent_percentile']:.3f})", ""])
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
