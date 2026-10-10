"""Build label-blind generated-reply review or feature contexts.

This is an annotation aid, not an automatic relevance label or a test result.
Training is the default. Validation is allowed only as unlabeled feature
contexts, never as a human-label queue or gate-fitting source. Test is excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def clean(value: object, limit: int = 500) -> str:
    return " ".join(str(value).split())[:limit]


def examples(path: Path, original_ids: set[str], expected_split: str = "train") -> list[dict]:
    seen_news: set[str] = set()
    seen_nodes: set[str] = set()
    result: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        graph = json.loads(line)
        news_id = str(graph["news_id"])
        if graph["official_split"] != expected_split or news_id in seen_news:
            raise ValueError(f"Wrong split or duplicate news at line {line_number}: {news_id}")
        seen_news.add(news_id)
        image_path = Path(str(graph["image_path"]))
        with Image.open(image_path) as image:
            image.verify()
        nodes = {int(node["node_index"]): node for node in graph["graph_nodes"]}
        if len(nodes) != len(graph["graph_nodes"]) or 0 not in nodes or nodes[0]["node_type"] != "news":
            raise ValueError(f"Invalid node indices: {news_id}")
        generation_edges = {(int(edge["source"]), int(edge["target"]))
                            for edge in graph["edges"] if int(edge["edge_type"]) == 1}
        parents = {int(edge["target"]): int(edge["source"]) for edge in graph["edges"]}
        for position, generated in enumerate(graph["generated_nodes"], 1):
            child = int(generated["node_index"])
            parent = int(generated["parent_node_index"])
            uid = str(generated["generated_node_id"])
            if (uid in seen_nodes or child not in nodes or parent not in nodes or
                    parent >= child or (parent, child) not in generation_edges or
                    nodes[child]["node_type"] != "generated_comment"):
                raise ValueError(f"Invalid generated attachment: {news_id}/{uid}")
            seen_nodes.add(uid)
            ancestors = []
            current = parent
            while current != 0 and len(ancestors) < 3:
                ancestors.append({"node_index": current,
                                  "node_type": nodes[current]["node_type"],
                                  "content": clean(nodes[current]["content"], 300)})
                if current not in parents or parents[current] >= current:
                    raise ValueError(f"Invalid ancestry: {news_id}/{uid}")
                current = parents[current]
            parent_type = nodes[parent]["node_type"]
            result.append({
                "annotation_id": uid,
                "news_id": news_id,
                "official_split": expected_split,
                "source_group": ("original_relevance_filtered" if news_id in original_ids
                                 else "additional_hard_context"),
                "generated_position": position,
                "image_path": str(graph["image_path"]),
                "headline": clean(nodes[0]["content"]),
                "parent_node_index": parent,
                "parent_node_type": parent_type,
                "parent_content": clean(nodes[parent]["content"]),
                "recent_ancestors": ancestors,
                "generated_comment": clean(nodes[child]["content"]),
                "content_relevance": None,
                "parent_fit": None,
                "unsupported_specific_claim": None,
                "review_notes": "",
            })
    return result


def select_balanced(candidates: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in candidates:
        position = "first" if item["generated_position"] == 1 else "later"
        key = (item["source_group"], position, item["parent_node_type"])
        buckets[key].append(item)
    for items in buckets.values():
        rng.shuffle(items)
    selected: list[dict] = []
    used_news: set[str] = set()
    keys = sorted(buckets)
    # First pass maximizes distinct news and diverse context/position/parent
    # strata.  Second pass fills any remaining capacity without replacement.
    for distinct_only in (True, False):
        while len(selected) < limit:
            progress = False
            for key in keys:
                bucket = buckets[key]
                index = next((i for i, item in enumerate(bucket)
                              if not distinct_only or item["news_id"] not in used_news), None)
                if index is None:
                    continue
                item = bucket.pop(index)
                selected.append(item)
                used_news.add(item["news_id"])
                progress = True
                if len(selected) == limit:
                    break
            if not progress:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--original-graphs", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path,
                        help="Optional readable review document; omit for all-node feature contexts")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "validate"], default="train")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    if args.split != "train" and args.output_md:
        raise ValueError("Only the training split may be exported as a human-annotation review")
    original_rows = [json.loads(line) for line in args.original_graphs.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    if any(row["official_split"] != args.split for row in original_rows):
        raise ValueError("Original graph source split mismatch")
    original_ids = {str(row["news_id"]) for row in original_rows}
    if len(original_ids) != len(original_rows):
        raise ValueError("Duplicate news in original graph source")
    candidates = examples(args.graphs, original_ids, args.split)
    chosen = select_balanced(candidates, min(args.limit, len(candidates)), args.seed)
    if len(chosen) != min(args.limit, len(candidates)):
        raise RuntimeError("Could not fill the annotation queue")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in chosen) + "\n",
                                 encoding="utf-8")
    lines = ["# Generated-comment annotation queue (official train only)", "",
             "These are proposed model-generated comments and parent links, **not** verified user replies.",
             "Review the actual image as well as the title. Do not infer a news truth label from the comment.",
             "Fill the matching JSONL fields; leave a field null when genuinely unclear.", "",
             "- `content_relevance`: `relevant`, `partly_relevant`, or `off_topic`.",
             "- `parent_fit`: `good`, `uncertain`, or `wrong` (judge this proposed parent, not whether a unique true parent exists).",
             "- `unsupported_specific_claim`: `true`, `false`, or `null`; mark true only for a specific asserted fact/identity/source not supported by visible news, image, or discussion. `false` does not mean fact-checked true.",
             "- `review_notes`: optional brief reason. Keep the original comment text unchanged.", "",
             f"Sampled {len(chosen)} of {len(candidates)} generated comments from the official training split; seed {args.seed}.", ""]
    if args.output_md:
        for number, item in enumerate(chosen, 1):
            older_context = item["recent_ancestors"][1:]
            older_text = " | ".join(f"{node['node_type']}: {node['content']}" for node in older_context)
            lines.extend([
                f"## {number}. {item['annotation_id']}", "",
                f"- News ID: `{item['news_id']}`; source: `{item['source_group']}`; generation step: {item['generated_position']}",
                f"- Title: {item['headline']}",
                f"- Image: `{item['image_path']}`",
                f"- Proposed parent ({item['parent_node_type']}): {item['parent_content']}",
                f"- Earlier context: {older_text or '(none shown)'}",
                f"- Generated comment: {item['generated_comment']}",
                f"- Labels: content_relevance= ; parent_fit= ; unsupported_specific_claim= ; notes=", "",
            ])
        args.output_md.write_text("\n".join(lines), encoding="utf-8")
    counts = Counter((item["source_group"], "first" if item["generated_position"] == 1 else "later",
                      item["parent_node_type"]) for item in chosen)
    print(json.dumps({"split": args.split, "sampled": len(chosen),
                      "distinct_news": len({item["news_id"] for item in chosen}),
                      "source_sha256": hashlib.sha256(args.graphs.read_bytes()).hexdigest(),
                      "strata": {"/".join(key): count for key, count in sorted(counts.items())},
                      "jsonl": str(args.output_jsonl),
                      "markdown": str(args.output_md) if args.output_md else None},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
