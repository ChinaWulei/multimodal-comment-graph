"""Export deterministic validation examples for human review, not model scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def snapshot(path):
    raw = path.read_bytes()
    return [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()], hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--context-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("Preserve existing qualitative reports; use a new output path")
    generated, generated_hash = snapshot(args.input_jsonl)
    contexts, context_hash = snapshot(args.context_jsonl)
    by_sample = {r["sample_id"]: r for r in contexts}
    if len(by_sample) != len(contexts):
        raise ValueError("Duplicate context sample IDs")
    expected_ids = sorted({str(r["metadata"]["news_id"]) for r in contexts
                           if r["metadata"]["official_split"] == "validate" and r["model_input"]["observed_nodes"]})
    if not 0 < args.count <= len(expected_ids):
        raise ValueError("Invalid requested number of validation news")
    by_news = {str(r["news_id"]): r for r in generated}
    if len(by_news) != len(generated) or any(r["official_split"] != "validate" for r in generated):
        raise ValueError("Need unique validation news only")
    selected = expected_ids[:args.count]
    missing = set(selected) - set(by_news)
    if missing:
        raise RuntimeError(f"Selected examples not generated yet: {sorted(missing)}")
    cases = []
    for news_id in selected:
        graph = by_news[news_id]
        context = by_sample[graph["sample_id"]]
        if (str(context["metadata"]["news_id"]) != news_id or
                context["metadata"]["official_split"] != "validate" or
                graph["graph_nodes"][0]["content"] != context["model_input"]["news_text"] or
                Path(graph["image_path"]).resolve() != Path(context["model_input"]["image_path"]).resolve()):
            raise ValueError(f"Source mismatch for {news_id}")
        cases.append({
            "news_id": news_id, "split": "validate", "sample_id": graph["sample_id"],
            "news_text": graph["graph_nodes"][0]["content"], "image_path": graph["image_path"],
            "observed_real_comments": [n for n in graph["graph_nodes"] if n["node_type"] == "real_comment"],
            "generated_comments": graph["generated_nodes"], "edges": graph["edges"],
            "one_actual_next_reply": context["target"],
            "adapter_dir": graph["adapter_dir"], "generation_config": graph["generation_config"],
            "human_review": {"topical_relevance": None, "coherent_reply_to_parent": None,
                             "image_grounding": None, "unsupported_assertions": None, "notes": ""},
        })
    report = {
        "selection": "First N unique news IDs sorted from the validation contexts with >=1 observed node; not selected by output quality.",
        "input_snapshot_graph_count": len(generated),
        "input_sha256": generated_hash, "context_sha256": context_hash,
        "scope": "Qualitative review only; no classifier setting or dataset sample is changed.",
        "cautions": ["Structural validity does not establish semantic quality or fact-checking ability.",
                     "One actual next reply is not the only valid continuation and was not placed in the generator prompt.",
                     "Visible image paths and image-related words do not prove causal image use; no image ablation is claimed.",
                     "Human review fields are deliberately unscored until reviewed."],
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "news_ids": selected, "output": str(args.output_json.resolve())}, indent=2))


if __name__ == "__main__":
    main()
