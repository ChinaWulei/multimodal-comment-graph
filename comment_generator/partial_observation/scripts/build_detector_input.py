from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config, read_jsonl, resolve_path, write_jsonl


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator.yaml"
    parser = argparse.ArgumentParser(description="Build downstream fake-news detector input.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    generated = read_jsonl(args.generated_jsonl.resolve())
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    contexts = {
        record["metadata"]["news_id"]: record
        for record in read_jsonl(data_dir / config["data"]["news_contexts_jsonl"])
    }
    label_column = str(config["detector"]["label_column"])
    rows = []
    for record in generated:
        context = contexts.get(record["news_id"])
        if context is None:
            raise KeyError(f"Missing context for {record['news_id']}")
        model_input = context["model_input"]
        rows.append(
            {
                "news_id": record["news_id"],
                "split": context["metadata"]["split"],
                "image_path": model_input["image_path"],
                "news_text": model_input["news_text"],
                "observed_real_comments": model_input["observed_comments"],
                "generated_comments": record["generated_comments"],
                "label": int(context["metadata"]["labels"][label_column]),
                "label_column": label_column,
            }
        )
    output = (
        args.output_jsonl.resolve()
        if args.output_jsonl
        else config_path.parent.parent / "outputs" / "detector_input.jsonl"
    )
    write_jsonl(output, rows)
    print(json.dumps({"rows": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
