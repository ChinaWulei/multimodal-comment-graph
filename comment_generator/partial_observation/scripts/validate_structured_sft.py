from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from common import load_config, read_jsonl, resolve_path


FORBIDDEN_MODEL_INPUT_KEYS = {"label", "2_way_label", "3_way_label", "6_way_label", "parent_id"}


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator_structure.yaml"
    parser = argparse.ArgumentParser(description="Validate structure-aware SFT records and image inputs.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Optional explicit SFT file. Defaults to data.structure_train_jsonl from the config.",
    )
    parser.add_argument(
        "--expected-official-split",
        choices=("train", "validate", "test"),
        default=None,
        help="Require every record to come from this official split.",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    if args.input_jsonl:
        path = args.input_jsonl.resolve()
    else:
        split_key = {
            "validate": "structure_validate_jsonl",
            "test": "structure_test_jsonl",
        }.get(args.expected_official_split, "structure_train_jsonl")
        path = data_dir / config["data"][split_key]
    records = read_jsonl(path)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"No records in {path}")
    ids: set[str] = set()
    root_targets = reply_targets = 0
    max_nodes = 0
    for record in records:
        expected = {"sample_id", "metadata", "model_input", "target", "target_json"}
        if set(record) != expected:
            raise ValueError(f"Unexpected record schema: {record.get('sample_id')}")
        if record["sample_id"] in ids:
            raise ValueError(f"Duplicate sample ID: {record['sample_id']}")
        ids.add(record["sample_id"])
        if args.expected_official_split and record["metadata"].get("official_split") != args.expected_official_split:
            raise ValueError(
                f"Split leakage: {record['sample_id']} has official_split="
                f"{record['metadata'].get('official_split')!r}, expected {args.expected_official_split!r}"
            )
        model_input = record["model_input"]
        if set(model_input) != {"image_path", "news_text", "observed_nodes"}:
            raise ValueError(f"Forbidden model input field: {record['sample_id']}")
        # Inspect schema keys only. Comment text may legitimately contain a word such as
        # "label" and must not trigger a false leakage alarm.
        def walk_keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield str(key)
                    yield from walk_keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from walk_keys(child)

        if any(key.casefold() in FORBIDDEN_MODEL_INPUT_KEYS for key in walk_keys(model_input)):
            raise ValueError(f"Possible label/raw-parent leakage in model input schema: {record['sample_id']}")
        observed = model_input["observed_nodes"]
        indices = [int(node["node_index"]) for node in observed]
        if indices != list(range(1, len(observed) + 1)):
            raise ValueError(f"Observed node indices are not contiguous: {record['sample_id']}")
        for node in observed:
            parent = int(node["parent_node_index"])
            if parent < 0 or parent >= int(node["node_index"]):
                raise ValueError(f"Invalid visible edge: {record['sample_id']}")
        parent = int(record["target"]["parent_node_index"])
        if parent < 0 or parent > len(observed):
            raise ValueError(f"Target parent not in visible graph: {record['sample_id']}")
        if json.loads(record["target_json"]) != record["target"]:
            raise ValueError(f"Target JSON mismatch: {record['sample_id']}")
        if not str(record["target"]["content"]).strip():
            raise ValueError(f"Empty target content: {record['sample_id']}")
        with Image.open(model_input["image_path"]) as image:
            image.verify()
        max_nodes = max(max_nodes, len(observed))
        if parent == 0:
            root_targets += 1
        else:
            reply_targets += 1
    result = {
        "records_checked": len(records),
        "unique_sample_ids": len(ids),
        "root_reply_targets": root_targets,
        "reply_targets": reply_targets,
        "max_visible_nodes": max_nodes,
        "model_input_fields": ["image_path", "news_text", "observed_nodes"],
        "label_leakage_check": "passed",
        "parent_supervision_check": "passed",
        "image_decode_check": "passed",
        "expected_official_split": args.expected_official_split,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
