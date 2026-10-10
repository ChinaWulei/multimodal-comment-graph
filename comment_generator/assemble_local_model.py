from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Assemble manually downloaded Qwen shards into a local model directory."
    )
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "models" / "Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--copy-weights",
        action="store_true",
        help="Copy large shards instead of creating same-volume hard links.",
    )
    return parser.parse_args()


def locate_metadata(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
    else:
        path = Path(snapshot_download(MODEL_ID, local_files_only=True)).resolve()
    required = ["config.json", "model.safetensors.index.json", "preprocessor_config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Metadata directory {path} is missing: {missing}")
    return path


def inspect_shards(weights_dir: Path, shard_names: list[str]) -> dict[str, str]:
    actual_map: dict[str, str] = {}
    for shard_name in shard_names:
        shard_path = weights_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing downloaded shard: {shard_path}")
        with safe_open(str(shard_path), framework="pt") as shard:
            for tensor_name in shard.keys():
                if tensor_name in actual_map:
                    raise ValueError(f"Tensor appears in more than one shard: {tensor_name}")
                actual_map[tensor_name] = shard_name
    return actual_map


def place_weight(source: Path, destination: Path, copy_weights: bool) -> str:
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(
                f"Existing destination has the wrong size: {destination}. Remove it explicitly first."
            )
        return "existing"
    if not copy_weights:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def main() -> None:
    args = parse_args()
    weights_dir = args.weights_dir.resolve()
    metadata_dir = locate_metadata(args.metadata_dir)
    output_dir = args.output_dir.resolve()
    original_index_path = metadata_dir / "model.safetensors.index.json"
    original_index = json.loads(original_index_path.read_text(encoding="utf-8"))
    expected_map = original_index["weight_map"]
    shard_names = sorted(set(expected_map.values()))
    actual_map = inspect_shards(weights_dir, shard_names)

    expected_keys = set(expected_map)
    actual_keys = set(actual_map)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            f"Downloaded shards do not match {MODEL_ID}: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_metadata = []
    for source in metadata_dir.iterdir():
        if not source.is_file() or source.name.endswith(".safetensors"):
            continue
        if source.name == "model.safetensors.index.json":
            continue
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        copied_metadata.append(source.name)

    placement = {}
    for shard_name in shard_names:
        placement[shard_name] = place_weight(
            weights_dir / shard_name,
            output_dir / shard_name,
            args.copy_weights,
        )

    rebuilt_index = {
        "metadata": original_index.get("metadata", {}),
        "weight_map": dict(sorted(actual_map.items())),
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(rebuilt_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    wrong_shard_entries = sum(
        expected_map[name] != actual_map[name] for name in expected_map
    )
    report = {
        "model_id": MODEL_ID,
        "metadata_source": str(metadata_dir),
        "weights_source": str(weights_dir),
        "output_dir": str(output_dir),
        "tensor_count": len(actual_map),
        "missing_tensors": 0,
        "extra_tensors": 0,
        "rebuilt_index_entries": wrong_shard_entries,
        "shards": {
            name: {
                "bytes": (weights_dir / name).stat().st_size,
                "placement": placement[name],
            }
            for name in shard_names
        },
        "metadata_files": sorted(copied_metadata),
    }
    (output_dir / "assembly_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
