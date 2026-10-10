from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image, ImageOps
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from common import load_config, read_jsonl, resolve_model_reference, resolve_path, write_jsonl
from structure_utils import (
    build_structure_messages,
    load_structure_template,
    parse_structure_output,
    validate_structure_input,
)


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator_structure.yaml"
    parser = argparse.ArgumentParser(description="Iteratively generate comment content and attachment edges.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially written output JSONL, skipping completed news IDs.",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--official-split", choices=["train", "validate", "test"])
    parser.add_argument(
        "--one-per-news", action="store_true",
        help="Keep the richest visible context per news ID; required for detector graph output.",
    )
    parser.add_argument(
        "--min-observed-nodes", type=int, default=0,
        help="Select only contexts containing at least this many real visible comments.",
    )
    parser.add_argument("--nodes-per-graph", type=int)
    parser.add_argument("--context-policy", choices=["iterative", "observed_only"], default="iterative",
                        help="observed_only hides synthetic content and permits only news/real parents.")
    parser.add_argument("--depth-reference", choices=["news_root", "observed_boundary"], default="news_root",
                        help="Bound total depth or only consecutive generated hops after an observed node.")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sample_seed(base: int, sample_id: str, node_number: int, attempt: int) -> int:
    digest = hashlib.sha256(f"{base}:{sample_id}:{node_number}:{attempt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_adapter(config: dict, config_path: Path, adapter_dir: Path, local_only: bool):
    model_path = resolve_model_reference(config_path, config["model"]["name_or_path"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = None
    device_map = None
    if bool(config["model"]["use_4bit"]):
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config["model"]["quant_type"]),
            bnb_4bit_use_double_quant=bool(config["model"]["double_quant"]),
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = {"": 0}
    processor_source = adapter_dir if (adapter_dir / "preprocessor_config.json").is_file() else model_path
    processor = AutoProcessor.from_pretrained(
        processor_source, local_files_only=local_only,
        min_pixels=int(config["model"]["min_pixels"]), max_pixels=int(config["model"]["max_pixels"])
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, local_files_only=local_only, dtype=dtype, quantization_config=quantization,
        device_map=device_map, attn_implementation=str(config["model"]["attn_implementation"])
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, processor


def parent_depth(node_index: int, parents: dict[int, int], observed_boundary: set[int] | None = None) -> int:
    depth = 0
    seen = set()
    while node_index:
        if observed_boundary is not None and node_index in observed_boundary:
            break
        if node_index in seen:
            raise ValueError("cycle in the current generated graph")
        seen.add(node_index)
        depth += 1
        node_index = parents[node_index]
    return depth


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_view(current: dict, original: dict, policy: str) -> dict:
    if policy not in {"iterative", "observed_only"}:
        raise ValueError("Unknown context policy")
    return deepcopy(original if policy == "observed_only" else current)


def verify_resume(output: list, records: list, contract: dict, metadata_path: Path) -> None:
    if not metadata_path.is_file():
        raise ValueError("Missing run metadata: refusing unverified legacy resume; use a new output path")
    stored = json.loads(metadata_path.read_text(encoding="utf-8"))
    if stored["contract"] != contract:
        raise ValueError("Resume contract mismatch: inputs/model/prompt/config/policy changed")
    expected = {str(row["metadata"]["news_id"]): row["sample_id"] for row in records}
    if len(expected) != len(records):
        raise ValueError("Resume requires unique news IDs")
    for row in output:
        if expected.get(str(row["news_id"])) != row["sample_id"]:
            raise ValueError("Completed graph does not match selected context")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen-VL structured generation")
    config, config_path = load_config(args.config)
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    output_path = (
        args.output_jsonl.resolve()
        if args.output_jsonl
        else config_path.parent.parent / "outputs" / "generated_structured_graphs.jsonl"
    )
    input_path = args.input_jsonl.resolve() if args.input_jsonl else data_dir / config["data"]["structure_train_jsonl"]
    records = read_jsonl(input_path)
    if args.official_split:
        records = [
            record for record in records
            if str(record.get("metadata", {}).get("official_split", "")) == args.official_split
        ]
    if args.min_observed_nodes > 0:
        records = [
            record for record in records
            if len(record.get("model_input", {}).get("observed_nodes", [])) >= args.min_observed_nodes
        ]
    if args.one_per_news:
        selected = {}
        for record in records:
            news_id = str(record["metadata"]["news_id"])
            visible = len(record["model_input"]["observed_nodes"])
            key = (visible, record["sample_id"])
            if news_id not in selected or key > selected[news_id][0]:
                selected[news_id] = (key, record)
        records = [selected[news_id][1] for news_id in sorted(selected)]
    if args.limit > 0:
        records = records[: args.limit]
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"Preserving existing output; use --resume: {output_path}")
    adapter_dir = args.adapter_dir.resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(adapter_dir)
    generation = config["generation"]
    requested = int(args.nodes_per_graph if args.nodes_per_graph is not None else generation["nodes_per_graph"])
    max_depth = int(generation["max_generated_depth"])
    max_attempts = int(generation["max_attempts_per_node"])
    if min(requested, max_depth, max_attempts) < 1:
        raise ValueError("Generation count, depth and retry limits must be positive")
    template_path = resolve_path(config_path, config["prompt"]["template_path"])
    contract = {"version": 1, "config_sha256": fingerprint(config_path),
                "input_sha256": fingerprint(input_path), "template_sha256": fingerprint(template_path),
                "script_sha256": fingerprint(Path(__file__)),
                "structure_utils_sha256": fingerprint(Path(__file__).with_name("structure_utils.py")),
                "adapter_dir": str(adapter_dir),
                "adapter_files": {p.name: fingerprint(p) for p in sorted(adapter_dir.iterdir())
                                  if p.is_file() and (p.suffix in {".json", ".safetensors", ".bin", ".jinja"})},
                "selected_samples": [r["sample_id"] for r in records],
                "context_policy": args.context_policy, "depth_reference": args.depth_reference,
                "nodes_per_graph": requested}
    metadata_path = output_path.with_suffix(".run.json")
    output = read_jsonl(output_path) if args.resume and output_path.is_file() else []
    if output or metadata_path.exists():
        verify_resume(output, records, contract, metadata_path)
    else:
        atomic_json(metadata_path, {"contract": contract, "completed_graphs": 0})
    if any(row.get("generation_config", {}).get("depth_reference", "news_root") != args.depth_reference
           for row in output):
        raise ValueError("Cannot resume an output generated under a different depth policy")
    if any(row.get("generation_config", {}).get("context_policy", "iterative") != args.context_policy
           for row in output):
        raise ValueError("Cannot resume an output generated under a different context policy")
    completed_news = {str(item.get("news_id", "")) for item in output}
    if len(completed_news) != len(output):
        raise ValueError(f"Duplicate or missing news_id in resumable output: {output_path}")
    if args.official_split and any(
        str(item.get("official_split", "")) != args.official_split for item in output
    ):
        raise ValueError("Existing output contains a different official split")
    records = [record for record in records if str(record["metadata"]["news_id"]) not in completed_news]
    if not records:
        if output:
            print(json.dumps({"graphs": len(output), "output": str(output_path), "resumed": True}, ensure_ascii=False))
            return
        raise RuntimeError("No structural contexts selected")
    started = time.perf_counter()
    initial_count = len(output)
    torch.cuda.reset_peak_memory_stats()
    model, processor = load_adapter(config, config_path, adapter_dir, bool(args.local_files_only or config["model"]["local_files_only"]))
    template = load_structure_template(template_path)
    for record in records:
        model_input = deepcopy(record["model_input"])
        validate_structure_input(model_input)
        original_input = deepcopy(model_input)
        with Image.open(model_input["image_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB").copy()
        parents = {int(node["node_index"]): int(node["parent_node_index"]) for node in model_input["observed_nodes"]}
        # Snapshot the original observed nodes, not the mutable graph. Existing
        # real depth must not consume the allowed number of synthetic hops.
        boundary = set(parents) if args.depth_reference == "observed_boundary" else None
        observed_ids = {
            int(index): str(comment_id)
            for index, comment_id in record.get("metadata", {}).get("observed_node_comment_ids", {}).items()
        }
        graph_nodes = [{"node_index": 0, "node_type": "news", "content": model_input["news_text"]}]
        graph_nodes.extend(
            {"node_index": int(node["node_index"]), "node_type": "real_comment", "content": node["content"],
             "source_comment_id": observed_ids.get(int(node["node_index"]), "")}
            for node in model_input["observed_nodes"]
        )
        edges = [
            {"source": int(node["parent_node_index"]), "target": int(node["node_index"]), "edge_type": 0}
            for node in model_input["observed_nodes"]
        ]
        generated_nodes, failures = [], []
        seen_contents = {str(node["content"]).strip().casefold() for node in model_input["observed_nodes"]}
        for number in range(requested):
            accepted = None
            visible_input = generation_view(model_input, original_input, args.context_policy)
            visible_indices = {int(node["node_index"]) for node in visible_input["observed_nodes"]}
            # Bound depth relative to the declared origin; both policies still
            # require an existing parent and append only acyclic new nodes.
            allowed_parents = [
                index for index in sorted(parents)
                if index in visible_indices and parent_depth(index, parents, boundary) + 1 <= max_depth
            ]
            allowed_parents = [0, *allowed_parents]
            for attempt in range(max_attempts):
                messages = build_structure_messages(
                    template, visible_input, allowed_parent_indices=allowed_parents
                )
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt", truncation=False).to(model.device)
                if inputs["input_ids"].shape[1] > int(config["training"]["max_sequence_length"]):
                    failures.append({"generated_position": number + 1, "attempt": attempt + 1,
                                     "reason": "context_overflow_no_truncation", "raw_model_output": ""})
                    break
                seed = sample_seed(int(config["seed"]), record["sample_id"], number, attempt)
                torch.manual_seed(seed)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, do_sample=True, temperature=float(generation["temperature"]), top_p=float(generation["top_p"]),
                        repetition_penalty=float(generation["repetition_penalty"]), max_new_tokens=int(generation["max_new_tokens"]),
                        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
                    )
                raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
                try:
                    candidate = parse_structure_output(raw, len(visible_input["observed_nodes"]))
                    if candidate["content"].casefold() in seen_contents:
                        raise ValueError("duplicate_content")
                    if candidate["parent_node_index"] not in allowed_parents:
                        raise ValueError("parent_not_in_allowed_depth_bounded_set")
                    if parent_depth(candidate["parent_node_index"], parents, boundary) + 1 > max_depth:
                        raise ValueError("maximum_depth_exceeded")
                    accepted = {**candidate, "raw_model_output": raw, "seed": seed, "attempt": attempt + 1}
                    break
                except ValueError as exc:
                    failures.append({"generated_position": number + 1, "attempt": attempt + 1, "reason": str(exc), "raw_model_output": raw})
            if accepted is None:
                continue
            index = len(model_input["observed_nodes"]) + 1
            node = {"node_index": index, "parent_node_index": accepted["parent_node_index"], "content": accepted["content"]}
            model_input["observed_nodes"].append(node)
            parents[index] = accepted["parent_node_index"]
            seen_contents.add(accepted["content"].casefold())
            generated_node_id = f"generated:{record['sample_id']}:{index}"
            parent_comment_id = observed_ids.get(accepted["parent_node_index"], "news")
            generated_nodes.append({"node_index": index, "generated_node_id": generated_node_id,
                                    "parent_comment_id": parent_comment_id, **accepted})
            observed_ids[index] = generated_node_id
            graph_nodes.append({"node_index": index, "node_type": "generated_comment", "content": accepted["content"],
                                "generated_node_id": generated_node_id, "parent_comment_id": parent_comment_id})
            edges.append({"source": accepted["parent_node_index"], "target": index, "edge_type": 1})
        output.append({
            "sample_id": record["sample_id"], "news_id": record["metadata"]["news_id"],
            "official_split": record["metadata"]["official_split"], "image_path": record["model_input"]["image_path"],
            "graph_nodes": graph_nodes, "edges": edges, "generated_nodes": generated_nodes,
            "generation_failures": failures, "adapter_dir": str(adapter_dir),
            "generation_config": {"context_policy": args.context_policy, "seed": int(config["seed"]), "depth_reference": args.depth_reference, "nodes_per_graph": requested, "max_attempts_per_node": max_attempts, "max_generated_depth": max_depth, "temperature": generation["temperature"], "top_p": generation["top_p"], "repetition_penalty": generation["repetition_penalty"], "max_new_tokens": generation["max_new_tokens"]},
            "parent_selection_policy": ("model_predicted_news_or_real_only" if args.context_policy == "observed_only" else
                                        "model_predicted_with_generated_depth_bounded_candidates" if boundary is not None
                                        else "model_predicted_with_depth_bounded_candidates"),
            "label_used_for_generation": False,
            "time_interpretation": "iterative generation order only; no comment timestamps are available",
        })
        # Persist after every completed news graph.  This makes a long VLM
        # generation run safely resumable instead of losing all prior work on
        # interruption or a transient CUDA failure.
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        write_jsonl(temporary, output)
        temporary.replace(output_path)
        atomic_json(metadata_path, {"contract": contract, "completed_graphs": len(output),
                                   "session_graphs": len(output) - initial_count,
                                   "session_seconds": time.perf_counter() - started,
                                   "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3})
        print(f"generated sample_id={record['sample_id']} nodes={len(generated_nodes)}/{requested}", flush=True)
    print(json.dumps({"graphs": len(output), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
