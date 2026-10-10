from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from string import Formatter
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from common import load_config, read_jsonl, resolve_model_reference, resolve_path, write_jsonl


JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
FORBIDDEN_META_PHRASES = (
    "thread",
    "discussion",
    "conversation",
    "comment",
    "discussing",
    "discusses",
    "discussed",
    "users think",
    "user discusses",
    "user shares",
    "user says",
    "some users",
)
RELEVANT_TYPES = {"reaction", "question", "doubt", "context", "humor", "discussion", "other"}
REJECTED_TYPES = {"off_topic", "spam", "meaningless"}
PERSONAL_EXPERIENCE = re.compile(
    r"\b(?:i've|i have|i was|i am|i'm|i live|i saw|i had|i used|my|mine|we|our)\b",
    re.IGNORECASE,
)
AUTHENTICITY_ASSERTION = re.compile(
    r"\b(?:fake|hoax|fabricated|photoshopped)\b", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator_tree.yaml"
    parser = argparse.ArgumentParser(description="Filter and distill comment reply trees with a VLM.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--limit-news", type=int, default=0)
    parser.add_argument("--limit-threads", type=int, default=0)
    parser.add_argument(
        "--max-threads-per-news", type=int, default=0,
        help="Keep only the first N roots (already ordered by ups) per news item; 0 keeps all.",
    )
    parser.add_argument(
        "--news-id",
        action="append",
        default=[],
        help="Process only this news ID; repeat to select multiple news items.",
    )
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--pool-jsonl", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rebuild-pool-only",
        action="store_true",
        help="Reapply quality gates to existing teacher results without loading the model.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8").strip()
    fields = {name for _, name, _, _ in Formatter().parse(template) if name}
    if fields != {"news_text", "thread_text"}:
        raise ValueError(f"Unexpected distillation template fields: {sorted(fields)}")
    forbidden = ["2_way_label", "3_way_label", "6_way_label", "{label"]
    if any(value in template.casefold() for value in forbidden):
        raise ValueError("Label field detected in thread distillation prompt")
    return template


def flatten_thread(root: dict[str, Any], max_nodes: int, max_chars: int) -> tuple[str, list[str], bool]:
    lines: list[str] = []
    source_ids: list[str] = []
    seen: set[str] = set()
    stack = [(root, 0)]
    truncated = False
    while stack and len(source_ids) < max_nodes:
        node, depth = stack.pop()
        comment_id = str(node["comment_id"])
        if comment_id in seen:
            truncated = True
            continue
        seen.add(comment_id)
        body = " ".join(str(node["body"]).split())
        prefix = "  " * min(depth, 8) + ("- " if depth else "Root: ")
        candidate = prefix + body
        used = sum(len(line) + 1 for line in lines)
        if used + len(candidate) > max_chars:
            # A single Reddit comment can itself exceed the teacher budget.
            # Keep a bounded prefix for relevance judgement; dropping it would
            # send an empty tree and make the model invent meta-commentary.
            available = max_chars - used
            if not lines and available > len(prefix) + 1:
                snippet = body[: available - len(prefix) - 1].rstrip()
                if snippet:
                    lines.append(prefix + snippet)
                    source_ids.append(comment_id)
            truncated = True
            break
        lines.append(candidate)
        source_ids.append(comment_id)
        for child in reversed(node.get("children", [])):
            stack.append((child, depth + 1))
    if stack:
        truncated = True
    return "\n".join(lines), source_ids, truncated


def parse_result(text: str) -> dict[str, Any]:
    match = JSON_OBJECT.search(text.strip())
    if not match:
        raise ValueError("Teacher output does not contain a JSON object")
    value = json.loads(match.group(0))
    required = {"relevant", "relevance_score", "comment_type", "distilled_comment"}
    if set(value) != required:
        raise ValueError(f"Unexpected teacher output keys: {sorted(value)}")
    if not isinstance(value["relevant"], bool):
        raise ValueError("relevant must be boolean")
    score = float(value["relevance_score"])
    if not 0 <= score <= 1:
        raise ValueError("relevance_score must be in [0, 1]")
    comment = " ".join(str(value["distilled_comment"]).split()).strip()
    if value["relevant"] and not comment:
        raise ValueError("Relevant thread has an empty distilled_comment")
    if value["relevant"] and any(
        phrase in comment.casefold() for phrase in FORBIDDEN_META_PHRASES
    ):
        raise ValueError("Distilled comment uses forbidden meta-language")
    if value["relevant"] and len(comment.split()) > 40:
        raise ValueError("Distilled comment exceeds 40 words")
    if value["relevant"] and re.search(r"[.!?]\s+\S", comment):
        raise ValueError("Distilled comment contains more than one sentence")
    if not value["relevant"]:
        comment = ""
    comment_type = str(value["comment_type"]).strip()
    allowed_types = RELEVANT_TYPES if value["relevant"] else REJECTED_TYPES
    if comment_type not in allowed_types:
        raise ValueError(f"Invalid comment_type: {comment_type}")
    return {
        "relevant": value["relevant"],
        "relevance_score": score,
        "comment_type": comment_type,
        "distilled_comment": comment,
    }


def load_model(config: dict, config_path: Path, local_only: bool):
    teacher = config["thread_preprocessing"]["teacher"]
    model_path = resolve_model_reference(config_path, teacher["name_or_path"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = None
    device_map = None
    if bool(teacher["use_4bit"]):
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(teacher["quant_type"]),
            bnb_4bit_use_double_quant=bool(teacher["double_quant"]),
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = {"": 0}
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=local_only,
        min_pixels=int(teacher["min_pixels"]),
        max_pixels=int(teacher["max_pixels"]),
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=local_only,
        dtype=dtype,
        quantization_config=quantization,
        device_map=device_map,
        attn_implementation=str(teacher["attn_implementation"]),
    )
    model.eval()
    return model, processor, str(model_path)


def distilled_quality_rejection(comment: str) -> str:
    if PERSONAL_EXPERIENCE.search(comment):
        return "first_person_experience_or_identity"
    if AUTHENTICITY_ASSERTION.search(comment):
        return "explicit_authenticity_assertion"
    if any(phrase in comment.casefold() for phrase in FORBIDDEN_META_PHRASES):
        return "meta_language"
    if re.search(r"[.!?]\s+\S", comment):
        return "multiple_sentences"
    if len(comment.split()) > 40:
        return "over_40_words"
    return ""


def build_pool(
    results: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_bodies: dict[tuple[str, str], set[str]] = {}
    quality_rejections: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "ok" or not result.get("relevant"):
            continue
        if float(result["relevance_score"]) < threshold:
            continue
        body = str(result["distilled_comment"]).strip()
        reason = distilled_quality_rejection(body)
        if reason:
            quality_rejections.append(
                {
                    "news_id": result["news_id"],
                    "thread_root_id": result["thread_root_id"],
                    "reason": reason,
                    "distilled_comment": body,
                }
            )
            continue
        key = (result["news_id"], result["split"])
        body_key = body.casefold()
        if body_key in seen_bodies.setdefault(key, set()):
            continue
        seen_bodies[key].add(body_key)
        grouped.setdefault(key, []).append(
            {
                "comment_id": f"thread_{result['thread_root_id']}",
                "body": body,
                "author": "thread_distiller",
                "ups": result["root_ups"],
                "parent_id": result["news_id"],
                "submission_id": result["news_id"],
                "is_top_level": True,
                "source_kind": "distilled_reply_tree",
                "source_comment_ids": result["source_comment_ids"],
                "thread_root_id": result["thread_root_id"],
                "relevance_score": result["relevance_score"],
                "comment_type": result["comment_type"],
                "thread_truncated_for_teacher": result["thread_truncated_for_teacher"],
            }
        )
    pool = [
        {"news_id": news_id, "split": split, "comments": comments}
        for (news_id, split), comments in sorted(grouped.items())
    ]
    return pool, quality_rejections


def main() -> None:
    args = parse_args()
    if not args.rebuild_pool_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multimodal thread distillation")
    config, config_path = load_config(args.config)
    data = config["data"]
    preprocessing = config["thread_preprocessing"]
    output_dir = resolve_path(config_path, data["output_dir"])
    trees = read_jsonl(output_dir / data["raw_comment_trees_jsonl"])
    if args.news_id:
        selected_ids = set(args.news_id)
        trees = [record for record in trees if record["news_id"] in selected_ids]
        missing = selected_ids - {record["news_id"] for record in trees}
        if missing:
            raise ValueError(f"Unknown news IDs: {sorted(missing)}")
    if args.limit_news > 0:
        trees = trees[: args.limit_news]
    template = load_template(resolve_path(config_path, preprocessing["prompt_template_path"]))
    output_path = (
        args.output_jsonl.resolve()
        if args.output_jsonl
        else output_dir / data["thread_distillation_results_jsonl"]
    )
    pool_path = (
        args.pool_jsonl.resolve()
        if args.pool_jsonl
        else output_dir / data["distilled_comment_pool_jsonl"]
    )
    existing = read_jsonl(output_path) if args.resume and output_path.is_file() else []
    threshold = float(preprocessing["min_relevance_score"])
    if args.rebuild_pool_only:
        if not output_path.is_file():
            raise RuntimeError(f"Missing teacher results: {output_path}")
        results = read_jsonl(output_path)
        pool, quality_rejections = build_pool(results, threshold)
        write_jsonl(pool_path, pool)
        audit_path = output_dir / "distilled_pool_quality_audit.json"
        audit = {
            "teacher_results": len(results),
            "quality_rejections": len(quality_rejections),
            "rejection_reason_counts": {
                reason: sum(item["reason"] == reason for item in quality_rejections)
                for reason in sorted({item["reason"] for item in quality_rejections})
            },
            "rejection_examples": quality_rejections[:20],
            "pool_news": len(pool),
            "pool_comments": sum(len(record["comments"]) for record in pool),
            "pool": str(pool_path),
        }
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    # Both accepted outputs and content-policy rejections are terminal for this
    # preprocessing pass.  Retrying rejected content on --resume would only
    # duplicate audit rows and spend another VLM call without changing data.
    completed = {
        (record["news_id"], record["thread_root_id"])
        for record in existing
        if record.get("status") in {"ok", "rejected"}
    }
    model, processor, model_path = load_model(
        config,
        config_path,
        bool(args.local_files_only or preprocessing["teacher"]["local_files_only"]),
    )
    max_nodes = int(preprocessing["max_thread_nodes"])
    max_chars = int(preprocessing["max_thread_chars"])
    max_new_tokens = int(preprocessing["teacher"]["max_new_tokens"])
    results = list(existing)
    processed_now = 0
    for news in trees:
        with Image.open(news["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
        roots = news["roots"]
        if args.max_threads_per_news > 0:
            roots = roots[: args.max_threads_per_news]
        for root in roots:
            key = (news["news_id"], root["comment_id"])
            if key in completed:
                continue
            if args.limit_threads > 0 and processed_now >= args.limit_threads:
                break
            thread_text, source_ids, truncated = flatten_thread(root, max_nodes, max_chars)
            prompt = template.format(news_text=news["news_text"], thread_text=thread_text)
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You filter and distill social-media discussions. Follow the JSON schema exactly.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            raw_output = ""
            try:
                parsed = None
                parse_error = None
                for attempt in range(3):
                    chat = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(
                        text=[chat], images=[image], return_tensors="pt", truncation=True
                    ).to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=(
                                processor.tokenizer.pad_token_id
                                if processor.tokenizer.pad_token_id is not None
                                else processor.tokenizer.eos_token_id
                            ),
                        )
                    raw_output = processor.batch_decode(
                        generated[:, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )[0].strip()
                    try:
                        parsed = parse_result(raw_output)
                        break
                    except ValueError as exc:
                        parse_error = exc
                        if attempt < 2:
                            messages.extend(
                                [
                                    {
                                        "role": "assistant",
                                        "content": [{"type": "text", "text": raw_output}],
                                    },
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": (
                                                    "Correct the JSON. The distilled_comment must sound like a user "
                                                    "posting directly under the news, contain exactly one sentence "
                                                    "of at most 40 words, and "
                                                    "must not mention a discussion, thread, comments, or users. "
                                                    "State the point directly: write 'Owning a modem can save money', "
                                                    "not 'Users discuss how owning a modem can save money'."
                                                ),
                                            }
                                        ],
                                    },
                                ]
                            )
                if parsed is None:
                    raise parse_error or ValueError("Unable to parse teacher output")
                record = {
                    "status": "ok",
                    "news_id": news["news_id"],
                    "split": news["split"],
                    "image_path": news["image_path"],
                    "thread_root_id": root["comment_id"],
                    "root_ups": root["ups"],
                    "source_comment_ids": source_ids,
                    "thread_node_count_sent": len(source_ids),
                    "thread_truncated_for_teacher": truncated,
                    "teacher_model": model_path,
                    **parsed,
                    "raw_teacher_output": raw_output,
                }
            except ValueError as exc:
                # A syntactically valid but policy-violating teacher answer is
                # a content-quality rejection, not an infrastructure failure.
                # Keep it auditable and ensure build_pool never promotes it.
                record = {
                    "status": "rejected",
                    "news_id": news["news_id"],
                    "split": news["split"],
                    "image_path": news["image_path"],
                    "thread_root_id": root["comment_id"],
                    "root_ups": root["ups"],
                    "source_comment_ids": source_ids,
                    "thread_node_count_sent": len(source_ids),
                    "thread_truncated_for_teacher": truncated,
                    "teacher_model": model_path,
                    "raw_teacher_output": raw_output,
                    "rejection_reason": f"{type(exc).__name__}: {exc}",
                }
            except Exception as exc:
                record = {
                    "status": "error",
                    "news_id": news["news_id"],
                    "split": news["split"],
                    "image_path": news["image_path"],
                    "thread_root_id": root["comment_id"],
                    "root_ups": root["ups"],
                    "source_comment_ids": source_ids,
                    "thread_node_count_sent": len(source_ids),
                    "thread_truncated_for_teacher": truncated,
                    "teacher_model": model_path,
                    "raw_teacher_output": raw_output,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(record)
            processed_now += 1
            print(
                f"distilled news_id={news['news_id']} root={root['comment_id']} "
                f"status={record['status']} relevant={record.get('relevant')}"
            )
            write_jsonl(output_path, results)
        if args.limit_threads > 0 and processed_now >= args.limit_threads:
            break

    pool, quality_rejections = build_pool(results, threshold)
    write_jsonl(pool_path, pool)
    summary = {
        "processed_now": processed_now,
        "results_total": len(results),
        "successful": sum(record.get("status") == "ok" for record in results),
        "errors": sum(record.get("status") == "error" for record in results),
        "relevant_above_threshold": sum(
            record.get("status") == "ok"
            and bool(record.get("relevant"))
            and float(record.get("relevance_score", 0)) >= threshold
            for record in results
        ),
        "news_in_distilled_pool": len(pool),
        "distilled_comments_in_pool": sum(len(record["comments"]) for record in pool),
        "quality_rejections": len(quality_rejections),
        "output": str(output_path),
        "pool": str(pool_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
