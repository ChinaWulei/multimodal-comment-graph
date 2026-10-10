from __future__ import annotations

import argparse
import json
import random
from difflib import SequenceMatcher
from pathlib import Path

from common import load_config, read_jsonl, resolve_path


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator.yaml"
    parser = argparse.ArgumentParser(description="Evaluate generated comment text.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--compute-bertscore", action="store_true")
    parser.add_argument("--example-count", type=int, default=20)
    return parser.parse_args()


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    generated = read_jsonl(args.generated_jsonl.resolve())
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    gold = {
        record["news_id"]: record
        for record in read_jsonl(data_dir / config["data"]["gold_targets_jsonl"])
    }
    phrases = [str(value).casefold() for value in config["evaluation"]["hallucination_phrases"]]
    flat = []
    per_news = []
    bert_candidates = []
    bert_references = []
    for record in generated:
        news_id = record["news_id"]
        targets = [item["body"] for item in gold.get(news_id, {}).get("target_comments", [])]
        observed = record["observed_comments"]
        comments = [str(value).strip() for value in record["generated_comments"]]
        for comment in comments:
            row = {
                "news_id": news_id,
                "comment": comment,
                "empty": not bool(comment),
                "length_chars": len(comment),
                "exact_observed": comment.casefold() in {value.casefold() for value in observed},
                "exact_target": comment.casefold() in {value.casefold() for value in targets},
                "max_observed_similarity": max(
                    (similarity(comment, value) for value in observed), default=0.0
                ),
                "max_target_similarity": max(
                    (similarity(comment, value) for value in targets), default=0.0
                ),
                "hallucination_phrase": next(
                    (phrase for phrase in phrases if phrase in comment.casefold()), ""
                ),
            }
            flat.append(row)
            if targets:
                best_target = max(targets, key=lambda value: similarity(comment, value))
                bert_candidates.append(comment)
                bert_references.append(best_target)
        per_news.append(
            {
                **record,
                "real_target_comments": targets,
                "unique_generated": len({value.casefold() for value in comments}),
            }
        )
    if not flat:
        raise RuntimeError("No generated comments to evaluate")

    bertscore_mean = None
    bertscore_status = "not requested"
    if args.compute_bertscore:
        try:
            from bert_score import score

            _, _, f1 = score(
                bert_candidates,
                bert_references,
                lang="en",
                verbose=False,
            )
            bertscore_mean = float(f1.mean().item())
            bertscore_status = "computed against the lexically closest real target per output"
        except Exception as exc:
            bertscore_status = f"unavailable: {type(exc).__name__}: {exc}"

    total = len(flat)
    unique = len({row["comment"].casefold() for row in flat if row["comment"]})
    metrics = {
        "news": len(per_news),
        "generated_comments": total,
        "unique_rate": unique / total,
        "duplicate_rate": 1.0 - unique / total,
        "mean_length_chars": sum(row["length_chars"] for row in flat) / total,
        "empty_rate": sum(row["empty"] for row in flat) / total,
        "exact_observed_rate": sum(row["exact_observed"] for row in flat) / total,
        "exact_target_rate": sum(row["exact_target"] for row in flat) / total,
        "mean_max_observed_sequence_similarity": sum(
            row["max_observed_similarity"] for row in flat
        )
        / total,
        "mean_max_target_sequence_similarity": sum(
            row["max_target_similarity"] for row in flat
        )
        / total,
        "hallucination_phrase_count": sum(bool(row["hallucination_phrase"]) for row in flat),
        "bertscore_f1_mean": bertscore_mean,
        "bertscore_status": bertscore_status,
    }
    reports_dir = config_path.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(config["seed"]))
    examples = rng.sample(per_news, min(args.example_count, len(per_news)))
    lines = ["# Generator evaluation report", "", "## Metrics", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "Semantic similarity is an auxiliary diagnostic, not a complete measure of comment quality.",
            "",
            "## Qualitative examples",
            "",
        ]
    )
    for index, record in enumerate(examples, 1):
        lines.extend(
            [
                f"### {index}. {record['news_id']}",
                "",
                f"- News: {record['news_text']}",
                f"- Image: `{record['image_path']}`",
                f"- Observed: `{json.dumps(record['observed_comments'], ensure_ascii=False)}`",
                f"- Real targets: `{json.dumps(record['real_target_comments'], ensure_ascii=False)}`",
                f"- Generated: `{json.dumps(record['generated_comments'], ensure_ascii=False)}`",
                "",
            ]
        )
    report_path = reports_dir / "generator_eval_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics_path = reports_dir / "generator_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
