"""Check scored real-case coverage and label exclusion from the VLM input."""
from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.score_generated_quality_vlm import QUESTIONS, digest, input_digest, source_data


def main() -> None:
    reports = ROOT / "comment_graph/reports"
    queue_path = reports / "generated_quality_annotation_train60.jsonl"
    score_path = reports / "generated_quality_train60_vlm_features.jsonl"
    meta = json.loads((reports / "generated_quality_train60_vlm_features_meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    scores = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines()]
    score_by_id = {row["annotation_id"]: row for row in scores}
    assert len(rows) == len(scores) == len(score_by_id) == 60
    assert meta["complete"] and meta["protocol"]["queue_sha256"] == digest(queue_path)
    for row in rows:
        changed_labels = deepcopy(row)
        changed_labels.update({"label": 2, "3_way_label": 1, "content_relevance": "off_topic",
                               "parent_fit": "wrong", "unsupported_specific_claim": True,
                               "review_notes": "This quality judgment must never be provided to the scorer."})
        assert source_data(row) == source_data(changed_labels)
        assert input_digest(row) == input_digest(changed_labels)
        changed_text = deepcopy(row)
        changed_text["generated_comment"] += " [changed]"
        assert input_digest(row) != input_digest(changed_text)
        score = score_by_id[row["annotation_id"]]
        assert score["input_sha256"] == input_digest(row)
        assert score["image_sha256"] == digest(Path(row["image_path"]))
        assert score["news_id"] == row["news_id"] and score["official_split"] == "train"
        for task in QUESTIONS:
            assert math.isfinite(score[f"vlm_{task}_log_odds"])
            assert 0 <= score[f"vlm_{task}_yes_probability"] <= 1
            assert 0 <= score[f"vlm_{task}_answer_mass"] <= 1.00001
    print("60 real-case/image hashes verified; truth/quality-label changes do not alter VLM input.")


if __name__ == "__main__":
    main()
