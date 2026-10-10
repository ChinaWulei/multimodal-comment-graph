"""Audit metric safety: ambiguous acceptance is not a true positive; no threshold refit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.evaluate_reserved_quality_candidate import summarize


def main():
    criteria = {"auc_strictly_above": .6, "precision_at_least": .8,
                "recall_at_least": .25, "accepted_at_least": 5}
    # Perfect ranking but an overpermissive fixed threshold: must fail precision.
    examples = [{"content_target": int(i < 5), "content_probability": 1.-i*.05,
                 "content_accepted": True} for i in range(11)]
    examples += [{"content_target": None, "content_probability": .99, "content_accepted": True},
                 {"content_target": None, "content_probability": .01, "content_accepted": False}]
    result = summarize(examples, "content", criteria)
    assert result["roc_auc"] == 1. and not result["passed"]
    assert result["precision"] == 5/11 and result["accepted_ambiguous"] == 1
    assert result["accepted_total"] == 12 and result["accepted_known"] == 11
    # A tiny flawless accepted set is insufficient evidence, not success.
    tiny = [{"content_target": 1, "content_probability": .9, "content_accepted": True},
            {"content_target": 0, "content_probability": .1, "content_accepted": False}]
    result = summarize(tiny, "content", criteria)
    assert result["precision"] == 1. and result["coverage_inconclusive"] and not result["passed"]
    result = summarize([examples[-1]], "content", criteria)
    assert result["roc_auc"] is None and result["precision"] is None and not result["passed"]
    print("Reserved-quality metrics smoke passed: no refit, no promotion of ambiguous cases, no tiny-set success claim.")


if __name__ == "__main__":
    main()
