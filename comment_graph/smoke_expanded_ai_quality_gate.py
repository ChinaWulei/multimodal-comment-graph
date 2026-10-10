"""Check the actual expansion boundary; never inspect the reserved audit rows."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.train_ai_quality_gate import load_expansion
from comment_graph.train_generated_quality_gate import load_data
from comment_graph.validate_generated_quality_annotations import load_unique


def main():
    reports = ROOT / "comment_graph" / "reports"
    expansion = reports / "quality_expansion_20261008"
    previous = list(load_unique(reports / "generated_quality_annotation_train60.jsonl").values())
    features = reports / "generated_quality_parent_merged_train.jsonl"
    original_read = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path.name == "reserved_audit.jsonl":
            raise AssertionError("Fitting must never read reserved audit contents")
        return original_read(path, *args, **kwargs)

    with patch.object(Path, "read_text", guarded_read):
        reference, annotations, provenance = load_expansion(expansion, previous, features)
    assert len(reference) == len(annotations) == 80
    assert provenance["reserved_audit_content_read"] is False
    assert all(row["annotation_source"] == "ai" and row["human_reviewed"] is False
               for row in annotations)
    output = ROOT / "comment_graph" / "outputs" / "ai_quality_gate_train140"
    combined, _, _ = load_data(output / "reference_fit.jsonl", output / "annotations_ai.jsonl", features)
    assert len(combined) == 140 and len({r["news_id"] for r in combined}) == 140
    report = json.loads((output / "fit_report.json").read_text(encoding="utf-8"))
    assert report["human_annotations"] == 0
    if not all(report[name]["passed"] for name in ("content", "parent")):
        assert not report["gate_saved"] and not (output / "gate.joblib").exists()
    negative_checks = 0
    with tempfile.TemporaryDirectory(prefix="quality_expansion_smoke_") as directory:
        target = Path(directory)
        for name in ("manifest.json", "fit.jsonl", "fit_ai_draft.jsonl"):
            shutil.copyfile(expansion / name, target / name)
        for field, value in (("news_id", "different_news"), ("human_reviewed", True),
                             ("annotation_source", "human")):
            drafts = list(load_unique(expansion / "fit_ai_draft.jsonl").values())
            drafts[0][field] = value
            (target / "fit_ai_draft.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in drafts), encoding="utf-8")
            try:
                load_expansion(target, previous, features)
            except ValueError:
                negative_checks += 1
            else:
                raise AssertionError(f"Invalid provenance accepted: {field}")
        shutil.copyfile(expansion / "fit_ai_draft.jsonl", target / "fit_ai_draft.jsonl")
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        manifest["reserved_audit_news_ids"].append(reference[0]["news_id"])
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        try:
            load_expansion(target, previous, features)
        except ValueError:
            negative_checks += 1
        else:
            raise AssertionError("Audit overlap accepted")
    assert negative_checks == 4
    print(json.dumps({"passed": True, "fitting_news": len(combined),
                      "invalid_input_rejections": negative_checks,
                      "audit_contents_read": False, "human_annotations": 0}))


if __name__ == "__main__":
    main()
