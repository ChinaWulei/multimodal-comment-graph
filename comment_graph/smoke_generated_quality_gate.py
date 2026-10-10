"""Synthetic unit smoke for the fit path; never a research quality result."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.train_generated_quality_gate import (  # noqa: E402
    CONTENT_FEATURES, PARENT_FEATURES, fit_head,
)


def main() -> None:
    annotations = []
    features = []
    for i in range(40):
        positive = i % 2 == 0
        annotations.append({
            "content_relevance": "relevant" if positive else "off_topic",
            "parent_fit": "good" if positive else "wrong",
            "unsupported_specific_claim": False,
        })
        # Intentionally linearly separable test vectors, not real comment data.
        value = 2.0 if positive else -2.0
        features.append({name: value + 0.001 * i
                         for name in set(CONTENT_FEATURES) | set(PARENT_FEATURES)})
    for name, fields in (("content", CONTENT_FEATURES), ("parent", PARENT_FEATURES)):
        model, report = fit_head(annotations, features, name, fields, seed=42)
        assert report["oof_roc_auc"] > 0.9
        assert report["oof_precision"] >= 0.8
        assert report["oof_recall"] >= 0.25
        assert len(model.predict_proba([[features[0][field] for field in fields]])) == 1
        _, ai_report = fit_head(annotations, features, name, fields, seed=42, label_source="ai")
        assert ai_report["annotation_source"] == "ai"
        assert ai_report["ai_labeled"] == 40 and "human_labeled" not in ai_report
    diagnostics = {}
    flat = [{name: 0.0 for name in set(CONTENT_FEATURES) | set(PARENT_FEATURES)} for _ in features]
    try:
        fit_head(annotations, flat, "content", CONTENT_FEATURES, seed=42,
                 label_source="ai", diagnostics=diagnostics)
    except ValueError as error:
        assert "CV discrimination" in str(error)
        assert not diagnostics["passed"] and len(diagnostics["oof_predictions"]) == 40
    else:
        raise AssertionError("An uninformative AI-label gate must fail the shared criteria")
    built = []
    def factory():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.2, class_weight="balanced", max_iter=1000))
        built.append(model)
        return model
    fit_head(annotations, features, "content", CONTENT_FEATURES, seed=42,
             label_source="ai", model_factory=factory)
    assert len(built) == 6 and len({id(model) for model in built}) == 6
    assert [int(model[0].n_samples_seen_) for model in built] == [32] * 5 + [40]
    print("Synthetic gate fit-path smoke passed; no real annotations were used.")


if __name__ == "__main__":
    main()
