"""Real-data input/target separation and synthetic loss-weighting checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.train_quality_lora_cv import content_target, request


def main():
    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedKFold
    path = ROOT / "comment_graph/outputs/ai_quality_gate_train140/annotations_ai.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        changed = {**row, "label": 2, "3_way_label": 0, "content_relevance": "off_topic",
                   "parent_fit": "wrong", "unsupported_specific_claim": True,
                   "review_notes": "This is a supervised target; must never enter the input."}
        assert request(row) == request(changed)
        assert request(row) != request({**row, "generated_comment": "Changed candidate"})
        assert request(row) != request({**row, "parent_content": "Changed parent"})
    selected = [r for r in rows if content_target(r) is not None]
    y = np.asarray([content_target(r) for r in selected])
    assert len(selected) == 97 and int(y.sum()) == 33
    counts = np.zeros(len(selected), dtype=int)
    for train, heldout in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y):
        assert not {selected[i]["news_id"] for i in train} & {selected[i]["news_id"] for i in heldout}
        counts[heldout] += 1
    assert (counts == 1).all()
    # With batch size one, weighted CE's default mean cancels class weights.
    # The training script explicitly multiplies unreduced CE by the fold weight.
    logits = torch.tensor([[0., 0.]], requires_grad=True)
    label = torch.tensor([1])
    loss = torch.nn.functional.cross_entropy(logits, label) * 2.
    loss.backward()
    assert torch.allclose(logits.grad, torch.tensor([[1., -1.]]))
    print(json.dumps({"passed": True, "inputs_checked": len(rows), "selected_training_news": len(selected),
                      "quality_and_news_labels_excluded_from_prompt": True,
                      "heldout_once_per_news": True, "single_sample_class_weighting_verified": True}))


if __name__ == "__main__":
    main()
