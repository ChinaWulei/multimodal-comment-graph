"""Exercise label-free entity-evidence data mapping, fusion and gradients on Weibo."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform, seed_everything
from comment_graph.model import CommentGraphMFCL


def run_arm(config, config_path, fusion: str, evidence: str):
    config.mm_fusion["mode"] = fusion
    config.entity_evidence["mode"] = evidence
    seed_everything(42)
    tokenizer, transform = make_tokenizer_and_transform(config)
    dataset = make_dataset(config, config_path, "train", "baseline", tokenizer, transform)
    labels = dataset.labels()
    chosen = []
    for label in (0, 1):
        chosen.extend([i for i, value in enumerate(labels) if value == label][:4])
    batch = next(iter(DataLoader(torch.utils.data.Subset(dataset, chosen), batch_size=8))).to("cuda")
    assert not hasattr(batch, "entity_evidence_labels")
    assert batch.entity_evidence_scores.shape == (8, 5)
    model = CommentGraphMFCL(config, "baseline", "star", "mean", "concat").cuda().train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    with torch.autocast("cuda"):
        output = model(batch)
        loss = F.cross_entropy(output["logits"], batch.y.view(-1))
        if fusion == "mimoe":
            mimoe = output["fusion_output"]
            loss = loss + .5 * (F.cross_entropy(mimoe["text_unimodal_logits"], batch.y.view(-1)) +
                                F.cross_entropy(mimoe["image_unimodal_logits"], batch.y.view(-1))) + .7 * mimoe["routing_loss"]
    loss.backward()
    evidence_params = [p for name, p in model.named_parameters() if name.startswith("entity_evidence_fusion.")]
    active_params = [p for name, p in model.named_parameters()
                     if name.startswith("entity_evidence_fusion.") and not (evidence == "direct" and ".gate." in name)]
    assert active_params and all(p.grad is not None and torch.isfinite(p.grad).all() for p in active_params)
    gate = output["evidence_output"]["evidence_gate"]
    assert torch.isfinite(gate).all() and (gate >= 0).all() and (gate <= 1).all()
    optimizer.step()
    result = {
        "fusion": fusion, "evidence_mode": evidence, "h_mm_dim": int(output["h_mm"].shape[-1]),
        "h_final_dim": int(output["h_final"].shape[-1]), "loss": float(loss.detach()),
        "evidence_covered_news": int(output["evidence_output"]["evidence_available"].sum()),
        "mean_evidence_gate": float(gate.detach().mean()),
        "evidence_gradient_l1": float(sum(p.grad.abs().sum() for p in active_params)),
        "unused_gate_parameters_in_direct_ablation": bool(evidence == "direct"),
        "retrieved_labels_exposed": False,
    }
    del model, optimizer, batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite smoke report")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real encoder smoke test")
    config_path = Path(__file__).with_name("config_weibo.yaml")
    report = {"test_used": False, "arms": []}
    for fusion, evidence in (("mfcl", "gated"), ("mimoe", "gated"), ("mimoe", "direct")):
        config, _ = load_config(config_path)
        report["arms"].append(run_arm(config, config_path, fusion, evidence))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
