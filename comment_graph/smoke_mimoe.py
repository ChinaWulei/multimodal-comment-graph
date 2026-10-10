"""Validate adapted MIMoE routing math and one real Weibo MFCL backward pass."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform, seed_everything
from comment_graph.mimoe_fusion import MIMoEFusion
from comment_graph.model import CommentGraphMFCL


def unit_check() -> dict:
    torch.manual_seed(20261005)
    fusion = MIMoEFusion(input_dim=16, semantic_dim=8, hidden_dim=24, num_experts=2)
    text, image = torch.randn(5, 16), torch.randn(5, 16)
    output = fusion(text, image)
    assert output["h_mm"].shape == (5, 32)
    assert output["router_weights"].shape == (5, 4)
    assert torch.allclose(output["router_weights"].sum(-1), torch.ones(5), atol=1e-6)
    assert ((output["router_targets"] >= 0) & (output["router_targets"] <= 3)).all()
    labels = torch.tensor([0, 1, 0, 1, 0])
    # The released MIMoE objective includes the two unimodal CE terms.  They
    # are essential: the router pseudo targets intentionally do not backprop
    # through the threshold operation into these heads.
    loss = (output["h_mm"].square().mean() + output["routing_loss"] + .5 * (
        F.cross_entropy(output["text_unimodal_logits"], labels) +
        F.cross_entropy(output["image_unimodal_logits"], labels)
    ))
    loss.backward()
    grads = [p.grad for p in fusion.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    return {
        "h_mm_shape": list(output["h_mm"].shape),
        "router_probability_rows_sum_to_one": True,
        "pseudo_targets_in_0_to_3": True,
        "all_mimoe_parameter_gradients_finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing smoke report")
    report = {"unit": unit_check(), "test_used": False}
    if not torch.cuda.is_available():
        raise SystemExit("Unit check passed; a real Weibo smoke test needs CUDA")

    config_path = Path(__file__).with_name("config_weibo.yaml")
    config, _ = load_config(config_path)
    config.mm_fusion["mode"] = "mimoe"
    seed_everything(42)
    tokenizer, transform = make_tokenizer_and_transform(config)
    dataset = make_dataset(config, config_path, "train", "baseline", tokenizer, transform)
    labels = dataset.labels()
    chosen = []
    for label in (0, 1):
        chosen.extend([i for i, value in enumerate(labels) if value == label][:4])
    batch = next(iter(DataLoader(torch.utils.data.Subset(dataset, chosen), batch_size=8))).to("cuda")
    model = CommentGraphMFCL(config, "baseline", "star", "mean", "concat").cuda().train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast("cuda"):
        output = model(batch)
        labels = batch.y.view(-1)
        main_ce = F.cross_entropy(output["logits"], labels)
        fused = output["fusion_output"]
        unimodal = .5 * (F.cross_entropy(fused["text_unimodal_logits"], labels) +
                          F.cross_entropy(fused["image_unimodal_logits"], labels))
        loss = main_ce + unimodal + .7 * fused["routing_loss"]
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    named = dict(model.named_parameters())
    required = [name for name in named if name.startswith("mimoe_fusion.")]
    assert required and all(named[name].grad is not None and torch.isfinite(named[name].grad).all() for name in required)
    router_grad = float(sum(named[name].grad.abs().sum() for name in required if ".router." in name))
    expert_grad = float(sum(named[name].grad.abs().sum() for name in required if ".conditions." in name))
    assert router_grad > 0 and expert_grad > 0
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    report["weibo_smoke"] = {
        "news_count": 8,
        "news_ids": list(batch.news_id),
        "h_mm_dim": int(output["h_mm"].shape[-1]),
        "h_graph_dim": int(output["h_graph"].shape[-1]),
        "h_final_dim": int(output["h_final"].shape[-1]),
        "loss": float(loss.detach()),
        "main_ce": float(main_ce.detach()),
        "routing_loss": float(fused["routing_loss"].detach()),
        "router_gradient_l1": router_grad,
        "expert_gradient_l1": expert_grad,
        "router_targets": fused["router_targets"].detach().cpu().tolist(),
        "router_weights": fused["router_weights"].detach().float().cpu().tolist(),
        "all_mimoe_gradients_non_null_finite": True,
        "optimizer_step_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    del model, batch, optimizer, scaler, output
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
