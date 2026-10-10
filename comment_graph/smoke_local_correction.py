"""Real-data optimizer smoke test and local-evidence invariants; no test split."""
import argparse
import copy
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader

from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform, seed_everything
from comment_graph.model import CommentGraphMFCL
from comment_graph.local_correction import CommentLocalCorrection
from comment_graph.train import evaluate, class_weights


def unit_checks(num_classes=2):
    torch.manual_seed(42)
    head = CommentLocalCorrection(global_dim=24, dim=16, dropout=0, num_classes=num_classes).eval()
    global_features = torch.randn(2, 24)
    nodes = torch.randn(5, 768)
    types = torch.tensor([0, 1, 2, 0, 2])
    batch = torch.tensor([0, 0, 0, 1, 1])
    tokens = torch.randn(2, 6, 768)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    images = torch.randn(2, 2048, 3, 3)
    def run(n=nodes, t=types, b=batch, tok=tokens, img=images):
        return head(global_features, n, t, b, tok, mask, img)
    output = run()
    assert output["logit_delta"].shape == (2, num_classes)
    weights = output["text_weights"]
    valid = mask[batch[output["comment_indices"]]].bool()
    assert torch.equal(weights[~valid], torch.zeros_like(weights[~valid]))
    assert torch.allclose(weights.sum(-1), torch.ones(3))
    changed_tokens = tokens.clone()
    changed_tokens[~mask.bool()] = 1000
    assert torch.allclose(output["logit_delta"], run(tok=changed_tokens)["logit_delta"], atol=1e-6)
    image_change = (output["logit_delta"] - run(img=torch.zeros_like(images))["logit_delta"]).abs().max()
    text_change = (output["logit_delta"] - run(tok=torch.zeros_like(tokens))["logit_delta"]).abs().max()
    assert image_change > 1e-7 and text_change > 1e-7
    permutation = torch.tensor([2, 0, 4, 1, 3])
    assert torch.allclose(output["logit_delta"], run(nodes[permutation], types[permutation], batch[permutation])["logit_delta"], atol=1e-6)
    # Root-only graph must have an exactly zero correction.
    no_comments = run(nodes[[0, 3]], torch.zeros(2, dtype=torch.long), torch.tensor([0, 1]))
    assert torch.count_nonzero(no_comments["logit_delta"]) == 0
    # Even within a mixed batch, a root-only graph must remain unchanged.
    mixed = run(nodes[:4], types[:4], batch[:4])
    assert torch.count_nonzero(mixed["logit_delta"][1]) == 0
    e = torch.randn(3, 16)
    ids = torch.tensor([0, 0, 1])
    source = torch.ones(3, dtype=torch.bool)
    high, _ = head.source_pool(e, torch.ones(3), source, ids, 2)
    low, _ = head.source_pool(e, torch.full((3,), .01), source, ids, 2)
    assert torch.allclose(low, high * .01, atol=1e-6)
    # Local checkpoint round trip.
    buffer = io.BytesIO()
    torch.save(head.state_dict(), buffer)
    buffer.seek(0)
    clone = copy.deepcopy(head)
    clone.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(output["logit_delta"], clone(global_features, nodes, types, batch, tokens, mask, images)["logit_delta"])
    return {"passed": True, "image_perturbation_delta": float(image_change.detach()),
            "text_perturbation_delta": float(text_change.detach())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--graph-structure", default="dynamic_cross_modal")
    parser.add_argument("--message-passing", choices=["legacy_gcn", "typed_rgcn"], default="legacy_gcn")
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--max-real-comments", type=int)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    num_classes = int(config.get("classification", {}).get("num_classes", 2))
    if args.samples_per_class < 1:
        raise ValueError("samples-per-class must be positive")
    invariants = unit_checks(num_classes)
    config.graph.structure_mode = args.graph_structure
    config.graph.message_passing = args.message_passing
    if args.max_real_comments is not None:
        if args.max_real_comments < 1:
            raise ValueError("max-real-comments must be positive")
        config.data.max_real_comments = args.max_real_comments
    seed_everything(int(config.seed))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(4)
    tokenizer, transform = make_tokenizer_and_transform(config)
    full = make_dataset(config, config_path, "train", "real_generated", tokenizer, transform)
    selected = []
    for label in range(num_classes):
        candidates = [i for i, y in enumerate(full.labels()) if y == label]
        if len(candidates) < args.samples_per_class:
            raise ValueError(f"Not enough training samples for class {label}")
        selected.extend(candidates[:args.samples_per_class])
    dataset = torch.utils.data.Subset(full, selected)
    loader = DataLoader(dataset, batch_size=len(selected), shuffle=False)
    batch = next(iter(loader)).to(device)
    assert batch.y.numel() == num_classes * args.samples_per_class
    model = CommentGraphMFCL(config, "real_generated", args.graph_structure, "mean",
                             decision_mode="global_local_correction").to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=amp)
    model.train()
    initial = model.local_correction.correction[-1].weight.detach().clone()
    losses = []
    gradient_report = {}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp):
            output = model(batch)
            loss = F.cross_entropy(output["logits"], batch.y.view(-1))
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        graph_module = model.typed_graph_encoder if args.message_passing == "typed_rgcn" else model.graph_model.encoder
        modules = [("gnn", graph_module)]
        if args.graph_structure == "dynamic_cross_modal":
            modules.append(("edges", model.dynamic_encoder.edge_builders))
        modules.extend([(name, getattr(model.local_correction, name)) for name in
                               ("text_query", "image_query", "text_projection", "image_projection",
                                "generated_reliability", "gate", "correction")])
        for name, module in modules:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
            norm = sum(float(g.abs().sum()) for g in grads)
            assert norm > 0, name
            gradient_report[name] = norm
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    assert not torch.equal(initial, model.local_correction.correction[-1].weight)
    model.eval()
    # FP32 for input-invariance checks; AMP + CUDA scatter introduces rounding noise.
    with torch.no_grad():
        first = model(batch)
        labels = batch.y.clone()
        batch.y = (batch.y + 1) % num_classes
        second = model(batch)
        batch.y = labels
    label_delta = float((first["logits"] - second["logits"]).abs().max())
    assert torch.allclose(first["logits"], second["logits"], atol=1e-5, rtol=1e-5), f"Label invariance delta={label_delta}"
    assert torch.allclose(first["logits"], first["global_logits"].float() + first["local_correction"]["logit_delta"])
    # Baseline bypasses all comments and has no local correction.
    baseline = make_dataset(config, config_path, "train", "baseline", tokenizer, transform, max_samples=8)
    model.mode = "baseline"
    with torch.no_grad():
        result = model(next(iter(DataLoader(baseline, batch_size=8))).to(device))
    assert torch.count_nonzero(result["local_correction"]["logit_delta"]) == 0
    model.mode = "real_generated"
    evaluation = evaluate(model, loader, device, amp, class_weights(full, device))
    report = {"device": str(device), "split": "train", "news_ids": list(batch.news_id),
              "num_classes": num_classes, "graph_structure": args.graph_structure,
              "message_passing": args.message_passing,
              "config_path": str(config_path), "max_real_comments": int(config.data.max_real_comments),
              "generated_source": str(config.data.generated_comments.train),
              "labels": batch.y.tolist(), "unit_checks": invariants, "optimizer_steps": 2,
              "losses": losses, "gradient_l1": gradient_report,
              "local_parameters": sum(p.numel() for p in model.local_correction.parameters()),
              "mean_gate": float(first["local_correction"]["correction_gate"].mean()),
              "label_invariance_max_delta_fp32": label_delta,
              "evaluation_after_two_steps": evaluation,
              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if amp else None,
              "status": "passed; technical smoke test only, not a quality result"}
    path = args.output_json or config_path.parent / "reports" / "smoke_local_correction.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "evaluation_after_two_steps"}, indent=2))
    print(f"report={path}")


if __name__ == "__main__":
    main()
