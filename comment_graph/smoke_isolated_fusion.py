"""Eight actual TRAIN news: gradients, source isolation, fallback and checkpoint tests."""
import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader

from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform, seed_everything
from comment_graph.isolated_fusion import classification_objective, select_nodes
from comment_graph.model import CommentGraphMFCL
from comment_graph.train import evaluate, class_weights


def check_edges(edge_history, indices, batch, forbidden_type):
    counts = []
    for edges, weights in edge_history:
        endpoints = indices[edges]
        assert (batch.node_type[endpoints] != forbidden_type).all()
        assert (batch.batch[endpoints[0]] == batch.batch[endpoints[1]]).all()
        counts.append(int(edges.size(1)))
        if weights is not None:
            assert torch.isfinite(weights).all()
    return counts


def assert_close(a, b):
    assert torch.allclose(a, b, atol=2e-5, rtol=2e-5), float((a - b).abs().max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    config, path = load_config(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: real multimodal smoke test not started on CPU")
    device = torch.device("cuda")
    seed_everything(int(config.seed))
    tokenizer, transform = make_tokenizer_and_transform(config)
    full = make_dataset(config, path, "train", "real_generated", tokenizer, transform)
    labels = full.labels()
    chosen = [i for label in (0, 1) for i in [j for j, y in enumerate(labels) if y == label][:4]]
    assert len(chosen) == 8
    loader = DataLoader(torch.utils.data.Subset(full, chosen), batch_size=8, shuffle=False)
    batch = next(iter(loader)).to(device)
    model = CommentGraphMFCL(config, "real_generated", "dynamic_cross_modal", "mean", "source_isolated").to(device)
    generated = model.generated_branch
    assert {id(p) for p in model.graph_model.encoder.parameters()}.isdisjoint(
        {id(p) for p in generated.encoder.parameters()})
    weights = class_weights(full, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = GradScaler("cuda", init_scale=float(config.isolated_fusion.amp_init_scale))
    original_dropout = generated.branch_dropout
    generated.branch_dropout = 0.0  # Deterministic gradient coverage in this bounded test.
    initial = generated.correction.weight.detach().clone()
    losses, gradients = [], {}
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            output = model(batch)
            loss, final_ce, base_ce = classification_objective(
                output, batch.y.view(-1), weights, float(config.isolated_fusion.base_loss_weight))
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, module in [("base_gnn", model.graph_model.encoder),
                             ("base_edges", model.dynamic_encoder.edge_builders),
                             ("base_classifier", model.classifier),
                             ("generated_gnn", generated.encoder),
                             ("generated_edges", generated.dynamic_encoder.edge_builders),
                             ("generated_gate", generated.gate),
                             ("generated_correction", generated.correction)]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), (name, len(grads), scaler.get_scale())
            norm = sum(float(g.abs().sum()) for g in grads)
            assert norm > 0, name
            gradients[name] = norm
        scaler.step(optimizer)
        scaler.update()
        losses.append({"total": float(loss.detach()), "final_ce": float(final_ce.detach()),
                       "base_ce": float(base_ce.detach())})
    assert not torch.equal(initial, generated.correction.weight)

    # Configurable full-branch dropout: training logits fall back EXACTLY to base.
    generated.branch_dropout = 1.0
    with torch.no_grad(), autocast("cuda"):
        dropped = model(batch)
    assert torch.equal(dropped["logits"], dropped["global_logits"].float())
    generated.branch_dropout = original_dropout
    model.eval()
    with torch.no_grad():
        result = model(batch)
        base_edges = check_edges(result["base_edge_history"], result["base_node_indices"], batch, 2)
        gen_edges = check_edges(result["local_correction"]["edge_history"],
                               result["generated_node_indices"], batch, 1)
        modified = batch.clone()
        modified.node_input_ids[modified.node_type == 2] = tokenizer.unk_token_id
        changed = model(modified)
        assert_close(result["global_logits"], changed["global_logits"])
        augmentation_change = float((result["logits"] - changed["logits"]).abs().max())
        assert augmentation_change > 1e-7
        real_view, _ = select_nodes(batch, batch.node_type != 2)
        removed = model(real_view)
        assert_close(result["global_logits"], removed["global_logits"])
        assert torch.equal(removed["logits"], removed["global_logits"].float())
        disabled = model(batch, disable_generated=True)
        assert_close(result["global_logits"], disabled["logits"])
        mixed_view, _ = select_nodes(batch, ~((batch.node_type == 2) & (batch.batch == 0)))
        mixed = model(mixed_view)
        assert_close(result["global_logits"], mixed["global_logits"])
        assert torch.count_nonzero(mixed["local_correction"]["logit_delta"][0]) == 0
        # Duplicating generated nodes must not alter base pooling or normalization.
        items = batch.to_data_list()
        item = items[0]
        duplicate = int((item.node_type == 2).nonzero()[0])
        n = item.num_nodes
        item.node_input_ids = torch.cat((item.node_input_ids, item.node_input_ids[duplicate:duplicate + 1]))
        item.node_attention_mask = torch.cat((item.node_attention_mask, item.node_attention_mask[duplicate:duplicate + 1]))
        item.node_type = torch.cat((item.node_type, item.node_type[duplicate:duplicate + 1]))
        item.edge_index = torch.cat((item.edge_index, item.edge_index.new_tensor([[0, n], [n, 0]])), dim=1)
        item.num_nodes = n + 1
        item.num_generated_comments += 1
        duplicated = next(iter(DataLoader(items, batch_size=8))).to(device)
        duplicate_result = model(duplicated)
        assert_close(result["global_logits"], duplicate_result["global_logits"])
        altered_labels = batch.clone()
        altered_labels.y = 1 - altered_labels.y
        assert_close(result["logits"], model(altered_labels)["logits"])
        assert_close(result["logits"], result["global_logits"].float() + result["local_correction"]["logit_delta"])
        assert result["local_correction"]["logit_delta"].abs().max() <= generated.max_logit_delta
        # Test serialization including new independent GCN and gate parameters.
        buffer = io.BytesIO()
        torch.save(generated.state_dict(), buffer)
        buffer.seek(0)
        generated.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        assert_close(result["logits"], model(batch)["logits"])
        # The same implementation also supports original star edges.
        model.graph_structure = "star"
        star = model(batch)
        assert torch.isfinite(star["logits"]).all()
        check_edges(star["base_edge_history"], star["base_node_indices"], batch, 2)
        check_edges(star["local_correction"]["edge_history"], star["generated_node_indices"], batch, 1)
        model.graph_structure = "dynamic_cross_modal"
        for mode, keep in [("baseline", batch.node_type == 0),
                           ("real_only", batch.node_type != 2),
                           ("generated_only", batch.node_type != 1)]:
            model.mode = mode
            view, _ = select_nodes(batch, keep)
            out = model(view)
            assert torch.isfinite(out["logits"]).all()
            if mode != "generated_only":
                assert torch.equal(out["logits"], out["global_logits"].float())
        model.mode = "real_generated"
    evaluation = evaluate(model, loader, device, True, weights)
    report = {
        "status": "passed; technical smoke test, NOT performance evidence",
        "split": "train", "news_ids": list(batch.news_id), "optimizer_steps": 2,
        "losses": losses, "gradient_l1": gradients,
        "base_nodes": result["base_node_indices"].numel(),
        "generated_view_nodes": result["generated_node_indices"].numel(),
        "base_edges_per_layer": base_edges, "generated_edges_per_layer": gen_edges,
        "base_invariance_max_delta": float((result["global_logits"] - changed["global_logits"]).abs().max()),
        "generated_content_final_logits_delta": augmentation_change,
        "checks": ["no cross-source edges", "no cross-news edges", "distinct GNN parameters",
                   "base invariant to generated replacement/removal/duplication", "mixed missing comments",
                   "disable branch fallback", "branch dropout=1 fallback", "label independence",
                   "star and dynamic graphs", "all four data modes", "branch checkpoint roundtrip"],
        "h_mm_dim": result["h_mm"].size(1), "base_graph_dim": result["h_graph"].size(1),
        "generated_graph_dim": result["local_correction"]["h_generated_graph"].size(1),
        "additional_parameters": sum(p.numel() for p in generated.parameters()),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "evaluation_after_two_steps": evaluation,
    }
    destination = path.parent / "reports" / "smoke_isolated_fusion.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "evaluation_after_two_steps"}, indent=2))
    print(f"report={destination}")


if __name__ == "__main__":
    main()
