"""Real-data smoke test plus label-blind swap and gradient isolation checks."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform, seed_everything
from comment_graph.model import CommentGraphMFCL
from comment_graph.isolated_fusion import classification_objective
from comment_graph.robustness import mismatch_generated_features, mismatch_forward, add_mismatch_loss
from comment_graph.train import evaluate, class_weights


def rng():
    return torch.Generator().manual_seed(123)


def main():
    # Unequal comment counts, an empty generated set, and repeated IDs.
    view = SimpleNamespace(root_input_ids=torch.zeros(3, 1),
                           batch=torch.tensor([0, 0, 0, 1, 1, 2]),
                           node_type=torch.tensor([0, 2, 2, 0, 2, 0]), news_id=['a', 'b', 'c'])
    features = torch.arange(18).reshape(6, 3).float()
    swapped, valid, donors = mismatch_generated_features(features, view, rng())
    assert donors == [1, 0, -1] and valid.tolist() == [True, True, False]
    assert torch.equal(features[view.node_type == 0], swapped[view.node_type == 0])
    assert torch.equal(swapped[1], features[4]) and torch.equal(swapped[2], features[4])
    view.y = torch.tensor([1, 0, 1])
    assert torch.equal(swapped, mismatch_generated_features(features, view, rng())[0])
    view.news_id = ['a', 'a', 'c']
    assert not mismatch_generated_features(features, view, rng())[1].any()
    view.node_type.zero_()
    assert not mismatch_generated_features(features, view, rng())[1].any()

    if not torch.cuda.is_available():
        raise SystemExit('CPU swap checks passed; real multimodal GPU test not run: CUDA unavailable')
    config, path = load_config(Path(__file__).with_name('config.yaml'))
    seed_everything(int(config.seed))
    device = torch.device('cuda')
    tokenizer, transform = make_tokenizer_and_transform(config)
    full = make_dataset(config, path, 'train', 'real_generated', tokenizer, transform)
    labels = full.labels()
    chosen = [i for label in (0, 1) for i in [j for j, y in enumerate(labels) if y == label][:4]]
    loader = DataLoader(torch.utils.data.Subset(full, chosen), batch_size=8)
    batch = next(iter(loader)).to(device)
    model = CommentGraphMFCL(config, 'real_generated', 'dynamic_cross_modal', 'mean', 'source_isolated').to(device)
    weights = class_weights(full, device)
    model.eval()
    with torch.no_grad(), autocast('cuda'):
        plain = model(batch)
        output = model(batch, return_generated_context=True)
        # GPU scatter reductions can differ slightly across identical calls.
        context_delta = float((plain['logits'] - output['logits']).abs().max())
        assert torch.allclose(plain['logits'], output['logits'], atol=2e-5, rtol=2e-5), context_delta
        ce, _, _ = classification_objective(output, batch.y.view(-1), weights, .5)
        same, zero, count = add_mismatch_loss(model, output, ce, rng(), 0)
        assert same is ce and zero == 0 and count == 0
        before_rng = torch.get_rng_state().clone()
        before_cuda_rng = torch.cuda.get_rng_state().clone()
        mismatch_forward(model, output, rng())
        assert torch.equal(before_rng, torch.get_rng_state())
        assert torch.equal(before_cuda_rng, torch.cuda.get_rng_state())

    model.train()
    output = model(batch, return_generated_context=True)
    bn = {name: m.num_batches_tracked.clone() for name, m in model.named_modules()
          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
    noisy = mismatch_forward(model, output, rng())
    assert noisy is not None and noisy['valid'].sum() == 8
    assert torch.isfinite(noisy['loss'])
    noisy['loss'].backward()
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith('generated_branch.'))
    grad_report = {}
    for name, module in [('generated_gnn', model.generated_branch.encoder),
                         ('generated_edges', model.generated_branch.dynamic_encoder.edge_builders),
                         ('gate', model.generated_branch.gate)]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        grad_report[name] = sum(float(g.abs().sum()) for g in grads)
        assert grad_report[name] > 0
    for name, m in model.named_modules():
        if name in bn:
            assert torch.equal(m.num_batches_tracked, bn[name])
    auxiliary_kl = float(noisy['loss'].detach())
    del output, noisy, plain
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = GradScaler('cuda', init_scale=1024)
    optimizer.zero_grad(set_to_none=True)
    initial = model.generated_branch.correction.weight.detach().clone()
    with autocast('cuda'):
        output = model(batch, return_generated_context=True)
        clean, final_ce, base_ce = classification_objective(output, batch.y.view(-1), weights, .5)
        loss, kl, count = add_mismatch_loss(model, output, clean, rng(), .5)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None for p in model.graph_model.encoder.parameters())
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(initial, model.generated_branch.correction.weight)
    result = evaluate(model, loader, device, True, weights, audit_mismatch=True)
    assert result['mismatch_audit']['evaluated_news'] == 8
    report = {'status': 'passed; technical test only', 'split': 'train', 'news_ids': list(batch.news_id),
              'optimizer_steps': 1, 'auxiliary_kl': auxiliary_kl, 'auxiliary_only_gradient_l1': grad_report,
              'context_logits_max_delta': context_delta,
              'total_loss': float(loss.detach()), 'final_ce': float(final_ce.detach()),
              'base_ce': float(base_ce.detach()), 'mismatch_kl': float(kl.detach()),
              'checks': ['different-news donors', 'unequal counts', 'empty generated sets', 'duplicate IDs skipped',
                         'label-blind swapping', 'roots unchanged', 'zero weight unchanged',
                         'optional context leaves logits unchanged', 'no extra global RNG consumption',
                         'auxiliary loss only updates generated branch', 'no extra BatchNorm updates',
                         'finite AMP backward and parameter update'],
              'evaluation_on_train_smoke_batch': result,
              'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    destination = path.parent / 'reports' / 'smoke_robustness.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'evaluation_on_train_smoke_batch'}, indent=2))
    print(f'report={destination}')


if __name__ == '__main__':
    main()
