"""Formula, zero-input, gradient and real-image tests for the complete MTS core."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch_geometric.loader import DataLoader
from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform, seed_everything
from comment_graph.model import CommentGraphMFCL
from comment_graph.mts_fusion import TaylorLayer
from comment_graph.isolated_fusion import classification_objective


def formula_check():
    torch.manual_seed(17)
    layer = TaylorLayer(7).double()
    x, y = torch.randn(4, 7, dtype=torch.float64), torch.randn(4, 7, dtype=torch.float64)
    a, b = layer(x, y)
    xn, yn = x / x.abs().max(), y / y.abs().max()
    def reference(own, other, p):
        return own @ p[0] + (other @ p[1]) * (own @ p[2]) + (own @ p[3]) * (own @ p[4])
    expected = reference(xn, yn, layer.image_direction.coefficients) + reference(yn, xn, layer.text_direction.coefficients)
    assert torch.allclose(a + b, expected, atol=1e-12, rtol=1e-12)
    (a + b).square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
    zeros = torch.zeros_like(x)
    assert all(torch.isfinite(z).all() and z.count_nonzero() == 0 for z in layer(zeros, zeros))
    return {'equation_18_max_error': float((a + b - expected).abs().max().detach()),
            'zero_input_finite': True, 'all_ten_matrices_have_gradient': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a new report path')
    report = {'formula_unit_test': formula_check(), 'datasets': {}, 'test_used': False}
    if not torch.cuda.is_available():
        raise SystemExit('Formula tests passed; real training smoke needs CUDA')
    for name, config_name, mode, decision in [('weibo', 'config_weibo.yaml', 'baseline', 'concat'),
                                             ('checked', 'config.yaml', 'real_generated', 'source_isolated')]:
        config, path = load_config(Path(__file__).with_name(config_name))
        config.mm_fusion = {'mode': 'mts', 'hidden_dim': 768, 'layers': 8}
        seed_everything(42)
        tokenizer, transform = make_tokenizer_and_transform(config)
        dataset = make_dataset(config, path, 'train', mode, tokenizer, transform)
        labels = dataset.labels()
        chosen = [i for label in (0, 1) for i in [j for j, y in enumerate(labels) if y == label][:4]]
        batch = next(iter(DataLoader(torch.utils.data.Subset(dataset, chosen), batch_size=8))).to('cuda')
        model = CommentGraphMFCL(config, mode, 'dynamic_cross_modal' if name == 'checked' else 'star',
                                'mean', decision).cuda().train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        before = model.mts_fusion.layers[0].image_direction.coefficients[0].detach().clone()
        with torch.autocast('cuda'):
            output = model(batch)
            loss, final_ce, base_ce = classification_objective(output, batch.y.view(-1), None,
                                                             .5 if decision == 'source_isolated' else 0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        gradients = [float(p.grad.abs().sum()) for p in model.mts_fusion.parameters() if p.grad is not None]
        assert len(gradients) == len(list(model.mts_fusion.parameters())) and min(gradients) > 0
        gnn_grad = None
        if name == 'checked':
            gnn_grad = sum(float(p.grad.abs().sum()) for p in model.graph_model.encoder.parameters() if p.grad is not None)
            assert gnn_grad > 0
        scaler.step(optimizer)
        scaler.update()
        assert not torch.equal(before, model.mts_fusion.layers[0].image_direction.coefficients[0])
        torch.cuda.synchronize()
        report['datasets'][name] = {'train_count': len(dataset), 'smoke_news': 8, 'mode': mode,
                                   'news_ids': batch.news_id, 'node_count': batch.num_nodes,
                                   'edge_count': batch.edge_index.shape[1], 'loss': float(loss.detach()),
                                   'h_mm_dim': output['h_mm'].shape[-1], 'h_graph_dim': output['h_graph'].shape[-1],
                                   'mts_layers': 8, 'mts_parameters': sum(p.numel() for p in model.mts_fusion.parameters()),
                                   'all_mts_gradients_nonzero_finite': True, 'base_gnn_gradient_l1': gnn_grad,
                                   'optimizer_step_seconds': time.perf_counter() - start,
                                   'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
        print(json.dumps(report['datasets'][name], indent=2), flush=True)
        del optimizer, model, output, loss, final_ce, base_ce, before, batch, scaler
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'PASSED: {args.output}')


if __name__ == '__main__':
    main()
