"""Bounded paired CHECKED pilot, validation only; never loads the test split.

Both arms use the identical proportional train subset, initial model, batch order,
optimizer and clean losses. Only the mismatch regularizer differs.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import Subset
from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform, seed_everything
from comment_graph.isolated_fusion import classification_objective
from comment_graph.model import CommentGraphMFCL
from comment_graph.robustness import add_mismatch_loss
from comment_graph.train import make_loader, class_weights, evaluate


def proportional_subset(labels, size, seed):
    if not 2 <= size <= len(labels):
        raise ValueError(f'train-size must be in [2, {len(labels)}]')
    counts = Counter(labels)
    quotas = {y: size * n / len(labels) for y, n in counts.items()}
    allocations = {y: math.floor(q) for y, q in quotas.items()}
    for y in sorted(quotas, key=lambda y: (quotas[y] - allocations[y], y), reverse=True)[:size - sum(allocations.values())]:
        allocations[y] += 1
    if min(allocations.values()) == 0:
        raise ValueError('Increase train-size to retain both classes proportionally')
    rng = random.Random(seed)
    indices = []
    for y in sorted(counts):
        candidates = [i for i, value in enumerate(labels) if value == y]
        indices.extend(rng.sample(candidates, allocations[y]))
    return sorted(indices)


def trainable_digest(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--train-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: no training started on CPU')
    if not 1 <= args.epochs <= 3 or not 8 <= args.train_size <= 256:
        raise SystemExit('Pilot guard: epochs must be 1..3 and train-size 8..256; use train.py for full training')
    config, path = load_config(args.config)
    config.seed = args.seed
    config.comment_graph_mode = 'real_generated'
    config.decision_mode = 'source_isolated'
    config.graph.structure_mode = 'dynamic_cross_modal'
    config.graph.pooling_mode = 'mean'
    config.training.epochs = args.epochs
    if args.train_size % int(config.training.batch_size):
        raise SystemExit('Use train-size divisible by batch_size to avoid silently dropping pilot samples')
    destination = (args.output_dir or path.parent / 'outputs' / f'robustness_pilot_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    destination.mkdir(parents=True, exist_ok=False)
    tokenizer, transform = make_tokenizer_and_transform(config)
    full = make_dataset(config, path, 'train', 'real_generated', tokenizer, transform)
    validation = make_dataset(config, path, 'validate', 'real_generated', tokenizer, transform)
    train_ids = {r['news_id'] for r in full.records}
    val_ids = {r['news_id'] for r in validation.records}
    assert train_ids.isdisjoint(val_ids), 'Train/validation ID leakage'
    assert not full.missing_generated_ids and not validation.missing_generated_ids
    chosen = proportional_subset(full.labels(), args.train_size, args.seed)
    subset = Subset(full, chosen)
    # Same full-training-prior weights in both arms, matching formal train.py.
    device = torch.device('cuda')
    weights = class_weights(full, device) if config.training.class_weighted_loss else None
    use_amp = bool(config.training.use_amp)
    audit_seed = int(config.robustness.audit_seed)
    if float(config.robustness.mismatch_weight) <= 0:
        raise ValueError('A paired pilot requires positive mismatch_weight in the experimental arm')
    base_weight = float(config.isolated_fusion.base_loss_weight)
    report = {'purpose': 'bounded validation-only pilot, not a final efficacy result',
              'seed': args.seed, 'config': config.toDict(), 'train_size': len(subset),
              'train_class_counts': dict(Counter(full.labels()[i] for i in chosen)),
              'train_news_ids': [full.records[i]['news_id'] for i in chosen],
              'validation_size': len(validation), 'validation_class_counts': dict(Counter(validation.labels())),
              'class_weights': weights.cpu().tolist() if weights is not None else None,
              'class_weight_source': 'full train split only', 'test_loaded': False,
              'selection': 'highest clean validation macro-F1, strict improvement; ties keep earlier epoch',
              'checkpoint_saved': False, 'arms': {}}
    report_path = destination / 'results.json'
    print(f'output={destination} train={len(subset)} validation={len(validation)}; TEST NOT LOADED', flush=True)
    initial_digest = None
    for arm, coefficient in [('control', 0.), ('mismatch_fallback', float(config.robustness.mismatch_weight))]:
        seed_everything(args.seed)
        train_loader = make_loader(subset, config, shuffle=True)
        val_loader = make_loader(validation, config, shuffle=False)
        model = CommentGraphMFCL(config, 'real_generated', 'dynamic_cross_modal', 'mean', 'source_isolated').to(device)
        digest = trainable_digest(model)
        if initial_digest is None:
            initial_digest = digest
        assert digest == initial_digest, 'Paired arms differ at initialization'
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.feature_encoder.parameters() if p.requires_grad],
             'lr': float(config.encoder.learning_rate)},
            {'params': [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('feature_encoder.')],
             'lr': float(config.training.learning_rate)}], weight_decay=float(config.training.weight_decay))
        scaler = GradScaler('cuda', enabled=use_amp, init_scale=float(config.isolated_fusion.amp_init_scale))
        mismatch_rng = torch.Generator().manual_seed(args.seed + 10000)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        best, best_epoch, history, batch_order = None, None, [], []
        for epoch in range(1, args.epochs + 1):
            model.train()
            clean_sum = base_sum = total_sum = denominator_sum = kl_sum = 0.
            kl_count = steps = 0
            for batch in train_loader:
                batch_order.append(list(batch.news_id))
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', enabled=use_amp):
                    output = model(batch, return_generated_context=coefficient > 0)
                    y = batch.y.view(-1)
                    clean_loss, final_ce, base_ce = classification_objective(output, y, weights, base_weight)
                    loss, kl, count = add_mismatch_loss(model, output, clean_loss, mismatch_rng, coefficient)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss in {arm}')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise RuntimeError(f'Nonfinite gradients in {arm}; reduce AMP scale before retrying')
                scaler.step(optimizer)
                scaler.update()
                steps += 1
                denominator = float(weights[y].sum()) if weights is not None else len(y)
                clean_sum += float(final_ce.detach()) * denominator
                base_sum += float(base_ce.detach()) * denominator
                total_sum += float(loss.detach()) * denominator
                denominator_sum += denominator
                kl_sum += float(kl.detach()) * count
                kl_count += count
            val = evaluate(model, val_loader, device, use_amp, weights, True, audit_seed)
            history.append({'epoch': epoch, 'steps': steps, 'train_total_loss': total_sum / denominator_sum,
                            'train_final_ce': clean_sum / denominator_sum, 'train_base_ce': base_sum / denominator_sum,
                            'train_mismatch_kl': kl_sum / max(kl_count, 1), 'validation': val})
            if best is None or val['macro_f1'] > best['macro_f1']:
                best, best_epoch = val, epoch
            print(f"arm={arm} epoch={epoch} train_ce={clean_sum / denominator_sum:.6f} "
                  f"kl={kl_sum / max(kl_count, 1):.7f} val_ce={val['loss']:.6f} val_f1={val['macro_f1']:.4f}", flush=True)
        order_digest = hashlib.sha256(json.dumps(batch_order).encode()).hexdigest()
        if report['arms']:
            assert order_digest == report['arms']['control']['batch_order_sha256']
        report['arms'][arm] = {'coefficient': coefficient, 'initial_trainable_sha256': digest,
                               'batch_order_sha256': order_digest, 'best_epoch': best_epoch, 'validation': best,
                               'history': history, 'runtime_seconds': time.perf_counter() - start,
                               'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        del model, optimizer, scaler, output, loss, clean_loss, final_ce, base_ce, kl
        gc.collect()
        torch.cuda.empty_cache()
    print(f'Completed paired pilot: {report_path}', flush=True)


if __name__ == '__main__':
    main()
