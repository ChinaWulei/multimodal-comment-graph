"""Placement-only diagnostic: frozen content, no root shortcuts, no test selection."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from comment_graph.connection_followup import experiment as previous

SOURCE = previous.SOURCE
OUT = HERE / 'outputs/round1'
SEEDS = [42, 13, 24]
ARMS = {'H_permuted_parent': 'permuted_parent_only'}
sha, save, read = previous.sha, previous.save, previous.read
original_transform = previous.transform_edges


def transform_edges(item, policy='permuted_parent_only'):
    import torch
    if policy != 'permuted_parent_only': raise ValueError(policy)
    # Reuse the exact previously audited deterministic, degree-matched permutation.
    # Its implementation validates against the module's ARMS at call time.
    saved_arms = previous.ARMS
    try:
        previous.ARMS = {'F': 'root_parent', 'G': 'root_permuted_parent'}
        result = original_transform(item, 'root_permuted_parent')
    finally:
        previous.ARMS = saved_arms
    generated = set((item.node_type == 2).nonzero(as_tuple=False).flatten().tolist())
    existing_root_children = {int(t) for s, t in item.edge_index.t().tolist() if s == 0 and t in generated}
    shortcut_children = generated - existing_root_children
    keep = [not ((s == 0 and t in shortcut_children) or (t == 0 and s in shortcut_children))
            for s, t in result.edge_index.t().tolist()]
    mask = torch.tensor(keep, dtype=torch.bool, device=result.edge_index.device)
    result.edge_index = result.edge_index[:, mask]
    result.edge_type = result.edge_type[mask]
    result.shortcut_count = torch.tensor([0], dtype=torch.long)
    result.connection_policy = policy
    assert result.edge_index.shape == item.edge_index.shape
    assert torch.equal(torch.bincount(result.edge_index[0], minlength=item.num_nodes),
                       torch.bincount(item.edge_index[0], minlength=item.num_nodes))
    assert torch.equal(result.edge_index[:, result.edge_type == 0], item.edge_index[:, item.edge_type == 0])
    return result


def freeze():
    original = json.loads((SOURCE / 'protocol.json').read_text(encoding='utf-8'))
    execution = json.loads((SOURCE / 'execution_lock.json').read_text(encoding='utf-8'))
    for path, expected in execution['code_sha256'].items():
        assert sha(path) == expected, f'Original frozen execution changed: {path}'
    files = [*map(Path, execution['code_sha256']), Path(previous.__file__), HERE / 'experiment.py',
             HERE / 'test_protocol.py', SOURCE / 'protocol.json', SOURCE / 'execution_lock.json',
             *[SOURCE / f'observed_only_{s}.jsonl' for s in ['train', 'validate', 'validate_primary']]]
    config, path = previous.configured()
    from comment_graph.common import resolve_path
    files += [resolve_path(path, config.data.manifest_csv), resolve_path(path, config.data.real_trees_jsonl)]
    files += [SOURCE / 'classification' / f'{a}_seed{s}.json'
              for a in ['A_real', 'D_observed_root', 'E_observed_parent'] for s in SEEDS]
    protocol = {'arms': ARMS, 'seeds': SEEDS, 'primary_ids': original['primary_validation_news'],
        'smoke_ids': original['smoke_news'], 'train_news': 292, 'auxiliary_news': 77,
        'permutation_seed': previous.PERMUTATION_SEED,
        'sources': {str(p): sha(p) for p in files},
        'parameters': {'epochs': 20, 'patience': 5, 'real_comments': 16, 'generated_comments': 5,
                       'batch_size': 8, 'learning_rate': 1e-4, 'message_passing': 'typed_rgcn',
                       'pooling': 'mean', 'decision': 'concat', 'quality_gate': False},
        'hypothesis': 'Predicted parent assignment should outperform degree-matched shuffled assignment if placement carries useful information.',
        'comparison': 'Reuse frozen A/D/E, train H with identical text/features/training and no shortcuts.',
        'success_gate': {'E_over': ['A_real', 'D_observed_root', 'H_permuted_parent'],
                         'mean_macro_f1_gain_at_least': .01, 'positive_seeds_at_least': 2},
        'test_enabled': False,
        'limitations': ['Explored validation, not blind.', 'Same-parent graphs cannot change under degree-preserving permutation.',
                       'One predeclared graph permutation; three training seeds.', 'No generated edge is claimed as a true reply edge.']}
    path = OUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Frozen protocol changed: use a new output directory')
    save(path, protocol)
    return protocol


def verify():
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for p, expected in protocol['sources'].items(): assert sha(p) == expected, p
    return protocol


def install():
    previous.OUT = OUT
    previous.ARMS = ARMS
    previous.verify = verify
    previous.transform_edges = transform_edges


def smoke():
    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    from comment_graph.common import make_dataset, make_tokenizer_and_transform, seed_everything
    from comment_graph.fakeddit_structure_dataset import bidirectional
    from comment_graph.model import CommentGraphMFCL
    protocol = verify()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    seed_everything(42)
    config, path = previous.configured()
    tokenizer, image_transform = make_tokenizer_and_transform(config)
    base = make_dataset(config, path, 'train', 'real_generated', tokenizer, image_transform,
                        allowed_news_ids=set(protocol['smoke_ids']))
    assert len(base) == 8
    originals = [base[i] for i in range(len(base))]
    changed = [transform_edges(x) for x in originals]
    for a, b in zip(originals, changed):
        for key in ['node_input_ids', 'node_attention_mask', 'root_input_ids', 'root_attention_mask', 'node_type', 'pixel_values', 'y']:
            assert torch.equal(getattr(a, key), getattr(b, key)), key
        assert torch.equal(transform_edges(a).edge_index, b.edge_index)
    model = CommentGraphMFCL(config, 'real_generated', graph_structure='observed_tree', pooling_mode='mean', decision_mode='concat').cuda()
    batch = next(iter(DataLoader(changed, batch_size=8))).cuda()
    model.train(); result = model(batch)
    loss = torch.nn.functional.cross_entropy(result['logits'], batch.y.view(-1)); loss.backward()
    grads = [p.grad for p in model.typed_graph_encoder.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
    report = {'passed': True, 'gnn_grad_finite_nonzero': True, 'image_and_text_unchanged': True,
              'real_edges_unchanged': True, 'node_degrees_and_edge_counts_matched': True,
              'loss': float(loss.detach()), 'h_mm': list(result['h_mm'].shape), 'h_graph': list(result['h_graph'].shape),
              'nodes': [x.num_nodes for x in changed], 'edges': [x.edge_index.size(1) for x in changed], 'coverage': {}}
    for split in ['train', 'validate']:
        graph_path = SOURCE / f'observed_only_{split}.jsonl'
        config.data.generated_comments[split] = str(graph_path)
        ds = make_dataset(config, path, split, 'real_generated', tokenizer, image_transform,
                          allowed_news_ids={r['news_id'] for r in read(graph_path)})
        counts = Counter()
        for row in ds.records:
            _, types, edges = ds._nodes_and_edges(row['news_id'])
            edge_index, edge_type = bidirectional(edges, set(range(len(types) + 1)))
            item = Data(news_id=row['news_id'], node_type=torch.tensor([0, *types]), edge_index=edge_index,
                        edge_type=edge_type, num_nodes=len(types) + 1)
            altered = transform_edges(item)
            counts['news'] += 1; counts['generated'] += types.count(2)
            counts['changed_assignments'] += int(altered.changed_parent_count)
            counts['news_with_changed_parent'] += int(altered.changed_parent_count > 0)
        report['coverage'][split] = dict(counts)
    save(OUT / 'smoke_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def summarize():
    import numpy as np
    from sklearn.metrics import f1_score
    protocol = verify(); ids = sorted(protocol['primary_ids'])
    arms = ['A_real', 'D_observed_root', 'E_observed_parent', 'H_permuted_parent']
    metrics, predictions, truth, auxiliary = {}, {}, None, {}
    for arm in arms:
        folder = OUT if arm in ARMS else SOURCE
        runs = []; predictions[arm] = []
        for seed in SEEDS:
            row = json.loads((folder / 'classification' / f'{arm}_seed{seed}.json').read_text(encoding='utf-8'))
            by_id = {p['news_id']: p for p in row['validation']['prediction_records']}
            assert set(by_id) == set(ids)
            y = np.array([by_id[n]['label'] for n in ids]); pred = np.array([by_id[n]['prediction'] for n in ids])
            if truth is None: truth = y
            assert np.array_equal(truth, y)
            assert abs(f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0) - row['validation']['macro_f1']) < 1e-12
            if arm in ARMS:
                assert row['protocol_sha256'] == sha(OUT / 'protocol.json')
                assert row['checkpoint_sha256'] == sha(row['checkpoint'])
                aux = json.loads((folder / 'classification' / f'{arm}_seed{seed}_validation77.json').read_text(encoding='utf-8'))
                assert aux['checkpoint_sha256'] == row['checkpoint_sha256']
                assert len(aux['validation']['prediction_records']) == 77
                assert {p['news_id'] for p in aux['validation']['prediction_records']} >= set(ids)
                auxiliary[str(seed)] = aux['validation']
            predictions[arm].append(pred); runs.append(row)
        predictions[arm] = np.stack(predictions[arm])
        metrics[arm] = {key: {'mean': mean(r['validation'][key] for r in runs), 'std': stdev(r['validation'][key] for r in runs),
                              'by_seed': {str(s): r['validation'][key] for s, r in zip(SEEDS, runs)}}
                        for key in ['accuracy', 'macro_f1', 'weighted_f1']}
    def fast_f1(y, p):
        cm = np.bincount(3*y+p, minlength=9).reshape(3, 3)
        denominator = cm.sum(0) + cm.sum(1)
        return np.divide(2*np.diag(cm), denominator, out=np.zeros(3), where=denominator != 0).mean()
    samples = np.random.default_rng(20261010).integers(0, len(ids), size=(2000, len(ids)))
    boot = {a: np.array([mean(fast_f1(truth[i], p[i]) for p in rows) for i in samples]) for a, rows in predictions.items()}
    comparisons = {}
    for baseline in protocol['success_gate']['E_over']:
        deltas = [metrics['E_observed_parent']['macro_f1']['by_seed'][str(s)] - metrics[baseline]['macro_f1']['by_seed'][str(s)] for s in SEEDS]
        comparisons[baseline] = {'mean_gain': mean(deltas), 'positive_seeds': sum(d > 0 for d in deltas),
            'by_seed': dict(zip(map(str, SEEDS), deltas)), 'passes': mean(deltas) >= .01 and sum(d > 0 for d in deltas) >= 2,
            'paired_news_bootstrap_95_interval': np.quantile(boot['E_observed_parent'] - boot[baseline], [.025, .975]).tolist()}
    save(OUT / 'comparison.json', {'primary44': metrics, 'H_auxiliary77': auxiliary, 'E_minus_baseline': comparisons,
        'gate_passed': all(x['passes'] for x in comparisons.values()), 'test_executed': False,
        'audit': {'source_hashes_verified': True, 'macro_f1_recomputed': True, 'paired_ids_and_labels_checked': True},
        'caveat': 'Exploratory reused validation; bootstrap omits model-selection uncertainty.'})
    print(json.dumps({'primary44': metrics, 'E_minus_baseline': comparisons}, indent=2), flush=True)


def child(stage, seed=42):
    name = f'{stage}_{seed}'
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    save(OUT / 'status.json', {'state': 'running', 'step': name})
    print('RUN', name, flush=True)
    with (OUT / 'logs' / f'{name}.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--stage', stage, '--seed', str(seed)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try: code = process.wait()
        except KeyboardInterrupt:
            process.terminate(); process.wait(); raise
    if code: raise RuntimeError(f'{name} failed; see logs')
    print('DONE', name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'full', 'train-one', 'auxiliary', 'summarize'], default='prepare')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    args = parser.parse_args()
    if args.stage in ['train-one', 'auxiliary']:
        verify(); assert json.loads((OUT / 'smoke_report.json').read_text(encoding='utf-8'))['passed']
        install()
        if args.stage == 'train-one': previous.train_one('H_permuted_parent', args.seed)
        else: previous.evaluate_aux('H_permuted_parent', args.seed)
        return
    if args.stage == 'smoke': smoke(); return
    from comment_graph.run_grounded_structure import exclusive_run
    OUT.mkdir(parents=True, exist_ok=True)
    with exclusive_run(OUT):
        try:
            freeze()
            if args.stage == 'full':
                child('smoke')
                for seed in SEEDS:
                    path = OUT / 'classification' / f'H_permuted_parent_seed{seed}.json'
                    if not path.exists(): child('train-one', seed)
                    row = json.loads(path.read_text(encoding='utf-8'))
                    assert row['protocol_sha256'] == sha(OUT / 'protocol.json') and row['checkpoint_sha256'] == sha(row['checkpoint'])
                    if not path.with_name(path.stem + '_validation77.json').exists(): child('auxiliary', seed)
                summarize()
            elif args.stage == 'summarize': summarize()
            save(OUT / 'status.json', {'state': 'completed', 'stage': args.stage})
        except BaseException as exc:
            save(OUT / 'status.json', {'state': 'failed', 'error': str(exc)}); raise


if __name__ == '__main__': main()
