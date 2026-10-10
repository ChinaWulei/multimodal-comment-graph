"""Degree-matched root shortcut experiment. Reuses the frozen trainer without editing it."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, stdev
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'comment_graph/outputs/grounded_structure_20261010'
OUT = HERE / 'outputs/round1'
CONFIG = ROOT / 'comment_graph/config_fakeddit_structure_expanded_generateddepth.yaml'
ARMS = {'F_root_parent': 'root_parent', 'G_root_permuted_parent': 'root_permuted_parent'}
SEEDS = [42, 13, 24]
PERMUTATION_SEED = 20261010


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def transform_edges(item, policy):
    """Add root shortcuts, optionally permuting predicted parents WITHIN this news.

    Both variants keep the degree of every node, all real edges and all node
    features identical. Permutations do not read text, labels, embeddings or scores.
    The result is an augmented graph, NOT a claimed ground-truth reply tree.
    """
    import torch
    if policy not in ARMS.values(): raise ValueError(policy)
    result = item.clone()
    real_mask = item.edge_type == 0
    original = item.edge_index[:, ~real_mask]
    targets = sorted((item.node_type == 2).nonzero(as_tuple=False).flatten().tolist())
    real_ids = set((item.node_type == 1).nonzero(as_tuple=False).flatten().tolist())
    parents = []
    for target in targets:
        incoming = original[0, original[1] == target].tolist()
        if len(incoming) != 1 or incoming[0] not in real_ids | {0}:
            raise ValueError('Expected exactly one news/real parent per observed-only generated node')
        parents.append(incoming[0])
    replacements = list(parents)
    eligible = [i for i, p in enumerate(parents) if p != 0]
    if policy == 'root_permuted_parent' and len(eligible) > 1:
        local_seed = int.from_bytes(hashlib.sha256(f'{PERMUTATION_SEED}:{item.news_id}'.encode()).digest()[:8], 'big')
        rng = random.Random(local_seed)
        candidates = [parents[i] for i in eligible]
        rng.shuffle(candidates)
        # If chance leaves the entire assignment unchanged, use a deterministic rotation.
        if candidates == [parents[i] for i in eligible] and len(set(candidates)) > 1:
            candidates = candidates[1:] + candidates[:1]
        for index, parent in zip(eligible, candidates): replacements[index] = parent
    assert Counter(parents) == Counter(replacements)
    pairs = []
    for target, parent in zip(targets, replacements):
        for source in ([0] if parent == 0 else [parent, 0]):
            pairs.extend([(source, target), (target, source)])
    added = torch.tensor(pairs, dtype=torch.long, device=item.edge_index.device).t().contiguous() if pairs else item.edge_index[:, :0]
    result.edge_index = torch.cat([item.edge_index[:, real_mask], added], dim=1)
    result.edge_type = torch.cat([item.edge_type[real_mask], item.edge_type.new_ones(len(pairs))])
    result.shortcut_count = torch.tensor([sum(p != 0 for p in parents)], dtype=torch.long)
    result.changed_parent_count = torch.tensor([sum(a != b for a, b in zip(parents, replacements))], dtype=torch.long)
    result.connection_policy = policy
    assert result.edge_index.size(1) == item.edge_index.size(1) + 2 * int(result.shortcut_count)
    assert not bool((result.edge_index[0] == result.edge_index[1]).any())
    assert len(set(map(tuple, result.edge_index.t().tolist()))) == result.edge_index.size(1)
    return result


class ConnectionDataset:
    def __init__(self, base, policy): self.base, self.policy = base, policy
    def __len__(self): return len(self.base)
    def __getattr__(self, key): return getattr(self.base, key)
    def __getitem__(self, index): return transform_edges(self.base[index], self.policy)


def configured():
    from comment_graph.common import load_config
    config, path = load_config(CONFIG)
    config.data.max_real_comments = 16; config.data.max_generated_comments = 5
    config.graph.structure_mode = 'observed_tree'; config.graph.message_passing = 'typed_rgcn'
    config.graph.pooling_mode = 'mean'; config.graph.generated_reliability.mode = 'none'
    config.decision_mode = 'concat'
    config.data.generated_comments.train = str(SOURCE / 'observed_only_train.jsonl')
    config.data.generated_comments.validate = str(SOURCE / 'observed_only_validate_primary.jsonl')
    return config, path


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((SOURCE / 'protocol.json').read_text(encoding='utf-8'))
    files = [Path(__file__), HERE / 'audit.py', CONFIG, SOURCE / 'protocol.json', SOURCE / 'comparison.json',
             *[SOURCE / f'observed_only_{s}.jsonl' for s in ['train', 'validate', 'validate_primary']],
             *[ROOT / 'comment_graph' / n for n in ['train.py', 'model.py', 'typed_graph.py', 'common.py',
                'dataset.py', 'fakeddit_structure_dataset.py', 'dynamic_graph.py', 'checkpoint_io.py']]]
    config, config_path = configured()
    from comment_graph.common import resolve_path
    files.extend([resolve_path(config_path, config.data.manifest_csv), resolve_path(config_path, config.data.real_trees_jsonl)])
    files.extend(sorted((SOURCE / 'classification').glob('*.json')))
    protocol = {'arms': ARMS, 'seeds': SEEDS, 'permutation_seed': PERMUTATION_SEED,
                'train_news': 292, 'primary_ids': original['primary_validation_news'], 'auxiliary_news': 77,
                'smoke_ids': original['smoke_news'], 'parameters': {'epochs': 20, 'patience': 5,
                    'real_comments': 16, 'generated_comments': 5, 'batch_size': 8, 'lr': 1e-4,
                    'loss': 'unchanged classification CE', 'message_passing': 'typed_rgcn', 'pooling': 'mean', 'fusion': 'concat'},
                'sources': {str(p): sha(p) for p in files},
                'hypothesis': 'A root shortcut may retain news context while preserving a local predicted connection.',
                'control': 'Permute non-root parent assignments within news; preserve every node degree and all real edges.',
                'success_gate': {'F_over': ['A_real', 'D_observed_root', 'G_root_permuted_parent'],
                                 'mean_macro_f1_gain_at_least': 0.01, 'positive_seeds_at_least': 2},
                'test_enabled': False,
                'caveats': ['Exploratory follow-up after inspecting reused validation; not a new blind experiment.',
                            'Shortcuts are synthetic graph edges, not ground-truth reply edges.',
                            'If all parents in a news are identical, the degree-matched permutation is unchanged.',
                            'No new generated text, labels in generation, extra attention, quality filter or loss.']}
    output = OUT / 'protocol.json'
    if output.exists() and json.loads(output.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Frozen sources/protocol changed; use a new round directory')
    save(output, protocol)
    return protocol


def verify():
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for path, expected in protocol['sources'].items():
        assert sha(path) == expected, f'Frozen source changed: {path}'
    return protocol


def smoke(protocol):
    import torch
    from torch_geometric.loader import DataLoader
    from comment_graph.common import make_dataset, make_tokenizer_and_transform, seed_everything
    from comment_graph.model import CommentGraphMFCL
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    seed_everything(42)
    config, path = configured(); tokenizer, transform = make_tokenizer_and_transform(config)
    base = make_dataset(config, path, 'train', 'real_generated', tokenizer, transform,
                        allowed_news_ids=set(protocol['smoke_ids']))
    assert len(base) == 8
    items = [base[i] for i in range(len(base))]
    paired = {a: [transform_edges(x, policy) for x in items] for a, policy in ARMS.items()}
    for original, f, g in zip(items, paired['F_root_parent'], paired['G_root_permuted_parent']):
        assert torch.equal(original.edge_index[:, original.edge_type == 0], f.edge_index[:, f.edge_type == 0])
        assert torch.equal(f.edge_index[:, f.edge_type == 0], g.edge_index[:, g.edge_type == 0])
        for key in ['node_input_ids', 'node_attention_mask', 'root_input_ids', 'root_attention_mask', 'node_type', 'pixel_values', 'y']:
            assert torch.equal(getattr(original, key), getattr(f, key)) and torch.equal(getattr(f, key), getattr(g, key))
        assert torch.equal(torch.bincount(f.edge_index[0], minlength=f.num_nodes), torch.bincount(g.edge_index[0], minlength=g.num_nodes))
        assert torch.equal(g.edge_index, transform_edges(original, 'root_permuted_parent').edge_index)
    # Exercise empty-generation handling on a copy, not as experimental data.
    synthetic = items[0].clone(); synthetic.node_type[synthetic.node_type == 2] = 1
    synthetic.edge_index = synthetic.edge_index[:, synthetic.edge_type == 0]; synthetic.edge_type = synthetic.edge_type[synthetic.edge_type == 0]
    assert transform_edges(synthetic, 'root_parent').edge_index.shape == synthetic.edge_index.shape
    report = {'passed': True, 'real_inputs_and_edges_unchanged': True, 'matched_node_degrees': True, 'arms': {}}
    model = CommentGraphMFCL(config, 'real_generated', graph_structure='observed_tree', pooling_mode='mean', decision_mode='concat').cuda()
    for arm, data in paired.items():
        batch = next(iter(DataLoader(data, batch_size=8))).cuda()
        model.train(); model.zero_grad(set_to_none=True)
        result = model(batch)
        loss = torch.nn.functional.cross_entropy(result['logits'], batch.y.view(-1)); loss.backward()
        grads = [p.grad for p in model.typed_graph_encoder.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
        report['arms'][arm] = {'loss': float(loss.detach()), 'gnn_has_finite_nonzero_grad': True,
            'nodes': [x.num_nodes for x in data], 'directed_edges': [x.edge_index.size(1) for x in data],
            'h_mm': list(result['h_mm'].shape), 'h_graph': list(result['h_graph'].shape)}
    # Full graph-only audit: no image encoding, labels or validation predictions are used to choose permutations.
    from torch_geometric.data import Data
    from comment_graph.fakeddit_structure_dataset import bidirectional
    report['coverage'] = {}
    for split, suffix in [('train', 'train'), ('validate', 'validate')]:
        config.data.generated_comments[split] = str(SOURCE / f'observed_only_{suffix}.jsonl')
        dataset = make_dataset(config, path, split, 'real_generated', tokenizer, transform,
                               allowed_news_ids={r['news_id'] for r in read(SOURCE / f'observed_only_{suffix}.jsonl')})
        counters = Counter()
        for row in dataset.records:
            texts, types, edges = dataset._nodes_and_edges(row['news_id'])
            edge_index, edge_type = bidirectional(edges, set(range(len(types) + 1)))
            item = Data(news_id=row['news_id'], node_type=torch.tensor([0, *types]), edge_index=edge_index, edge_type=edge_type, num_nodes=len(types) + 1)
            f, g = [transform_edges(item, p) for p in ARMS.values()]
            assert torch.equal(torch.bincount(f.edge_index[0], minlength=f.num_nodes), torch.bincount(g.edge_index[0], minlength=g.num_nodes))
            counters['news'] += 1; counters['generated'] += types.count(2)
            counters['shortcuts'] += int(f.shortcut_count); counters['permuted_parent_assignments'] += int(g.changed_parent_count)
            counters['news_with_changed_parent'] += int(g.changed_parent_count > 0)
        report['coverage'][split] = dict(counters)
    save(OUT / 'smoke_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def train_one(arm, seed):
    import torch
    import comment_graph.train as trainer
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    protocol = verify(); policy = ARMS[arm]
    original_factory = trainer.make_dataset
    def factory(config, *args, **kwargs):
        config.graph.connection_followup = {'policy': policy, 'permutation_seed': PERMUTATION_SEED,
                                           'requires_connection_followup_loader': True, 'protocol_sha256': sha(OUT / 'protocol.json')}
        return ConnectionDataset(original_factory(config, *args, **kwargs), policy)
    trainer.make_dataset = factory
    summary = OUT / 'classification' / f'{arm}_seed{seed}.json'
    sys.argv = [str(ROOT / 'comment_graph/train.py'), '--config', str(CONFIG), '--mode', 'real_generated',
        '--graph-structure', 'observed_tree', '--message-passing', 'typed_rgcn', '--pooling-mode', 'mean',
        '--decision-mode', 'concat', '--generated-reliability', 'none', '--mm-fusion', 'mfcl', '--evidence-mode', 'none',
        '--generated-train-jsonl', str(SOURCE / 'observed_only_train.jsonl'),
        '--generated-validate-jsonl', str(SOURCE / 'observed_only_validate_primary.jsonl'),
        '--max-real-comments', '16', '--max-generated-comments', '5', '--epochs', '20', '--patience', '5',
        '--seed', str(seed), '--allow-formal-training', '--checkpoint-dir', str(OUT / 'checkpoints' / f'{arm}_seed{seed}'),
        '--summary-json', str(summary)]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    trainer.main()
    record = json.loads(summary.read_text(encoding='utf-8'))
    record.update({'connection_followup': {'policy': policy, 'permutation_seed': PERMUTATION_SEED},
                   'protocol_sha256': sha(OUT / 'protocol.json'), 'checkpoint_sha256': sha(record['checkpoint']),
                   'session_seconds': time.perf_counter() - started,
                   'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3})
    assert record['train_size'] == 292 and record['validate_size'] == 44 and 'test' not in record
    assert {r['news_id'] for r in record['validation']['prediction_records']} == set(protocol['primary_ids'])
    save(summary, record)


def evaluate_aux(arm, seed):
    import torch
    from munch import DefaultMunch
    from comment_graph.common import make_dataset, make_tokenizer_and_transform, seed_everything
    from comment_graph.model import CommentGraphMFCL
    from comment_graph.train import evaluate, make_loader
    verify()
    record = json.loads((OUT / 'classification' / f'{arm}_seed{seed}.json').read_text(encoding='utf-8'))
    assert sha(record['checkpoint']) == record['checkpoint_sha256']
    checkpoint = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)
    config = DefaultMunch.fromDict(checkpoint['config'])
    assert config.graph.connection_followup.policy == ARMS[arm]
    assert config.graph.connection_followup.protocol_sha256 == sha(OUT / 'protocol.json')
    config.data.generated_comments.validate = str(SOURCE / 'observed_only_validate.jsonl')
    seed_everything(seed)
    tokenizer, transform = make_tokenizer_and_transform(config)
    ids = {r['news_id'] for r in read(config.data.generated_comments.validate)}
    dataset = ConnectionDataset(make_dataset(config, CONFIG, 'validate', 'real_generated', tokenizer, transform, allowed_news_ids=ids), ARMS[arm])
    assert len(dataset) == 77
    model = CommentGraphMFCL(config, 'real_generated', graph_structure='observed_tree', pooling_mode='mean', decision_mode='concat').cuda()
    model.load_state_dict(checkpoint['model'], strict=True)
    result = evaluate(model, make_loader(dataset, config, False), torch.device('cuda'), bool(config.training.use_amp), None)
    save(OUT / 'classification' / f'{arm}_seed{seed}_validation77.json', {'validation': result,
         'checkpoint_sha256': record['checkpoint_sha256'], 'protocol_sha256': sha(OUT / 'protocol.json'),
         'used_for_checkpoint_selection': False, 'connection_policy': ARMS[arm]})


def child(stage, arm=None, seed=None):
    name = stage if arm is None else f'{stage}_{arm}_{seed}'
    args = [sys.executable, '-u', str(Path(__file__)), '--stage', stage]
    if arm: args.extend(['--arm', arm, '--seed', str(seed)])
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    save(OUT / 'status.json', {'state': 'running', 'step': name})
    print('RUN', name, flush=True)
    with (OUT / 'logs' / f'{name}.log').open('a', encoding='utf-8') as log:
        result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{name} failed; see logs')
    print('DONE', name, flush=True)


def aggregate():
    protocol = verify(); all_runs = {}
    for arm in [*ARMS, 'A_real', 'D_observed_root', 'E_observed_parent']:
        folder = OUT if arm in ARMS else SOURCE
        all_runs[arm] = [json.loads((folder / 'classification' / f'{arm}_seed{s}.json').read_text(encoding='utf-8')) for s in SEEDS]
    metrics = {}
    for arm, runs in all_runs.items():
        for row in runs:
            assert {r['news_id'] for r in row['validation']['prediction_records']} == set(protocol['primary_ids'])
        metrics[arm] = {key: {'mean': mean(r['validation'][key] for r in runs), 'std': stdev(r['validation'][key] for r in runs),
                              'by_seed': {str(s): r['validation'][key] for s, r in zip(SEEDS, runs)}}
                        for key in ['accuracy', 'macro_f1', 'weighted_f1']}
    comparisons = {}
    for baseline in protocol['success_gate']['F_over']:
        deltas = [f['validation']['macro_f1'] - b['validation']['macro_f1'] for f, b in zip(all_runs['F_root_parent'], all_runs[baseline])]
        comparisons[baseline] = {'mean_gain': mean(deltas), 'positive_seeds': sum(d > 0 for d in deltas),
            'by_seed': dict(zip(map(str, SEEDS), deltas)), 'passes': mean(deltas) >= .01 and sum(d > 0 for d in deltas) >= 2}
    auxiliary = {}
    for arm in ARMS:
        rows = [json.loads((OUT / 'classification' / f'{arm}_seed{s}_validation77.json').read_text(encoding='utf-8')) for s in SEEDS]
        auxiliary[arm] = {key: {'mean': mean(r['validation'][key] for r in rows), 'std': stdev(r['validation'][key] for r in rows)}
                          for key in ['accuracy', 'macro_f1', 'weighted_f1']}
    save(OUT / 'comparison.json', {'primary44': metrics, 'auxiliary77': auxiliary, 'F_minus_baseline': comparisons,
        'validation_gate_passed': all(r['passes'] for r in comparisons.values()), 'test_executed': False,
        'interpretation': 'Exploratory follow-up. No new blind validation or test claims.'})
    lines = ['# 新闻根节点补充连接实验', '', '生成评论内容、真实评论、真实边和训练参数保持不变；仅改变生成节点连接。', '',
             '| 组别 | Accuracy 均值 | Macro-F1 均值 | Macro-F1 标准差（百分点） |', '|---|---:|---:|---:|']
    for arm, row in metrics.items():
        lines.append(f"| {arm} | {row['accuracy']['mean']:.2%} | {row['macro_f1']['mean']:.2%} | {100*row['macro_f1']['std']:.2f} |")
    lines.extend(['', 'F：新闻根节点＋预测父节点；G：根节点＋打乱后的父节点，保持每个节点度数一致。',
                  '真实评论没有改写，生成评论没有重生成；额外边是图增强边，不声称真实传播关系。',
                  '主验证44条，三种训练种子；完整77条仅作冻结检查点辅助评估。已探索数据，不是盲测。', '', '## 预设门槛'])
    for name, result in comparisons.items():
        lines.append(f"- F 相比 {name}：平均 {100*result['mean_gain']:+.2f} 个百分点，{result['positive_seeds']}/3 种子提升，通过={result['passes']}。")
    lines.extend(['', '测试集未运行。不能因为某一个种子提升就宣称方法有效。'])
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'primary44': metrics, 'comparisons': comparisons}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'train-one', 'auxiliary', 'full', 'summarize'], default='prepare')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    args = parser.parse_args()
    if args.stage == 'train-one': train_one(args.arm, args.seed); return
    if args.stage == 'auxiliary': evaluate_aux(args.arm, args.seed); return
    if args.stage == 'smoke': smoke(verify()); return
    from comment_graph.run_grounded_structure import exclusive_run
    OUT.mkdir(parents=True, exist_ok=True)
    with exclusive_run(OUT):
        try:
            protocol = freeze()
            if args.stage == 'full':
                child('smoke')
                assert json.loads((OUT / 'smoke_report.json').read_text(encoding='utf-8'))['passed']
                for seed in SEEDS:
                    for arm in ARMS:
                        summary = OUT / 'classification' / f'{arm}_seed{seed}.json'
                        if summary.exists():
                            row = json.loads(summary.read_text(encoding='utf-8'))
                            assert row['protocol_sha256'] == sha(OUT / 'protocol.json')
                            assert row['connection_followup']['policy'] == ARMS[arm] and row['seed'] == seed
                            assert row['checkpoint_sha256'] == sha(row['checkpoint'])
                        else: child('train-one', arm, seed)
                        auxiliary = OUT / 'classification' / f'{arm}_seed{seed}_validation77.json'
                        if not auxiliary.exists(): child('auxiliary', arm, seed)
                        aux = json.loads(auxiliary.read_text(encoding='utf-8'))
                        assert aux['protocol_sha256'] == sha(OUT / 'protocol.json')
                aggregate()
            if args.stage == 'summarize': aggregate()
            save(OUT / 'status.json', {'state': 'completed', 'stage': args.stage})
        except BaseException as error:
            save(OUT / 'status.json', {'state': 'failed', 'stage': args.stage, 'error': str(error)})
            raise


if __name__ == '__main__': main()
