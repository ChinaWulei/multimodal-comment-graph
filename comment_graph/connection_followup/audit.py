"""Inspect fixed D/E predictions and train/validation attachment patterns; never tune by test."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'comment_graph/outputs/grounded_structure_20261010'
OUT = Path(__file__).resolve().parent / 'outputs/round1'


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def analyze():
    result = {'prediction_pairs': {}, 'graph_statistics': {}, 'test_used': False}
    cases = {}
    for seed in [42, 13, 24]:
        rows = {}
        for name in ['D_observed_root', 'E_observed_parent']:
            summary = json.loads((SOURCE / 'classification' / f'{name}_seed{seed}.json').read_text(encoding='utf-8'))
            rows[name] = {r['news_id']: r for r in summary['validation']['prediction_records']}
        counts = Counter()
        for nid, d in rows['D_observed_root'].items():
            e = rows['E_observed_parent'][nid]
            assert d['label'] == e['label']
            dc, ec = d['prediction'] == d['label'], e['prediction'] == e['label']
            outcome = 'parent_helps' if ec and not dc else 'parent_hurts' if dc and not ec else 'both_correct' if dc else 'both_wrong'
            counts[outcome] += 1
            cases.setdefault(nid, {'news_id': nid, 'label': d['label'], 'seeds': {}})['seeds'][seed] = {
                'root_prediction': d['prediction'], 'parent_prediction': e['prediction'], 'outcome': outcome}
        result['prediction_pairs'][seed] = dict(counts)
    for split in ['train', 'validate']:
        graphs = read(SOURCE / f'observed_only_{split}.jsonl')
        counts = Counter(); depth_counts = Counter(); unique_parents = Counter()
        for graph in graphs:
            parents = {e['target']: e['source'] for e in graph['edges']}
            real = [n['node_index'] for n in graph['graph_nodes'] if n['node_type'] == 'real_comment']
            used = []
            for node in graph['generated_nodes']:
                p = node['parent_node_index']; used.append(p)
                depth = 0; cursor = p; seen = set()
                while cursor:
                    assert cursor not in seen
                    seen.add(cursor); cursor = parents[cursor]; depth += 1
                depth_counts[depth] += 1
                counts['generated_comments'] += 1
                counts['parent_is_root'] += int(p == 0)
                counts['parent_is_last_visible_real'] += int(bool(real) and p == max(real))
                counts['parent_is_first_visible_real'] += int(bool(real) and p == min(real))
            unique_parents[len(set(used))] += 1
        result['graph_statistics'][split] = {'counts': dict(counts), 'real_parent_depth': dict(depth_counts),
                                           'distinct_parents_per_news': dict(unique_parents), 'news': len(graphs)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'previous_round_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT / 'previous_round_cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in cases.values()), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    analyze()
