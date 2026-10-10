"""Confirm fixed-content diagnostics on news absent from generator SFT, train split only."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comment_graph.attachment_consistency import probe

ORIGINAL_OUT = probe.OUT
OUT = probe.HERE / 'outputs/fixed_content_unseen_train24'


def prepare():
    from common import load_config
    config, config_path = load_config(probe.CONFIG)
    reference = json.loads((ORIGINAL_OUT / 'protocol.json').read_text(encoding='utf-8'))
    for path, expected in reference['sources'].items(): assert probe.sha(path) == expected, path
    frozen = json.loads((probe.SOURCE / 'protocol.json').read_text(encoding='utf-8'))
    data = probe.SCRIPTS.parent / 'data/official_subset'
    sft = probe.read(data / 'structure_train_expanded.jsonl')
    sft_ids = {r['metadata']['news_id'] for r in sft}
    exclusions = set(frozen['smoke_excluded_reviewed_news']) | set(frozen['smoke_news']) | sft_ids
    contexts_file = probe.SOURCE / 'contexts_train.jsonl'
    contexts = probe.read(contexts_file)
    eligible = [r for r in contexts if r['metadata']['news_id'] not in exclusions
                and len(r['model_input']['observed_nodes']) >= 4]
    eligible.sort(key=lambda r: hashlib.sha256(f'20261010:unseen:{r["metadata"]["news_id"]}'.encode()).hexdigest())
    if len(eligible) < 24: raise RuntimeError(f'Need 24 unseen eligible train news; found {len(eligible)}')
    selected = eligible[:24]
    graphs = {r['news_id']: r for r in probe.read(probe.SOURCE / 'observed_only_train.jsonl')}
    cases = []
    for record in selected:
        news_id = record['metadata']['news_id']; graph = graphs[news_id]
        assert record['metadata']['official_split'] == graph['official_split'] == 'train'
        assert graph['label_used_for_generation'] is False and news_id not in sft_ids
        real = [n for n in graph['graph_nodes'] if n['node_type'] == 'real_comment']
        parents = {e['target']: e['source'] for e in graph['edges'] if e['edge_type'] == 0}
        assert [{'node_index': n['node_index'], 'parent_node_index': parents[n['node_index']], 'content': n['content']}
                for n in real] == record['model_input']['observed_nodes']
        # First generated node by original node index, never selected for quality or parent outcome.
        target = min(graph['generated_nodes'], key=lambda n: n['node_index'])
        cases.append({'case_id': f'{news_id}:generated:{target["node_index"]}', 'news_id': news_id, 'kind': 'generated',
            'model_input': deepcopy(record['model_input']), 'target_content': target['content'],
            'original_predicted_parent': target['parent_node_index'], 'expected_parent': None})
        control = probe.real_leaf_case(record)
        control['news_seen_in_generator_sft'] = False
        control['target_seen_as_generator_sft_target'] = False
        cases.append(control)
    raw = {r['news_id']: r for r in probe.read(data / 'raw_comment_trees.jsonl') if r['news_id'] in {c['news_id'] for c in cases}}
    def flatten(nodes):
        result = {}
        for n in nodes:
            result[n['comment_id']] = n
            result.update(flatten(n.get('children', [])))
        return result
    for record in selected:
        control = next(c for c in cases if c['news_id'] == record['metadata']['news_id'] and c['kind'] == 'real_control')
        node = flatten(raw[control['news_id']]['roots'])[control['target_source_comment_id']]
        original_target = next(n for n in record['model_input']['observed_nodes'] if n['node_index'] == control['target_original_node_index'])
        expected_id = record['metadata']['observed_node_comment_ids'].get(str(original_target['parent_node_index']), control['news_id'])
        assert node['parent_id'].removeprefix('t1_').removeprefix('t3_') == expected_id
        assert ' '.join(node['body'].split()) == ' '.join(control['target_content'].split())
    sources = dict(reference['sources'])
    files = [Path(__file__), ORIGINAL_OUT / 'protocol.json', contexts_file,
             *[Path(r['model_input']['image_path']) for r in selected]]
    sources.update({str(p): probe.sha(p) for p in files})
    protocol = {**reference, 'sources': sources, 'cases': len(cases), 'calls': len(cases)*6,
        'news_ids': [r['metadata']['news_id'] for r in selected], 'eligible_news': len(eligible),
        'selection_rule': 'Fixed SHA256 ordering seed 20261010; exclude generator SFT news and earlier reviewed/smoke news; no label or prediction selection.',
        'generator_sft_news_overlap': 0, 'relation_targets_verified_against_raw': True,
        'limitations': ['This is an additional training-split diagnostic, not classifier validation or a final test.',
            'Selected news were absent from this adapter SFT but base-model pretraining exposure cannot be ruled out.',
            'Parent-only selection differs from original joint JSON generation; no blanket claims about joint-task accuracy.',
            'Only 24 true parent targets; candidate contexts are incomplete, not the full Reddit discussion.',
            'Generated comments have no ground-truth parents; stability does not establish correctness.']}
    for path, value in [(OUT / 'protocol.json', protocol), (OUT / 'cases.json', cases)]:
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value: raise ValueError(f'Frozen output changed: {path}')
        probe.save(path, value)
    return config, config_path, cases, protocol


if __name__ == '__main__':
    probe.OUT = OUT
    probe.prepare = prepare
    probe.main()
