"""Identity-preserving index/order augmentation. Preview and bounded training smoke only."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from comment_graph.attachment_consistency import probe
from structure_utils import load_structure_template, STRUCTURE_SYSTEM_PROMPT

OUT = probe.HERE / 'outputs/augmented_sft_v1'
DATA = probe.SCRIPTS.parent / 'data/official_subset'


def canonical_validate(model_input):
    """Validate graph identities, not an artificial numeric ordering of nodes."""
    if set(model_input) != {'image_path', 'news_text', 'observed_nodes'}: raise ValueError('Input field leakage')
    nodes = model_input['observed_nodes']; mapping = {n['node_index']: n for n in nodes}
    if len(mapping) != len(nodes) or set(mapping) != set(range(1, len(nodes)+1)): raise ValueError('IDs not a bijection')
    for node in nodes:
        if set(node) != {'node_index', 'parent_node_index', 'content'}: raise ValueError('Unexpected node fields')
        if not node['content'].strip(): raise ValueError('Empty node')
        current = node['node_index']; seen = set()
        while current:
            if current in seen or current not in mapping: raise ValueError('Cycle or missing ancestor')
            seen.add(current); current = mapping[current]['parent_node_index']


def permute_record(record):
    result = deepcopy(record); source = record['model_input']; nodes = source['observed_nodes']
    canonical_validate(source)
    rng = random.Random(int.from_bytes(hashlib.sha256(f'20261010:augment:{record["sample_id"]}'.encode()).digest()[:8], 'big'))
    old_ids = [n['node_index'] for n in nodes]; new_ids = list(old_ids); rng.shuffle(new_ids)
    mapping = {0: 0, **dict(zip(old_ids, new_ids))}
    transformed = [{'node_index': mapping[n['node_index']], 'parent_node_index': mapping[n['parent_node_index']],
                    'content': n['content']} for n in nodes]
    rng.shuffle(transformed)
    result['model_input']['observed_nodes'] = transformed
    result['target']['parent_node_index'] = mapping[record['target']['parent_node_index']]
    result['target_json'] = json.dumps(result['target'], ensure_ascii=False, separators=(',', ':'))
    result['augmentation'] = {'old_to_new_ids': mapping, 'display_order': [n['node_index'] for n in transformed],
                              'root_context_added': record.get('root_context_added', False)}
    canonical_validate(result['model_input'])
    inverse = {v: k for k, v in mapping.items()}
    restored = {inverse[n['node_index']]: {'node_index': inverse[n['node_index']],
        'parent_node_index': inverse[n['parent_node_index']], 'content': n['content']} for n in transformed}
    assert restored == {n['node_index']: n for n in nodes}
    assert inverse[result['target']['parent_node_index']] == record['target']['parent_node_index']
    assert result['target']['content'] == record['target']['content']
    return result


def add_root_context(record, tree):
    if record['target']['parent_node_index'] != 0: return deepcopy(record)
    assert not record['model_input']['observed_nodes']
    # Other genuine top-level branches only. Exclude the target's entire branch, not just its text.
    target_id = record['metadata']['target_comment_id']; news_id = record['metadata']['news_id']
    candidates = []
    def visit(nodes):
        for node in nodes:
            candidates.append(node); visit(node.get('children', []))
    visit([r for r in tree['roots'] if r['comment_id'] != target_id])
    chosen, mapping = [], {news_id: 0}
    for node in candidates:
        parent = node['parent_id'].removeprefix('t1_').removeprefix('t3_')
        if parent not in mapping or not node['body'].strip() or len(node['body']) > 400: continue
        if node['body'] == record['target']['content']: continue
        idx = len(chosen)+1; mapping[node['comment_id']] = idx
        chosen.append({'node_index': idx, 'parent_node_index': mapping[parent], 'content': node['body']})
        if len(chosen) == 4: break
    result = deepcopy(record); result['model_input']['observed_nodes'] = chosen
    result['metadata']['observed_node_comment_ids'] = {str(i): raw for raw, i in mapping.items() if i > 0}
    result['root_context_added'] = bool(chosen)
    return result


def prepare():
    source = DATA / 'structure_train_expanded.jsonl'; raw_path = DATA / 'raw_comment_trees.jsonl'
    original = probe.read(source)
    trees = {r['news_id']: r for r in probe.read(raw_path) if r['split'] == 'train'}
    augmented = []; counts = Counter()
    for record in original:
        assert record['metadata']['official_split'] == 'train'
        changed = permute_record(add_root_context(record, trees[record['metadata']['news_id']]))
        n = len(changed['model_input']['observed_nodes']); parent = changed['target']['parent_node_index']
        counts['records'] += 1; counts['nonempty'] += n > 0; counts['root_parent'] += parent == 0
        counts['root_with_nonempty'] += parent == 0 and n > 0
        counts['highest_index_parent_nonempty'] += n > 0 and parent == n
        counts['last_displayed_parent_nonempty'] += n > 0 and parent == changed['model_input']['observed_nodes'][-1]['node_index']
        augmented.append(changed)
    paths = [Path(__file__), probe.HERE / 'test_augmentation.py', source, raw_path,
             probe.CONFIG, probe.SCRIPTS.parent / 'prompts/structured_comment_generation.txt']
    contract = {'seed': 20261010, 'sources': {str(p): probe.sha(p) for p in paths},
        'changes': ['Random bijective node IDs; remap every real edge and target parent.', 'Independently shuffle display order.',
                    'For original root targets, add up to four real nodes from other branches of the same news, when available.'],
        'unchanged': ['Original training news IDs and target comment content.', 'True raw reply identity.', 'No veracity label in prompt.',
                      'Validation/test not used for preparation.'],
        'smoke_only': True, 'counts': dict(counts), 'news': len({r['metadata']['news_id'] for r in augmented})}
    # Normalize integer mapping keys through JSON before comparing resumed outputs.
    for path, value in [(OUT / 'protocol.json', contract), (OUT / 'records.json', augmented)]:
        value = json.loads(json.dumps(value))
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value: raise ValueError('Frozen augmentation changed')
        probe.save(path, value)
    print(json.dumps(contract, indent=2))
    return augmented


def messages(record, template, target=False):
    canonical_validate(record['model_input'])
    source = record['model_input']
    text = template.format(news_text=source['news_text'], observed_nodes=json.dumps(source['observed_nodes'], ensure_ascii=False, separators=(',', ':')))
    result = [{'role': 'system', 'content': [{'type': 'text', 'text': STRUCTURE_SYSTEM_PROMPT}]},
              {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
    if target: result.append({'role': 'assistant', 'content': [{'type': 'text', 'text': record['target_json']}]})
    return result


def smoke(records):
    import torch
    from PIL import Image, ImageOps
    from peft import prepare_model_for_kbit_training
    from common import load_config
    from generate_structured_comments import load_adapter
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    output = OUT / 'smoke_training.json'
    if output.exists(): raise FileExistsError('Smoke already recorded; preserve it rather than overwriting')
    config, path = load_config(probe.CONFIG)
    protocol = json.loads((probe.OUT / 'protocol.json').read_text(encoding='utf-8'))
    adapter = Path(protocol['adapter_dir'])
    adapter_hash = probe.sha(adapter / 'adapter_model.safetensors')
    torch.manual_seed(42); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    model, processor = load_adapter(config, path, adapter, True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={'use_reentrant': False})
    for name, parameter in model.named_parameters(): parameter.requires_grad_('lora_' in name)
    model.config.use_cache = False; model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    assert params
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    template = load_structure_template(probe.SCRIPTS.parent / 'prompts/structured_comment_generation.txt')
    # Four independent real records, one true root with context plus three non-root examples.
    roots = [r for r in records if r['target']['parent_node_index'] == 0 and r['model_input']['observed_nodes']]
    others = [r for r in records if r['target']['parent_node_index'] != 0 and len(r['model_input']['observed_nodes']) >= 4]
    examples = [roots[0], *others[:3]]
    results = []
    for record in examples:
        with Image.open(record['model_input']['image_path']) as src: image = ImageOps.exif_transpose(src).convert('RGB').copy()
        prefix = processor.apply_chat_template(messages(record, template), tokenize=False, add_generation_prompt=True)
        complete = processor.apply_chat_template(messages(record, template, True), tokenize=False, add_generation_prompt=False)
        batch = processor(text=[complete], images=[image], return_tensors='pt', truncation=False).to(model.device)
        prefix_batch = processor(text=[prefix], images=[image], return_tensors='pt', truncation=False).to(model.device)
        prompt_length = prefix_batch['input_ids'].shape[1]
        assert batch['input_ids'].shape[1] <= config['training']['max_sequence_length'], 'No truncated target'
        assert torch.equal(batch['input_ids'][:, :prompt_length], prefix_batch['input_ids'])
        labels = batch['input_ids'].clone(); labels[:, :prompt_length] = -100
        assert (labels != -100).sum() > 0 and bool((labels[:, :prompt_length] == -100).all())
        optimizer.zero_grad(set_to_none=True)
        output_model = model(**batch, labels=labels); loss = output_model.loss
        assert torch.isfinite(loss); loss.backward()
        grads = [p.grad for p in params]
        assert all(g is not None and torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
        torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
        results.append({'sample_id': record['sample_id'], 'loss': float(loss.detach()), 'prompt_tokens': prompt_length,
            'supervised_tokens': int((labels != -100).sum()), 'root_with_context': record['target']['parent_node_index'] == 0,
            'finite_nonzero_lora_grad': True, 'image_loaded': True})
    # Deliberately do not save these four-step weights as a usable model.
    assert probe.sha(adapter / 'adapter_model.safetensors') == adapter_hash
    report = {'passed': True, 'steps': len(results), 'records': results, 'learning_rate': 1e-5,
        'source_adapter_unchanged': True, 'new_adapter_saved': False, 'formal_training': False,
        'session_seconds': time.perf_counter()-started, 'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
        'meaning': 'Plumbing/gradient smoke only. Different per-step samples; not evidence that loss or detection improves.'}
    probe.save(OUT / 'smoke_training.json', report); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'smoke'], default='prepare')
    args = parser.parse_args()
    records = prepare()
    if args.stage == 'smoke': smoke(records)
