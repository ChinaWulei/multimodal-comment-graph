"""Train-only paired display-order diagnostic; not a parent-accuracy benchmark."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'comment_generator/partial_observation/scripts'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SCRIPTS))
from comment_graph.connection_diagnostics.experiment import SOURCE, sha, save, read
from structure_utils import build_structure_messages, validate_structure_input

OUT = HERE / 'outputs/position_probe'
CONFIG = ROOT / 'comment_generator/partial_observation/config/generator_structure_official_expanded.yaml'
ORDERS = ['original', 'reverse', 'shuffle']


def node_order(nodes, news_id, variant):
    ids = [n['node_index'] for n in nodes]
    if variant == 'reverse': return list(reversed(ids))
    if variant == 'shuffle':
        rng = random.Random(int.from_bytes(hashlib.sha256(f'20261010:{news_id}'.encode()).digest()[:8], 'big'))
        result = list(ids); rng.shuffle(result)
        # Guarantee the last displayed node differs, so the positional hypothesis is testable.
        if len(result) > 1 and result[-1] == ids[-1]: result = result[1:] + result[:1]
        return result
    if variant != 'original': raise ValueError(variant)
    return ids


def messages_for(template, model_input, order):
    validate_structure_input(model_input)
    nodes = model_input['observed_nodes']
    mapping = {n['node_index']: n for n in nodes}
    if len(order) != len(mapping) or set(order) != set(mapping): raise ValueError('Order must be a permutation of stable node IDs')
    # Validate the canonical tree, then alter only serialization order, never IDs or edges.
    messages = build_structure_messages(template, model_input, allowed_parent_indices=[0, *sorted(mapping)])
    canonical = json.dumps(nodes, ensure_ascii=False, separators=(',', ':'))
    displayed = json.dumps([mapping[i] for i in order], ensure_ascii=False, separators=(',', ':'))
    old = template.format(news_text=model_input['news_text'].strip(), observed_nodes=canonical)
    new = template.format(news_text=model_input['news_text'].strip(), observed_nodes=displayed)
    text = messages[1]['content'][1]['text']
    assert text.startswith(old)
    messages[1]['content'][1]['text'] = new + text[len(old):]
    return messages


def prepare():
    from common import load_config, resolve_path, resolve_model_reference
    config, path = load_config(CONFIG)
    original = json.loads((SOURCE / 'protocol.json').read_text(encoding='utf-8'))
    records = read(SOURCE / 'contexts_smoke.jsonl')
    assert len(records) == 8
    assert {r['metadata']['news_id'] for r in records} == set(original['smoke_news'])
    assert not set(original['smoke_news']) & set(original['smoke_excluded_reviewed_news'])
    assert all(r['metadata']['official_split'] == 'train' for r in records)
    contract = json.loads((SOURCE / 'observed_only_train.run.json').read_text(encoding='utf-8'))['contract']
    adapter = Path(contract['adapter_dir'])
    for name, expected in contract['adapter_files'].items(): assert sha(adapter / name) == expected
    template = resolve_path(path, config['prompt']['template_path'])
    model = Path(resolve_model_reference(path, config['model']['name_or_path']))
    files = [Path(__file__), HERE / 'test_protocol.py', SOURCE / 'contexts_smoke.jsonl', SOURCE / 'protocol.json',
             SOURCE / 'observed_only_train.run.json', CONFIG, template,
             SCRIPTS / 'generate_structured_comments.py', SCRIPTS / 'structure_utils.py', SCRIPTS / 'common.py',
             *[p for p in model.iterdir() if p.is_file()],
             *[adapter / n for n in contract['adapter_files']],
             *[Path(r['model_input']['image_path']) for r in records]]
    protocol = {'news_ids': [r['metadata']['news_id'] for r in records], 'split': 'train', 'orders': ORDERS,
        'sources': {str(p): sha(p) for p in files}, 'adapter_dir': str(adapter),
        'decoding': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 160, 'repetition_penalty': config['generation']['repetition_penalty']},
        'calls': 24, 'seed': 42, 'label_used_for_generation': False,
        'controls': ['Same images, news text, real node IDs, parent IDs and contents.',
                     'Only displayed node-list order changes; allowed-parent list remains numerically sorted.',
                     'Greedy decoding avoids different random draws as a confound.', 'One joint parent+content response per condition; no retries or fake filling.'],
        'limitations': ['Eight training news only, not population inference.',
                       'Joint content changes can legitimately change the parent; this is sensitivity, not parent accuracy.',
                       'Noncanonical display order is outside original parent-first SFT format.',
                       'Original generation used stochastic sampling; greedy diagnostic rates are not directly comparable.']}
    output = OUT / 'protocol.json'
    if output.exists() and json.loads(output.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Probe contract changed; use a new output folder')
    save(output, protocol)
    return config, path, records, protocol


def aggregate(rows, protocol):
    by_key = {(r['news_id'], r['order']): r for r in rows}
    assert len(by_key) == len(rows)
    counts = {}
    for order in ORDERS:
        selected = [r for r in rows if r['order'] == order]
        valid = [r for r in selected if r['valid']]
        counts[order] = {'attempted': len(selected), 'valid_json_parent': len(valid),
            'last_displayed_parent': sum(r['parent_node_index'] == r['display_order'][-1] for r in valid),
            'highest_index_parent': sum(r['parent_node_index'] == max(r['display_order']) for r in valid),
            'root_parent': sum(r['parent_node_index'] == 0 for r in valid)}
    paired = {}
    for order in ORDERS[1:]:
        cases = []
        for news_id in protocol['news_ids']:
            a, b = by_key.get((news_id, 'original')), by_key.get((news_id, order))
            if a and b and a['valid'] and b['valid']:
                cases.append({'news_id': news_id, 'original_parent': a['parent_node_index'], 'changed_parent': b['parent_node_index'],
                    'parent_changed': a['parent_node_index'] != b['parent_node_index'],
                    'content_changed': a['content'] != b['content'],
                    'follows_last_in_both': a['parent_node_index'] == a['display_order'][-1] and b['parent_node_index'] == b['display_order'][-1]})
        paired[order] = {'valid_pairs': len(cases), 'parent_changed': sum(x['parent_changed'] for x in cases),
                         'content_changed': sum(x['content_changed'] for x in cases),
                         'follows_last_in_both': sum(x['follows_last_in_both'] for x in cases), 'cases': cases}
    result = {'complete': len(rows) == protocol['calls'], 'counts': counts, 'paired': paired,
              'test_used': False, 'limitations': protocol['limitations']}
    save(OUT / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'run', 'summarize'], default='prepare')
    args = parser.parse_args()
    from comment_graph.run_grounded_structure import exclusive_run
    OUT.mkdir(parents=True, exist_ok=True)
    with exclusive_run(OUT):
        config, path, records, protocol = prepare()
        output = OUT / 'responses.json'
        rows = json.loads(output.read_text(encoding='utf-8')) if output.exists() else []
        expected = {(n, o) for n in protocol['news_ids'] for o in ORDERS}
        assert len({(r['news_id'], r['order']) for r in rows}) == len(rows)
        for r in rows:
            assert (r['news_id'], r['order']) in expected and r['protocol_sha256'] == sha(OUT / 'protocol.json')
        if args.stage == 'prepare': return
        if args.stage == 'summarize' or len(rows) == protocol['calls']: aggregate(rows, protocol); return
        import torch
        from PIL import Image, ImageOps
        from generate_structured_comments import load_adapter
        from structure_utils import load_structure_template, parse_structure_output
        from common import resolve_path
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model, processor = load_adapter(config, path, Path(protocol['adapter_dir']), True)
        template = load_structure_template(resolve_path(path, config['prompt']['template_path']))
        done = {(r['news_id'], r['order']) for r in rows}
        for record in records:
            news_id = record['metadata']['news_id']; model_input = deepcopy(record['model_input'])
            with Image.open(model_input['image_path']) as source_image:
                image = ImageOps.exif_transpose(source_image).convert('RGB').copy()
            for order_name in ORDERS:
                if (news_id, order_name) in done: continue
                order = node_order(model_input['observed_nodes'], news_id, order_name)
                messages = messages_for(template, model_input, order)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors='pt', truncation=False).to(model.device)
                assert 'pixel_values' in inputs and inputs['pixel_values'].numel() > 0
                assert inputs['input_ids'].shape[1] <= int(config['training']['max_sequence_length']), 'No silent truncation'
                torch.manual_seed(42)
                with torch.inference_mode():
                    generated = model.generate(**inputs, **protocol['decoding'],
                        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
                raw = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
                row = {'news_id': news_id, 'order': order_name, 'display_order': order, 'news_text': model_input['news_text'],
                       'image_path': model_input['image_path'], 'observed_nodes': model_input['observed_nodes'],
                       'actual_image_loaded': True, 'input_tokens': inputs['input_ids'].shape[1],
                       'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'raw_output': raw,
                       'protocol_sha256': sha(OUT / 'protocol.json'), 'valid': False}
                try:
                    row.update(parse_structure_output(raw, len(order))); row['valid'] = True
                except ValueError as exc: row['error'] = str(exc)
                rows.append(row); save(output, rows)
                save(OUT / 'status.json', {'state': 'running', 'completed': len(rows), 'total': protocol['calls'],
                    'session_seconds': time.perf_counter() - started, 'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3})
                print(f"{len(rows)}/{protocol['calls']} news={news_id} order={order_name} parent={row.get('parent_node_index')} valid={row['valid']}", flush=True)
        aggregate(rows, protocol)
        save(OUT / 'status.json', {'state': 'completed', 'completed': len(rows), 'session_seconds': time.perf_counter() - started,
                                  'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3})


if __name__ == '__main__': main()
