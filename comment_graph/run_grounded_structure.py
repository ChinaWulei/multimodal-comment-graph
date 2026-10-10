"""Paired content x placement experiment. No test tuning; no quality weak-label gate.

Stages are explicit and resumable. Each child process has a persistent log.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, stdev
import subprocess
import sys
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'comment_generator/partial_observation/data/official_subset'
SCRIPTS = ROOT / 'comment_generator/partial_observation/scripts'
GEN_CONFIG = ROOT / 'comment_generator/partial_observation/config/generator_structure_official_expanded.yaml'
GRAPH_CONFIG = ROOT / 'comment_graph/config_fakeddit_structure_expanded_generateddepth.yaml'
DEFAULT_OUTPUT = ROOT / 'comment_graph/outputs/grounded_structure_20261010'
SEEDS = [42, 13, 24]
ARMS = {
    'A_real': ('iterative', 'real_only', 'observed_tree'),
    'B_iterative_root': ('iterative', 'real_generated', 'generated_root'),
    'C_iterative_parent': ('iterative', 'real_generated', 'observed_tree'),
    'D_observed_root': ('observed_only', 'real_generated', 'generated_root'),
    'E_observed_parent': ('observed_only', 'real_generated', 'observed_tree'),
}


@contextmanager
def exclusive_run(out):
    """Windows releases this lock even after an interrupted/killed process."""
    import msvcrt
    with (out/'pipeline.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as error:
            raise RuntimeError('Another pipeline is already using this output directory') from error
        try: yield
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


def freeze_execution(out):
    files=[Path(__file__),GEN_CONFIG,GRAPH_CONFIG,
           *[ROOT/'comment_graph'/n for n in ['train.py','model.py','common.py','dataset.py',
              'fakeddit_structure_dataset.py','typed_graph.py','dynamic_graph.py']],
           *[SCRIPTS/n for n in ['generate_structured_comments.py','structure_utils.py','common.py',
                                'validate_generated_structured_graphs.py']]]
    record={'code_sha256':{str(p):sha(p) for p in files}}
    path=out/'execution_lock.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != record:
        raise ValueError('Experiment code/config changed after execution freeze; use a new output directory')
    save(path,record)


def qualifies_gain(deltas):
    return len(deltas)==3 and mean(deltas)>=.01 and sum(d>0 for d in deltas)>=2


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in value) if jsonl
            else json.dumps(value, ensure_ascii=False, indent=2))
    if path.is_file() and path.read_text(encoding='utf8') == text:
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf8')
    temporary.replace(path)


def child(out, name, arguments):
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', *map(str, arguments)]
    save(out / 'status.json', {'state': 'running', 'step': name, 'command': command})
    print('RUN', name, flush=True)
    with (logs / (name + '.log')).open('a', encoding='utf8') as log:
        log.write('\nCOMMAND ' + json.dumps(command) + '\n'); log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f'{name} exited {code}; see {logs / (name + ".log")}')
    print('DONE', name, flush=True)


def contexts(graphs):
    result = []
    for graph in graphs:
        if graph['label_used_for_generation'] is not False:
            raise ValueError('Label-conditioned graph is not allowed')
        real = [n for n in graph['graph_nodes'] if n['node_type'] == 'real_comment']
        assert [n['node_index'] for n in real] == list(range(1, len(real)+1))
        parents = {e['target']: e['source'] for e in graph['edges'] if e['edge_type'] == 0}
        result.append({'sample_id': graph['sample_id'], 'metadata': {
            'news_id': graph['news_id'], 'official_split': graph['official_split'],
            'observed_node_comment_ids': {str(n['node_index']): n['source_comment_id'] for n in real}},
            'model_input': {'image_path': graph['image_path'], 'news_text': graph['graph_nodes'][0]['content'],
                            'observed_nodes': [{'node_index': n['node_index'],
                                                'parent_node_index': parents[n['node_index']],
                                                'content': n['content']} for n in real]}})
    return result


def prepare(out):
    from PIL import Image
    source = {s: DATA / f'generated_{s}_all_valid_images_observed_boundary.jsonl' for s in ['train','validate']}
    graphs = {s: read(p) for s,p in source.items()}
    assert [len(graphs[s]) for s in ['train','validate']] == [292, 77]
    ids = {s: {r['news_id'] for r in rows} for s,rows in graphs.items()}
    assert len(ids['train']) == 292 and len(ids['validate']) == 77 and not ids['train'] & ids['validate']
    selection = DATA / 'structure_validate_expanded.jsonl'
    generator_validation = {r['metadata']['news_id'] for r in read(selection)}
    primary = sorted(ids['validate'] - generator_validation)
    assert len(primary) == 44, 'Review unexpected generator-disjoint validation count'
    queues = [ROOT / 'comment_graph/reports/generated_quality_annotation_train60.jsonl',
              ROOT / 'comment_graph/reports/quality_expansion_20261008/fit.jsonl',
              ROOT / 'comment_graph/reports/quality_expansion_20261008/reserved_audit.jsonl']
    reviewed = {r['news_id'] for p in queues for r in read(p)}
    eligible = sorted(ids['train'] - reviewed)
    assert len(eligible) == 112
    smoke = sorted(random.Random(42).sample(eligible, 8))
    adapters = {r['adapter_dir'] for rows in graphs.values() for r in rows}
    assert len(adapters) == 1
    adapter = Path(next(iter(adapters)))
    assert (adapter / 'adapter_model.safetensors').is_file()
    records = {s: contexts(rows) for s,rows in graphs.items()}
    for rows in records.values():
        for row in rows:
            with Image.open(row['model_input']['image_path']) as im:
                im.load()
    protocol = {'version': 1, 'dataset': 'Fakeddit', 'train_news': 292, 'validation_news': 77,
                'primary_validation_news': primary, 'smoke_news': smoke, 'smoke_seed': 42,
                'smoke_excluded_reviewed_news': sorted(reviewed), 'smoke_eligible_news': len(eligible),
                'generation_seed': 42, 'classifier_seeds': SEEDS, 'arms': ARMS,
                'generator_adapter': str(adapter), 'generator_config_sha256': sha(GEN_CONFIG),
                'classifier_config_sha256': sha(GRAPH_CONFIG),
                'input_sources': {s: {'path': str(p), 'sha256': sha(p)} for s,p in source.items()},
                'generator_selection_sft_sha256': sha(selection),
                'quality_queues_sha256': {str(p): sha(p) for p in queues},
                'training': {'epochs':20,'patience':5,'max_real_comments':16,'max_generated_comments':5},
                'test_gate': {'comparisons':['E_observed_parent>A_real','E_observed_parent>D_observed_root'],
                              'minimum_mean_f1_gain':0.01,'minimum_positive_seeds':2},
                'quality_filter_enabled': False, 'news_labels_in_generation': False,
                'validation_is_new_blind_set': False}
    path = out / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != json.loads(json.dumps(protocol)):
        raise ValueError('Existing protocol changed; use a new output directory')
    save(path, protocol)
    for split, rows in records.items():
        save(out / f'contexts_{split}.jsonl', rows, True)
    save(out / 'contexts_smoke.jsonl', [r for r in records['train'] if r['metadata']['news_id'] in smoke], True)
    save(out / 'contexts_validate_primary.jsonl', [r for r in records['validate'] if r['metadata']['news_id'] in primary], True)
    print(json.dumps({'train':292,'validation':77,'primary_validation':len(primary),'smoke':smoke}), flush=True)
    return protocol


def generate(out, protocol, split, policy, smoke=False):
    suffix = 'smoke' if smoke else split
    path = out / f'{policy}_{suffix}.jsonl'
    args = [SCRIPTS / 'generate_structured_comments.py', '--config', GEN_CONFIG,
            '--adapter-dir', protocol['generator_adapter'], '--input-jsonl', out / f'contexts_{suffix}.jsonl',
            '--output-jsonl', path, '--limit', '0', '--one-per-news', '--official-split', split,
            '--nodes-per-graph','5','--depth-reference','observed_boundary','--context-policy',policy,
            '--local-files-only','--resume']
    child(out, f'generate_{policy}_{suffix}', args)
    child(out, f'validate_{policy}_{suffix}', [SCRIPTS/'validate_generated_structured_graphs.py',
          '--input-jsonl',path,'--expected-split',split,'--expected-context-jsonl',out/f'contexts_{suffix}.jsonl'])
    return path


def pair_audit(left_path, right_path):
    left = {r['news_id']:r for r in read(left_path)}
    right = {r['news_id']:r for r in read(right_path)}
    assert set(left) == set(right)
    stats = {}
    for name, rows in [('iterative',left),('observed_only',right)]:
        failures, parents, counts = Counter(), Counter(), Counter()
        for row in rows.values():
            assert row['label_used_for_generation'] is False
            node_map = {n['node_index']:n for n in row['graph_nodes']}
            counts[len(row['generated_nodes'])] += 1
            for node in row['generated_nodes']:
                parents[node_map[node['parent_node_index']]['node_type']] += 1
            failures.update(r['reason'] for r in row['generation_failures'])
            strings = [r['content'].strip().casefold() for r in row['generated_nodes']]
            assert len(strings) == len(set(strings))
        stats[name] = {'news':len(rows), 'generated_counts':dict(counts),'parent_types':dict(parents),
                       'failed_attempts':dict(failures),'generated_nodes':sum(k*v for k,v in counts.items())}
    for news in left:
        a,b = left[news],right[news]
        for key in ['sample_id','image_path','official_split']:
            assert a[key] == b[key]
        assert [n for n in a['graph_nodes'] if n['node_type'] != 'generated_comment'] == [n for n in b['graph_nodes'] if n['node_type'] != 'generated_comment']
        assert [e for e in a['edges'] if e['edge_type']==0] == [e for e in b['edges'] if e['edge_type']==0]
    assert stats['observed_only']['parent_types'].get('generated_comment',0)==0
    return {'same_news_images_real_nodes_and_edges':True,'arms':stats}


def graph_smoke(out):
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform, seed_everything
    from comment_graph.model import CommentGraphMFCL
    seed_everything(42)
    config, path = load_config(GRAPH_CONFIG)
    config.data.max_real_comments = 16; config.data.max_generated_comments = 5
    config.graph.message_passing = 'typed_rgcn'; config.graph.pooling_mode = 'mean'
    config.graph.generated_reliability.mode = 'none'; config.decision_mode = 'concat'
    tokenizer, transform = make_tokenizer_and_transform(config)
    device = torch.device('cuda')
    report = pair_audit(out/'iterative_smoke.jsonl', out/'observed_only_smoke.jsonl')
    batches = {}; reference = None; paired_inputs = {}
    model = CommentGraphMFCL(config,'real_generated',graph_structure='observed_tree',pooling_mode='mean',decision_mode='concat').to(device)
    for arm,(policy,mode,structure) in ARMS.items():
        config.data.generated_comments.train = str(out/f'{policy}_smoke.jsonl')
        config.graph.structure_mode = structure
        ids = {r['news_id'] for r in read(config.data.generated_comments.train)}
        dataset = make_dataset(config,path,'train',mode,tokenizer,transform,allowed_news_ids=ids)
        items = [dataset[i] for i in range(len(dataset))]
        paired_inputs[arm] = items
        signature = [(x.news_id, x.node_input_ids[x.node_type != 2], x.edge_index[:,x.edge_type==0]) for x in items]
        if reference is None: reference = signature
        else:
            assert len(signature)==len(reference)==8
            for a,b in zip(reference,signature):
                assert a[0]==b[0] and torch.equal(a[1],b[1]) and torch.equal(a[2],b[2])
        batch = next(iter(DataLoader(items,batch_size=8,shuffle=False))).to(device)
        model.mode = mode; model.graph_structure = structure
        model.train(); model.zero_grad(set_to_none=True)
        result = model(batch)
        loss = F.cross_entropy(result['logits'],batch.y.view(-1)); loss.backward()
        grads = [p.grad for p in model.typed_graph_encoder.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum()>0 for g in grads)
        assert result['logits'].shape==(8,3) and torch.isfinite(loss)
        for item in items:
            assert item.edge_index.shape[1] == 2*(item.num_nodes-1)
        batches[arm] = {'loss':float(loss.detach()), 'gnn_nonzero_finite_grad':True,
                        'nodes':[int(x.num_nodes) for x in items],
                        'directed_edges':[int(x.edge_index.shape[1]) for x in items],
                        'h_mm':list(result['h_mm'].shape), 'h_graph':list(result['h_graph'].shape)}
    for root_arm, parent_arm in [('B_iterative_root','C_iterative_parent'),('D_observed_root','E_observed_parent')]:
        for root, parent in zip(paired_inputs[root_arm], paired_inputs[parent_arm]):
            assert torch.equal(root.node_input_ids,parent.node_input_ids)
            assert torch.equal(root.node_type,parent.node_type)
            assert torch.equal(root.pixel_values,parent.pixel_values)
            synthetic = root.edge_index[:,root.edge_type==1]
            assert bool(((synthetic[0]==0)|(synthetic[1]==0)).all())
    report['root_vs_parent_same_text_image_and_node_types'] = True
    report['gnn'] = batches
    report['passed'] = True
    save(out/'smoke_report.json',report)
    print(json.dumps(report,indent=2),flush=True)


def smoke(out, protocol):
    child(out,'unit_tests',[ROOT/'comment_graph/smoke_grounded_generation.py'])
    for policy in ['iterative','observed_only']:
        generate(out,protocol,'train',policy,True)
        # Run the completed file again: a no-op resume must not change a byte.
        path = out/f'{policy}_smoke.jsonl'; before = sha(path)
        generate(out,protocol,'train',policy,True)
        assert before==sha(path)
    child(out,'gnn_smoke',[Path(__file__),'--stage','gnn-smoke','--output-dir',out])
    seconds = {}; peaks = {}
    for policy in ['iterative','observed_only']:
        meta=json.loads((out/f'{policy}_smoke.run.json').read_text(encoding='utf8'))
        seconds[policy]=meta['session_seconds']/meta['session_graphs']
        peaks[policy]=meta['peak_gpu_allocated_gib']
    save(out/'resource_estimate.json',{'seconds_per_news_including_load':seconds,'peak_gpu_gib':peaks,
        'full_generation_estimated_minutes':369*sum(seconds.values())/60,
        'note':'Generation only; CPU/GPU load and text length affect runtime. Classifier training additional.'})


def full_generation(out, protocol):
    if not json.loads((out/'smoke_report.json').read_text(encoding='utf8'))['passed']:
        raise ValueError('Smoke gate not passed')
    for split in ['train','validate']:
        for policy in ['iterative','observed_only']:
            generate(out,protocol,split,policy)
    audits={s:pair_audit(out/f'iterative_{s}.jsonl',out/f'observed_only_{s}.jsonl') for s in ['train','validate']}
    save(out/'full_pair_audit.json',audits)
    primary=set(protocol['primary_validation_news'])
    for policy in ['iterative','observed_only']:
        save(out/f'{policy}_validate_primary.jsonl',[r for r in read(out/f'{policy}_validate.jsonl') if r['news_id'] in primary],True)


def train(out, protocol):
    assert (out/'full_pair_audit.json').is_file()
    for seed in SEEDS:
        for arm,(policy,mode,structure) in ARMS.items():
            name=f'{arm}_seed{seed}'
            summary=out/'classification'/f'{name}.json'
            train_graph=out/f'{policy}_train.jsonl'
            val_graph=out/f'{policy}_validate_primary.jsonl'
            if summary.exists():
                existing=json.loads(summary.read_text(encoding='utf8'))
                assert existing['seed']==seed and existing['mode']==mode and existing['graph_structure']==structure
                assert existing['train_size']==292 and existing['validate_size']==44
                assert existing['generated_sources']['train']['sha256']==sha(train_graph)
                assert existing['generated_sources']['validate']['sha256']==sha(val_graph)
                assert Path(existing['checkpoint']).is_file()
                assert existing['message_passing']=='typed_rgcn' and existing['pooling_mode']=='mean'
                assert existing['decision_mode']=='concat' and existing['generated_reliability']=='none'
                assert existing['max_real_comments']==16 and existing['max_generated_comments']==5
                print('VERIFIED',name,flush=True)
            else:
                child(out,name,[ROOT/'comment_graph/train.py','--config',GRAPH_CONFIG,'--mode',mode,
                    '--graph-structure',structure,'--message-passing','typed_rgcn','--pooling-mode','mean',
                    '--decision-mode','concat','--generated-reliability','none','--mm-fusion','mfcl',
                    '--evidence-mode','none','--generated-train-jsonl',train_graph,
                    '--generated-validate-jsonl',val_graph,'--max-real-comments','16','--max-generated-comments','5',
                    '--epochs','20','--patience','5','--seed',seed,'--allow-formal-training',
                    '--checkpoint-dir',out/'checkpoints'/name,'--summary-json',summary])
            auxiliary=out/'classification'/f'{name}_validation77.json'
            if not auxiliary.exists():
                checkpoint=json.loads(summary.read_text(encoding='utf8'))['checkpoint']
                child(out,name+'_validation77',[Path(__file__),'--stage','evaluate-validation',
                    '--output-dir',out,'--checkpoint',checkpoint,'--generated-jsonl',out/f'{policy}_validate.jsonl',
                    '--evaluation-output',auxiliary])
            frozen_aux=json.loads(auxiliary.read_text(encoding='utf8'))
            saved_run=json.loads(summary.read_text(encoding='utf8'))
            assert frozen_aux['checkpoint_sha256']==sha(saved_run['checkpoint'])
            assert frozen_aux['generated_sha256']==sha(out/f'{policy}_validate.jsonl')


def evaluate_validation(args):
    import torch
    from munch import DefaultMunch
    from comment_graph.common import make_dataset,make_tokenizer_and_transform,seed_everything
    from comment_graph.model import CommentGraphMFCL
    from comment_graph.train import evaluate,make_loader
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=DefaultMunch.fromDict(checkpoint['config'])
    config.data.generated_comments.validate=str(args.generated_jsonl.resolve())
    seed_everything(config.seed)
    tokenizer,transform=make_tokenizer_and_transform(config)
    ids={r['news_id'] for r in read(args.generated_jsonl)}
    dataset=make_dataset(config,GRAPH_CONFIG,'validate',checkpoint['mode'],tokenizer,transform,allowed_news_ids=ids)
    assert len(dataset)==77 and len(ids)==77
    model=CommentGraphMFCL(config,checkpoint['mode'],graph_structure=checkpoint['graph_structure'],
                           pooling_mode=checkpoint['pooling_mode'],decision_mode=checkpoint['decision_mode']).cuda()
    model.load_state_dict(checkpoint['model'],strict=True)
    result=evaluate(model,make_loader(dataset,config,False),torch.device('cuda'),bool(config.training.use_amp),None)
    save(args.evaluation_output,{'checkpoint_sha256':sha(args.checkpoint),'generated_sha256':sha(args.generated_jsonl),
         'validation':result,'news':77,'used_for_checkpoint_selection':False})


def summarize(out,protocol):
    runs={arm:[json.loads((out/'classification'/f'{arm}_seed{seed}.json').read_text(encoding='utf8')) for seed in SEEDS] for arm in ARMS}
    aggregate={}
    for arm,rows in runs.items():
        for row in rows:
            assert 'test' not in row and row['validate_size']==44
            assert {r['news_id'] for r in row['validation']['prediction_records']}==set(protocol['primary_validation_news'])
        aggregate[arm]={metric:{'mean':mean(r['validation'][metric] for r in rows),
                               'std':stdev(r['validation'][metric] for r in rows),
                               'by_seed':{str(s):r['validation'][metric] for s,r in zip(SEEDS,rows)}}
                        for metric in ['accuracy','macro_f1','weighted_f1']}
    comparisons={}
    for base in ['A_real','D_observed_root','B_iterative_root','C_iterative_parent']:
        delta=[e['validation']['macro_f1']-b['validation']['macro_f1'] for e,b in zip(runs['E_observed_parent'],runs[base])]
        comparisons[base]={'mean_gain':mean(delta),'gains_by_seed':dict(zip(map(str,SEEDS),delta)),
                           'positive_seeds':sum(d>0 for d in delta),
                           'passes':qualifies_gain(delta)}
    passed=all(comparisons[b]['passes'] for b in ['A_real','D_observed_root'])
    auxiliary={}
    for arm in ARMS:
        rows=[json.loads((out/'classification'/f'{arm}_seed{seed}_validation77.json').read_text(encoding='utf8')) for seed in SEEDS]
        auxiliary[arm]={m:{'mean':mean(r['validation'][m] for r in rows),'std':stdev(r['validation'][m] for r in rows)}
                        for m in ['accuracy','macro_f1','weighted_f1']}
    result={'aggregate':aggregate,'auxiliary_validation77':auxiliary,'E_minus_baseline':comparisons,'test_gate_passed':passed,
            'test_executed':False,'selection_metric':'primary44_validation_macro_f1',
            'warning':'Exploratory reused validation, not an untouched test. No guaranteed improvement.'}
    save(out/'comparison.json',result)
    if passed:
        save(out/'frozen_selection.json',{'protocol_sha256':sha(out/'protocol.json'),'comparison_sha256':sha(out/'comparison.json'),
            'checkpoints':{a:{str(s):{'path':r['checkpoint'],'sha256':sha(r['checkpoint'])} for s,r in zip(SEEDS,rows)} for a,rows in runs.items()},
            'test_gate':protocol['test_gate']})
    print(json.dumps(result,indent=2),flush=True)


def test_frozen(out,protocol):
    """Optional final stage, strictly unavailable unless the validation gate passed."""
    comparison=json.loads((out/'comparison.json').read_text(encoding='utf8'))
    if comparison['test_gate_passed'] is not True:
        raise RuntimeError('Validation improvement gate failed. Test evaluation is disabled for this experiment.')
    lock=json.loads((out/'frozen_selection.json').read_text(encoding='utf8'))
    assert lock['protocol_sha256']==sha(out/'protocol.json')
    assert lock['comparison_sha256']==sha(out/'comparison.json')
    for arm in ARMS:
        for seed in SEEDS:
            record=lock['checkpoints'][arm][str(seed)]
            assert sha(record['path'])==record['sha256']
    # Fixed existing cohort, never selected according to new test performance.
    original=DATA/'generated_test_structured_graphs_expanded_generateddepth.jsonl'
    graphs=read(original)
    assert graphs and all(r['official_split']=='test' for r in graphs)
    used={r['metadata']['news_id'] for s in ['train','validate'] for r in read(out/f'contexts_{s}.jsonl')}
    assert not {r['news_id'] for r in graphs}&used
    contract={'source_path':str(original),'source_sha256':sha(original),
              'news_ids':sorted(r['news_id'] for r in graphs),
              'selection_sha256':sha(out/'frozen_selection.json'),
              'previously_examined_test_cohort':True,'test_is_new_blind_set':False}
    test_contract=out/'test_protocol.json'
    if test_contract.exists(): assert json.loads(test_contract.read_text(encoding='utf8'))==contract
    save(test_contract,contract)
    save(out/'contexts_test.jsonl',contexts(graphs),True)
    for policy in ['iterative','observed_only']: generate(out,protocol,'test',policy)
    save(out/'test_pair_audit.json',pair_audit(out/'iterative_test.jsonl',out/'observed_only_test.jsonl'))
    all_results={}
    for arm,(policy,mode,structure) in ARMS.items():
        all_results[arm]=[]
        for seed in SEEDS:
            record=lock['checkpoints'][arm][str(seed)]
            output=out/'test'/f'{arm}_seed{seed}.json'
            if not output.exists():
                child(out,f'test_{arm}_{seed}',[ROOT/'comment_graph/evaluate_frozen_checkpoint.py',
                    '--config',GRAPH_CONFIG,'--checkpoint',record['path'],'--output-json',output,
                    '--generated-test-jsonl',out/f'{policy}_test.jsonl',
                    '--expected-context-jsonl',out/'contexts_test.jsonl'])
            result=json.loads(output.read_text(encoding='utf8'))
            assert result['checkpoint_sha256']==record['sha256']
            assert result['generated_test_sha256']==sha(out/f'{policy}_test.jsonl')
            assert sorted(result['test_news_ids'])==contract['news_ids']
            all_results[arm].append(result)
    aggregate={a:{m:{'mean':mean(r['test'][m] for r in rows),'std':stdev(r['test'][m] for r in rows)}
                  for m in ['accuracy','macro_f1','weighted_f1']} for a,rows in all_results.items()}
    save(out/'test_comparison.json',{'aggregate':aggregate,'protocol':contract,'parameter_changes_after_test':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','smoke','gnn-smoke','generate','train','summarize','full','test','evaluate-validation'],default='prepare')
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--generated-jsonl',type=Path)
    p.add_argument('--evaluation-output',type=Path)
    args=p.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.stage=='evaluate-validation': evaluate_validation(args); return
    if args.stage=='gnn-smoke': graph_smoke(out); return
    started=time.perf_counter()
    with exclusive_run(out):
        try:
            protocol=prepare(out)
            if args.stage=='smoke': smoke(out,protocol)
            if args.stage in ['generate','train','summarize','full','test']: freeze_execution(out)
            if args.stage in ['generate','full']: full_generation(out,protocol)
            if args.stage in ['train','full']: train(out,protocol)
            if args.stage in ['summarize','full']: summarize(out,protocol)
            if args.stage=='test': test_frozen(out,protocol)
            save(out/'status.json',{'state':'completed','stage':args.stage,'session_seconds':time.perf_counter()-started})
        except BaseException as error:
            save(out/'status.json',{'state':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                                  'stage':args.stage,'error':str(error),'session_seconds':time.perf_counter()-started})
            raise


if __name__=='__main__': main()
