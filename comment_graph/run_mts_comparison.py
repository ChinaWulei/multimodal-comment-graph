"""Finite, paired MFCL-vs-MFCL+MTS validation runs; never selects on test."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['weibo', 'checked'], default='weibo')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use positive epochs/patience and unique seeds')
    here = Path(__file__).resolve().parent
    config = here / ('config_weibo.yaml' if args.dataset == 'weibo' else 'config.yaml')
    destination = (args.output_dir or here / 'outputs' / f'mts_{args.dataset}_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    destination.mkdir(parents=True, exist_ok=False)
    plan = {'dataset': args.dataset, 'seeds': args.seeds, 'arms': ['mfcl', 'mts'],
            'epochs': args.epochs, 'patience': args.patience, 'test_allowed': False,
            'primary_metric': 'validation macro-F1', 'selection': 'strictly highest F1; earliest tie',
            'mts_hidden_dim': 768, 'mts_layers': 8, 'source_sha256': {}, 'runs': []}
    for path in [config, *here.glob('*.py')]:
        plan['source_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for seed in args.seeds:
        for fusion in ['mfcl', 'mts']:
            name = f'{fusion}_seed{seed}'
            result = destination / f'{name}.json'
            command = [sys.executable, '-u', str(here / 'train.py'), '--config', str(config),
                       '--mode', 'baseline' if args.dataset == 'weibo' else 'real_generated',
                       '--graph-structure', 'star' if args.dataset == 'weibo' else 'dynamic_cross_modal',
                       '--pooling-mode', 'mean', '--decision-mode', 'concat' if args.dataset == 'weibo' else 'source_isolated',
                       '--mm-fusion', fusion, '--correction-reference', 'none', '--robustness-mode', 'none',
                       '--sampling', 'shuffle', '--seed', str(seed), '--epochs', str(args.epochs),
                       '--patience', str(args.patience), '--checkpoint-dir', str(destination / name),
                       '--summary-json', str(result), '--allow-formal-training']
            plan['runs'].append({'name': name, 'seed': seed, 'fusion': fusion, 'command': command,
                                 'status': 'planned', 'result': str(result)})
    def save():
        (destination / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    save()
    print(f'EXPERIMENT DIRECTORY: {destination}', flush=True)
    for run in plan['runs']:
        run['status'] = 'running'
        save()
        print(f"START {run['name']}", flush=True)
        with (destination / (run['name'] + '.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen(run['command'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace',
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(('mode=', 'epoch=')):
                        print(f"[{run['name']}] {line.rstrip()}", flush=True)
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                run['status'] = 'interrupted'
                save()
                raise
        run['exit_code'] = code
        run['status'] = 'completed' if code == 0 else 'failed'
        if code == 0:
            result = json.loads(Path(run['result']).read_text(encoding='utf-8'))
            run['validation_macro_f1'] = result['validation']['macro_f1']
            run['best_epoch'] = result['best_epoch']
        save()
        if code:
            raise SystemExit(f"{run['name']} failed; see its log")
        print(f"DONE {run['name']} F1={run['validation_macro_f1']:.6f}", flush=True)
    print(f'ALL DONE: {destination}', flush=True)


if __name__ == '__main__':
    main()
