"""Predeclared finite validation-only comparisons. No test evaluation or winner deletion."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ARMS = {
    'control': ['--robustness-mode', 'none', '--sampling', 'shuffle'],
    'mismatch': ['--robustness-mode', 'mismatch_fallback', '--sampling', 'shuffle'],
    'balanced': ['--robustness-mode', 'none', '--sampling', 'balanced'],
    'null_centered': ['--robustness-mode', 'none', '--sampling', 'shuffle',
                      '--correction-reference', 'null_generated'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--arms', choices=sorted(ARMS), nargs='+', default=['control', 'mismatch', 'balanced'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        raise ValueError('Duplicate seeds/arms are not allowed')
    destination = (args.output_dir or Path(__file__).parent / 'outputs' /
                   f'validation_search_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    destination.mkdir(parents=True, exist_ok=False)
    plan = {'created': datetime.now().isoformat(), 'seeds': args.seeds, 'arms': args.arms,
            'epochs': args.epochs, 'patience': args.patience, 'test_allowed': False,
            'primary_metric': 'clean validation macro-F1',
            'selection_rule': 'within run: strict best macro-F1, ties keep earliest epoch',
            'limit': 'finite predeclared comparisons; no guarantee or requirement to find a positive result',
            'runs': []}
    (destination / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'EXPERIMENT DIRECTORY: {destination}', flush=True)
    for seed in args.seeds:
        for arm in args.arms:
            name = f'{arm}_seed{seed}'
            result_path = destination / f'{name}.json'
            command = [sys.executable, '-u', str(Path(__file__).with_name('train.py')),
                       '--config', str(args.config.resolve()), '--mode', 'real_generated',
                       '--graph-structure', 'dynamic_cross_modal', '--pooling-mode', 'mean',
                       '--decision-mode', 'source_isolated', '--audit-mismatch',
                       '--correction-reference', 'none',
                       '--seed', str(seed), '--epochs', str(args.epochs), '--patience', str(args.patience),
                       '--checkpoint-dir', str(destination / name), '--summary-json', str(result_path),
                       '--allow-formal-training', *ARMS[arm]]
            run = {'name': name, 'command': command, 'status': 'running'}
            plan['runs'].append(run)
            (destination / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
            print(f'START {name}', flush=True)
            with (destination / f'{name}.log').open('w', encoding='utf-8') as log:
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding='utf-8', errors='replace',
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith(('mode=', 'epoch=')):
                            print(f'[{name}] {line.rstrip()}', flush=True)
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    child.wait()
                    run['status'] = 'interrupted'
                    (destination / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
                    raise
            run['exit_code'] = code
            run['status'] = 'completed' if code == 0 else 'failed'
            if code == 0:
                result = json.loads(result_path.read_text(encoding='utf-8'))
                run['validation_macro_f1'] = result['validation']['macro_f1']
                run['best_epoch'] = result['best_epoch']
            (destination / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
            if code:
                raise SystemExit(f'{name} failed ({code}); inspect {destination / (name + ".log")}')
            print(f'DONE {name} validation_macro_f1={run["validation_macro_f1"]:.6f}', flush=True)
    print(f'ALL DONE: {destination}', flush=True)


if __name__ == '__main__':
    main()
