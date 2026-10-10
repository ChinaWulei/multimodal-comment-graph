"""Summarize every declared run, including missing/failure status and paired seeds."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(directories):
    rows, seen = [], set()
    for directory in directories:
        plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        states = {r['name']: r for r in plan['runs']}
        for seed in plan['seeds']:
            for arm in plan['arms']:
                key = arm, seed
                if key in seen:
                    raise ValueError(f'Duplicate arm/seed {key}; do not cherry-pick among repeated runs')
                seen.add(key)
                name = f'{arm}_seed{seed}'
                row = {'arm': arm, 'seed': seed, 'status': states.get(name, {}).get('status', 'not_started'),
                       'directory': str(directory.resolve())}
                if row['status'] == 'completed':
                    result = json.loads((directory / f'{name}.json').read_text(encoding='utf-8'))
                    if 'test' in result:
                        raise ValueError('This report is validation-only, but a result contains test evaluation')
                    val = result['validation']
                    audit = val['mismatch_audit']
                    row.update({'best_epoch': result['best_epoch'], 'macro_f1': val['macro_f1'],
                                'validation_loss': val['loss'], 'confusion_matrix': val['confusion_matrix'],
                                'base_macro_f1': val['global_macro_f1'],
                                'mismatch_macro_f1': audit['mismatch_macro_f1'],
                                'clean_mismatch_probability_mae': audit['clean_vs_mismatch_probability_mae'],
                                'clean_mismatch_prediction_flip_rate': audit['clean_vs_mismatch_prediction_flip_rate'],
                                'checkpoint': result['checkpoint']})
                rows.append(row)
    controls = {r['seed']: r for r in rows if r['arm'] == 'control' and r['status'] == 'completed'}
    comparisons = {}
    for arm in sorted({r['arm'] for r in rows} - {'control'}):
        candidates = {r['seed']: r for r in rows if r['arm'] == arm and r['status'] == 'completed'}
        seeds = sorted(controls.keys() & candidates.keys())
        deltas = [candidates[s]['macro_f1'] - controls[s]['macro_f1'] for s in seeds]
        comparisons[arm] = {'paired_seeds': seeds, 'deltas': deltas,
                            'missing_pair_seeds': sorted(controls.keys() ^ candidates.keys())}
        if seeds:
            comparisons[arm].update({
                'control_mean': statistics.mean(controls[s]['macro_f1'] for s in seeds),
                'candidate_mean': statistics.mean(candidates[s]['macro_f1'] for s in seeds),
                'mean_delta': statistics.mean(deltas),
                'delta_sample_std': statistics.stdev(deltas) if len(deltas) > 1 else None,
                'wins': sum(d > 1e-9 for d in deltas), 'ties': sum(abs(d) <= 1e-9 for d in deltas),
                'losses': sum(d < -1e-9 for d in deltas),
            })
    return {'test_used': False, 'warning': 'Exploratory validation comparisons; same 118 news / 4 fake in every seed. '
                                         'Not independent test improvement, significance, or a comparison against all historical methods.',
            'runs': rows, 'paired_comparisons': comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directories', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise SystemExit('Use a new report path to preserve previous results')
    report = summarize(args.directories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 完整训练集探索性验证实验', '', report['warning'], '',
             '| 方法 | seed | 状态 | 最佳轮次 | 验证 F1 | 验证 CE | 错配 F1 |',
             '|---|---:|---|---:|---:|---:|---:|']
    for r in report['runs']:
        if r['status'] == 'completed':
            lines.append(f"| {r['arm']} | {r['seed']} | completed | {r['best_epoch']} | "
                         f"{r['macro_f1']:.6f} | {r['validation_loss']:.6f} | {r['mismatch_macro_f1']:.6f} |")
        else:
            lines.append(f"| {r['arm']} | {r['seed']} | {r['status']} | — | — | — | — |")
    lines += ['', '## 配对比较', '']
    for arm, result in report['paired_comparisons'].items():
        lines += [f'### {arm}', '', '```json', json.dumps(result, ensure_ascii=False, indent=2), '```', '']
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report['paired_comparisons'], indent=2))
    print(f'report={args.output.resolve()}')


if __name__ == '__main__':
    main()
