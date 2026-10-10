"""Post-run checks and paired news bootstrap; does not change model selection/gate."""
import json
from collections import Counter
import numpy as np
try:
    from .experiment import OUT, SOURCE, ARMS, SEEDS, sha, save, verify
except ImportError:
    from experiment import OUT, SOURCE, ARMS, SEEDS, sha, save, verify


def macro_f1(y, predictions):
    matrix = np.bincount(3 * y + predictions, minlength=9).reshape(3, 3)
    denominator = matrix.sum(0) + matrix.sum(1)
    return np.divide(2 * np.diag(matrix), denominator, out=np.zeros(3, dtype=float), where=denominator != 0).mean()


def main():
    protocol = verify()
    ids = sorted(protocol['primary_ids'])
    truth = None; predictions = {}; runs = {}
    for arm in [*ARMS, 'A_real', 'D_observed_root', 'E_observed_parent']:
        folder = OUT if arm in ARMS else SOURCE
        runs[arm] = []; predictions[arm] = []
        for seed in SEEDS:
            row = json.loads((folder / 'classification' / f'{arm}_seed{seed}.json').read_text(encoding='utf-8'))
            by_id = {r['news_id']: r for r in row['validation']['prediction_records']}
            assert set(by_id) == set(ids)
            labels = np.array([by_id[n]['label'] for n in ids])
            if truth is None: truth = labels
            else: assert np.array_equal(truth, labels)
            pred = np.array([by_id[n]['prediction'] for n in ids])
            assert abs(macro_f1(truth, pred) - row['validation']['macro_f1']) < 1e-12
            predictions[arm].append(pred); runs[arm].append(row)
            if arm in ARMS:
                assert row['protocol_sha256'] == sha(OUT / 'protocol.json')
                assert row['checkpoint_sha256'] == sha(row['checkpoint'])
                auxiliary = json.loads((folder / 'classification' / f'{arm}_seed{seed}_validation77.json').read_text(encoding='utf-8'))
                assert auxiliary['checkpoint_sha256'] == row['checkpoint_sha256']
                assert auxiliary['connection_policy'] == ARMS[arm]
                assert {r['news_id'] for r in auxiliary['validation']['prediction_records']} >= set(ids)
        predictions[arm] = np.stack(predictions[arm])
    comparisons = {}
    rng = np.random.default_rng(20261010)
    samples = rng.integers(0, len(ids), size=(2000, len(ids)))
    boot = {arm: np.array([np.mean([macro_f1(truth[index], p[index]) for p in values]) for index in samples])
            for arm, values in predictions.items()}
    for other in ['A_real', 'D_observed_root', 'G_root_permuted_parent']:
        differences = boot['F_root_parent'] - boot[other]
        by_seed = {}
        for s, f, b in zip(SEEDS, predictions['F_root_parent'], predictions[other]):
            fc, bc = f == truth, b == truth
            by_seed[str(s)] = {'F_only_correct': int((fc & ~bc).sum()), 'other_only_correct': int((~fc & bc).sum()),
                              'both_correct': int((fc & bc).sum()), 'both_wrong': int((~fc & ~bc).sum())}
        comparisons[other] = {'paired_news_bootstrap_95_interval': np.quantile(differences, [.025, .975]).tolist(),
                              'prediction_outcomes': by_seed}
    result = {'passed': True, 'exact_primary_ids_checked': 44, 'checkpoint_and_source_hashes_checked': True,
              'macro_f1_recomputed': True, 'paired_bootstrap': comparisons, 'bootstrap_repetitions': 2000,
              'warning': 'Exploratory news-level bootstrap averages fixed seeds; does not account for previous validation model selection or all uncertainty.',
              'test_evaluated': False}
    save(OUT / 'result_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
