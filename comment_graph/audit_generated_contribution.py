"""Validation-only: normal, removed, repeated and mismatched generated comments.

Loads a frozen source_isolated checkpoint. Never fits a parameter or accesses test.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from munch import DefaultMunch
from comment_graph.common import make_dataset, make_tokenizer_and_transform
from comment_graph.isolated_fusion import select_nodes
from comment_graph.model import CommentGraphMFCL
from comment_graph.train import make_loader, evaluate


class InterventionLoader:
    def __init__(self, loader, condition):
        self.loader, self.condition = loader, condition

    def __iter__(self):
        for batch in self.loader:
            if self.condition == 'removed':
                batch, _ = select_nodes(batch, batch.node_type != 2)
            elif self.condition == 'repeated':
                batch = batch.clone()
                for i in range(batch.root_input_ids.size(0)):
                    indices = ((batch.node_type == 2) & (batch.batch == i)).nonzero().flatten()
                    if indices.numel():
                        # Content repetition, same number of nodes and same news.
                        batch.node_input_ids[indices] = batch.node_input_ids[indices[0]].clone()
                        batch.node_attention_mask[indices] = batch.node_attention_mask[indices[0]].clone()
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; use a new path to preserve previous diagnostics')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['decision_mode'] != 'source_isolated':
        raise ValueError('This diagnostic requires source_isolated')
    config = DefaultMunch.fromDict(checkpoint['config'])
    tokenizer, transform = make_tokenizer_and_transform(config)
    path = args.config.resolve()
    validation = make_dataset(config, path, 'validate', checkpoint['mode'], tokenizer, transform)
    device = torch.device('cuda')
    model = CommentGraphMFCL(config, checkpoint['mode'], checkpoint['graph_structure'],
                            checkpoint['pooling_mode'], checkpoint['decision_mode']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    gc.collect()
    loader = make_loader(validation, config, shuffle=False)
    # Unweighted CE for this sensitivity-only report, identical across all views.
    reports = {'clean': evaluate(model, loader, device, bool(config.training.use_amp), None,
                                 audit_mismatch=True, audit_seed=int(config.robustness['audit_seed']))}
    clean_by_id = {r['news_id']: r for r in reports['clean']['prediction_records']}
    for condition in ('removed', 'repeated'):
        result = evaluate(model, InterventionLoader(loader, condition), device,
                          bool(config.training.use_amp), None)
        result['probability_mae_from_clean'] = float(np.mean([
            abs(r['fake_probability'] - clean_by_id[r['news_id']]['fake_probability'])
            for r in result['prediction_records']]))
        result['prediction_flips_from_clean'] = sum(
            r['prediction'] != clean_by_id[r['news_id']]['prediction'] for r in result['prediction_records'])
        reports[condition] = result
    base_predictions = {r['news_id']: r['global_fake_probability'] for r in reports['clean']['prediction_records']}
    removal_delta = max(abs(r['fake_probability'] - base_predictions[r['news_id']])
                        for r in reports['removed']['prediction_records'])
    assert removal_delta < 2e-4, f'Removing generated nodes changed isolated base: {removal_delta}'
    report = {'checkpoint': str(args.checkpoint.resolve()), 'split': 'validate', 'test_loaded': False,
              'news_count': len(validation), 'ce_weighting': 'unweighted in this diagnostic only',
              'base_removal_max_probability_delta': removal_delta, 'conditions': reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({condition: {k: result.get(k) for k in
                                  ['macro_f1', 'loss', 'prediction_flips_from_clean', 'probability_mae_from_clean']}
                      for condition, result in reports.items()}, indent=2))
    print(f'report={args.output.resolve()}')


if __name__ == '__main__':
    main()
