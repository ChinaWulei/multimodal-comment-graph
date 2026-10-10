"""Evaluate a frozen checkpoint without retraining; test requires explicit consent."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from munch import DefaultMunch
from comment_graph.common import make_dataset, make_tokenizer_and_transform
from comment_graph.model import CommentGraphMFCL
from comment_graph.train import class_weights, evaluate, make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True,
                        help='Original config location for resolving relative data paths; checkpoint settings are authoritative')
    parser.add_argument('--split', choices=['validate', 'test'], default='validate')
    parser.add_argument('--allow-test', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.split == 'test' and not args.allow_test:
        raise SystemExit('Freeze the method before passing --allow-test')
    if args.output.exists():
        raise SystemExit('Output exists; choose a new path, never overwrite results')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = DefaultMunch.fromDict(checkpoint['config'])
    tokenizer, transform = make_tokenizer_and_transform(config)
    training = make_dataset(config, args.config.resolve(), 'train', checkpoint['mode'], tokenizer, transform)
    dataset = make_dataset(config, args.config.resolve(), args.split, checkpoint['mode'], tokenizer, transform)
    if {r['news_id'] for r in training.records} & {r['news_id'] for r in dataset.records}:
        raise ValueError('Training/evaluation ID overlap')
    device = torch.device('cuda')
    model = CommentGraphMFCL(config, checkpoint['mode'], checkpoint['graph_structure'],
                            checkpoint['pooling_mode'], checkpoint['decision_mode']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    epoch = checkpoint['epoch']
    del checkpoint
    gc.collect()
    weights = class_weights(training, device) if config.training.class_weighted_loss else None
    result = evaluate(model, make_loader(dataset, config, False), device, bool(config.training.use_amp), weights)
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    report = {'checkpoint': str(args.checkpoint.resolve()), 'sha256': digest.hexdigest(),
              'split': args.split, 'epoch': epoch, 'news_count': len(dataset),
              'batch_size': int(config.training.batch_size), 'metrics': result,
              'training_performed': False, 'threshold': 'argmax; no test threshold fitting'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'prediction_records'}, indent=2))


if __name__ == '__main__':
    main()
