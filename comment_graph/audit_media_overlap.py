"""Exact primary-image overlap between existing generated train/validation inputs.

No test evaluation or split changes. Identical media is a risk indicator, not proof
that news articles are duplicates or that model inputs contain truth labels.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checked-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'comment_generator' / 'checked')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists; choose a new path')
    training = list(read(args.checked_dir / 'outputs' / 'generated_train.jsonl'))
    validation = list(read(args.checked_dir / 'outputs' / 'generated_validate.jsonl'))
    metadata = {r['metadata']['news_id']: r['metadata'] for r in read(args.checked_dir / 'data' / 'news_contexts.jsonl')
                if r['metadata']['split'] in {'train', 'validate'}}
    train_images = defaultdict(list)
    def fingerprint(record):
        return hashlib.sha256(Path(record['image_path']).read_bytes()).hexdigest()
    def normalize(text):
        return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).casefold()
    for r in training:
        train_images[fingerprint(r)].append(r)
    train_texts = {normalize(r['news_text']) for r in training}
    matches, groups = [], {'seen_image': Counter(), 'unseen_image': Counter()}
    for r in validation:
        digest = fingerprint(r)
        donors = train_images.get(digest, [])
        label = metadata[r['news_id']]['labels']['binary_label']
        groups['seen_image' if donors else 'unseen_image'][label] += 1
        if donors:
            matches.append({'validation_news_id': r['news_id'], 'validation_label': label,
                            'validation_text': r['news_text'], 'validation_image_path': r['image_path'],
                            'sha256': digest,
                            'training_matches': [{'news_id': d['news_id'], 'text': d['news_text'],
                                                  'image_path': d['image_path'],
                                                  'label': metadata[d['news_id']]['labels']['binary_label']}
                                                 for d in donors]})
    report = {'train_news': len(training), 'validation_news': len(validation),
              'id_overlap': len({r['news_id'] for r in training} & {r['news_id'] for r in validation}),
              'normalized_exact_text_overlap': sum(normalize(r['news_text']) in train_texts for r in validation),
              'exact_primary_image_overlap': len(matches), 'class_counts': groups,
              'validation_image_groups': {r['news_id']: ('seen_image' if fingerprint(r) in train_images else 'unseen_image')
                                          for r in validation},
              'matches': matches,
              'limitation': 'Exact file bytes only, not perceptual near-duplicates or event overlap. '
                            'Repeated stock/media images can accompany genuinely different news. '
                            'No test predictions, no split changes, no model selection on image groups.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'matches','validation_image_groups'}}, indent=2))
    print(f'report={args.output.resolve()}')


if __name__ == '__main__':
    main()
