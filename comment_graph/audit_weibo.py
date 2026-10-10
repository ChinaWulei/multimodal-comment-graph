"""Read-only source audit; write a separate JSON report, never change CSV/images."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image


def normalized_text(value):
    return re.sub(r'\s+', '', value).casefold()


def inspect(root):
    index = defaultdict(list)
    for directory in ('nonrumor_images', 'rumor_images'):
        for path in sorted((root / directory).glob('*')):
            if path.is_file():
                index[path.name].append(path)
    cache = {}

    def validate(path):
        if str(path) not in cache:
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                cache[str(path)] = {'valid': True, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            except Exception as exc:
                cache[str(path)] = {'valid': False, 'error': str(exc)}
        return cache[str(path)]

    report = {'root': str(root.resolve()), 'label_folder_counts': {}, 'splits': {}, 'records': {},
              'comments_available': False, 'source_modified': False,
              'identity_warning': 'post_id is rounded scientific notation; preserve it but never use it as unique key'}
    for split, filename in [('train', 'train_origin.csv'), ('validate', 'val_origin.csv'), ('test', 'test_origin.csv')]:
        with (root / filename).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            columns, raw = reader.fieldnames, list(reader)
        rows, folder_counts = [], Counter()
        for offset, row in enumerate(raw):
            text = str(row['content']).strip()
            paths = []
            for url in str(row['image_id']).split('|'):
                name = Path(unquote(urlsplit(url.strip()).path)).name
                paths.extend(index.get(name, []))
            paths = list(dict.fromkeys(paths))
            candidates = [p for p in paths if validate(p)['valid']]
            for p in candidates:
                folder_counts[f"label={row['label']} folder={p.parent.name}"] += 1
            selected = candidates[0] if candidates else None
            failures = []
            if not text:
                failures.append('missing_text')
            if row['label'] not in {'0', '1'}:
                failures.append('invalid_label')
            if selected is None:
                failures.append('no_valid_image')
            digest = hashlib.sha256(normalized_text(text).encode()).hexdigest()
            rows.append({'news_id': 'weibo_text_' + digest, 'source_csv': filename,
                         'source_row_index': offset, 'source_post_id': row['post_id'],
                         'split': split, 'news_text': text, 'label': row['label'],
                         'event_label': row.get('event_label'), 'category': row.get('category'),
                         'image_path': str(selected.resolve()) if selected else None,
                         'image_sha256': cache[str(selected)]['sha256'] if selected else None,
                         'matched_image_count': len(candidates), 'failure_reasons': failures})
        report['records'][split] = rows
        report['label_folder_counts'][split] = dict(folder_counts)
        valid = [r for r in rows if not r['failure_reasons']]
        report['splits'][split] = {'rows': len(rows), 'columns': columns,
                                  'label_counts': dict(Counter(r['label'] for r in rows)),
                                  'valid_rows': len(valid),
                                  'valid_label_counts': dict(Counter(r['label'] for r in valid)),
                                  'unique_source_post_ids': len({r['source_post_id'] for r in rows}),
                                  'unique_text_keys': len({r['news_id'] for r in rows}),
                                  'duplicate_text_rows': len(rows) - len({r['news_id'] for r in rows}),
                                  'failures': dict(Counter(x for r in rows for x in r['failure_reasons']))}
    report['overlaps'] = {}
    for a, b in [('train', 'validate'), ('train', 'test'), ('validate', 'test')]:
        ar, br = report['records'][a], report['records'][b]
        shared_text = {r['news_id'] for r in ar} & {r['news_id'] for r in br}
        shared_image = {r['image_sha256'] for r in ar if r['image_sha256']} & {r['image_sha256'] for r in br if r['image_sha256']}
        report['overlaps'][a + ':' + b] = {'shared_text_keys': sorted(shared_text),
                                        'shared_primary_image_hash_count': len(shared_image)}
    report['images_checked'] = len(cache)
    report['invalid_images'] = {p: r['error'] for p, r in cache.items() if not r['valid']}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new output path to preserve previous audits')
    result = inspect(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in {'records', 'invalid_images'}}, ensure_ascii=True, indent=2))
