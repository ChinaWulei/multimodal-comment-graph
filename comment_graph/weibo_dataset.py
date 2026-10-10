"""Image/text-only Weibo adapter from the audited local CSV dataset.

No comments or propagation tree are invented. Original row labels are retained.
"""
import json
from pathlib import Path

import torch

from comment_graph.dataset import CheckedCommentGraphDataset


class WeiboImageTextDataset(CheckedCommentGraphDataset):
    def __init__(self, audit_path, split, tokenizer, image_transform, config,
                 max_samples=None, allowed_news_ids=None):
        report = json.loads(Path(audit_path).read_text(encoding='utf-8'))
        if any(v['shared_text_keys'] for v in report['overlaps'].values()):
            raise ValueError('Weibo split text overlap: resolve the data protocol first')
        self.mode = 'baseline'
        self.tokenizer, self.image_transform = tokenizer, image_transform
        self.max_real_comments = self.max_generated_comments = 0
        self.max_news_length = int(config.data.max_news_length)
        self.max_comment_length = int(config.data.max_comment_length)
        self.real_comments, self.generated_comments = {}, {}
        self.records, self.missing_generated_ids = [], []
        seen = {}
        self.duplicate_rows_skipped = 0
        for row in report['records'][split]:
            if row['failure_reasons']:
                continue
            # This is an internal content+image key, NOT a recovered Weibo ID.
            key = row['news_id'] + '_' + row['image_sha256']
            if key in seen:
                if seen[key] != row['label']:
                    raise ValueError('Duplicate content/image has conflicting labels')
                self.duplicate_rows_skipped += 1
                continue
            seen[key] = row['label']
            if allowed_news_ids is not None and key not in allowed_news_ids:
                continue
            if not Path(row['image_path']).is_file():
                raise FileNotFoundError(row['image_path'])
            self.records.append(dict(row, news_id=key, binary_label=row['label'],
                                     primary_image_path=row['image_path']))
            if max_samples is not None and len(self.records) >= max_samples:
                break
        if not self.records:
            raise ValueError(f'No usable Weibo {split} records')
        self.entity_evidence = None
        evidence = config.get("entity_evidence", {})
        if str(evidence.get("mode", "none")) != "none":
            cache_path = Path(str(evidence.get("cache_dir", ""))).resolve() / f"{split}.pt"
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"Label-free entity evidence cache not found: {cache_path}. "
                    "Run prepare_weibo_entity_evidence.py first."
                )
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            metadata = cached.get("metadata", {})
            if metadata.get("label_free") is not True or metadata.get("candidate_memory") != "current valid train rows only":
                raise ValueError("Refusing a cache that is not explicitly label-free/current-valid-train-only")
            required = ("entity_scores", "text_sims", "image_sims", "reliability", "valid_mask")
            if any(key not in cached for key in required) or "retrieval_labels" in cached:
                raise ValueError("Invalid evidence cache: missing features or contains retrieved labels")
            source_rows = torch.as_tensor(cached["source_row_indices"], dtype=torch.long)
            if source_rows.ndim != 1 or any(torch.as_tensor(cached[key]).shape[0] != len(source_rows) for key in required):
                raise ValueError("Evidence cache source-row alignment is invalid")
            lookup = {int(source): index for index, source in enumerate(source_rows.tolist())}
            record_rows = [int(record["source_row_index"]) for record in self.records]
            if len(set(record_rows)) != len(record_rows) or any(source not in lookup for source in record_rows):
                raise ValueError("Evidence cache does not exactly cover current deduplicated dataset rows")
            positions = [lookup[source] for source in record_rows]
            self.entity_evidence = {key: torch.as_tensor(cached[key])[positions] for key in required}

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.entity_evidence is not None:
            # PyG concatenates tensors along dimension 0. Keep a leading sample
            # dimension so a batch becomes [B,K], never a flattened [B*K].
            data.entity_evidence_scores = self.entity_evidence["entity_scores"][index].unsqueeze(0)
            data.entity_evidence_text_sims = self.entity_evidence["text_sims"][index].unsqueeze(0)
            data.entity_evidence_image_sims = self.entity_evidence["image_sims"][index].unsqueeze(0)
            data.entity_evidence_reliability = self.entity_evidence["reliability"][index].unsqueeze(0)
            data.entity_evidence_mask = self.entity_evidence["valid_mask"][index].unsqueeze(0)
        return data
