"""Build a label-free, train-only entity evidence cache for current MFCL Weibo rows.

The raw Weibo CSVs and the DDU CSVs are checked by SHA-256 before processing.
Only rows that survive this project's image audit can be evidence candidates.
The resulting cache deliberately contains no retrieved label field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jieba
import jieba.posseg as pseg
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STOPWORDS = {"一个", "一些", "一种", "这个", "那个", "这些", "那些", "我们", "你们", "他们", "自己", "已经", "没有", "可以", "可能", "因为", "所以", "但是", "以及", "还是", "就是", "不是", "什么", "怎么", "这样", "那样", "今天", "昨天", "现在", "时候", "表示", "认为", "告诉", "看到", "发布", "消息", "新闻", "报道", "媒体", "微博", "视频", "图片", "照片", "内容", "网友", "相关", "进行", "目前", "现场", "事情", "真的", "感觉", "转发", "评论", "来自", "原来", "最后", "近日", "日前", "发生"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(token: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(token).lower()))


def entities(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for pair in pseg.cut(str(text), HMM=True):
        token = normalise(pair.word)
        if len(token) < 2 or token.isdigit() or token in STOPWORDS:
            continue
        flag = pair.flag.lower()
        if flag.startswith(("nr", "ns", "nt")) or flag == "nz":
            score = 2.0
        elif flag.startswith("n") or flag in {"vn", "eng"}:
            score = 1.0
        else:
            continue
        result[token] = max(result.get(token, 0.0), score)
    return {token: score for token, score in result.items()
            if not any(token != other and token in other for other in result)}


def valid_rows(report: dict, split: str) -> list[dict]:
    seen, rows = set(), []
    for row in report["records"][split]:
        if row["failure_reasons"]:
            continue
        key = row["news_id"] + "_" + row["image_sha256"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def top_indices(scores: torch.Tensor, k: int, excluded: int | None = None) -> list[int]:
    if excluded is not None:
        scores = scores.clone()
        scores[excluded] = -torch.inf
    return scores.topk(min(k, scores.numel())).indices.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path(__file__).with_name("reports") / "weibo_source_audit_20261004.json")
    parser.add_argument("--ddu-root", type=Path, default=Path(r"D:\PycharmProject\MM26-DDU"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("data") / "weibo_entity_evidence_valid_train_only")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-entities", type=int, default=6)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df-ratio", type=float, default=.02)
    parser.add_argument("--entity-rank-weight", type=float, default=.7)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Output cache directory already exists and is non-empty; refuse to overwrite it")
    if not (1 <= args.top_k and 0 < args.entity_rank_weight <= 1):
        raise ValueError("Invalid Top-K or entity rank weight")
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    raw_root = Path(report["root"])
    ddu_root = args.ddu_root
    split_file = {"train": "train_origin.csv", "validate": "val_origin.csv", "test": "test_origin.csv"}
    ddu_split = {"train": "train", "validate": "val", "test": "test"}
    raw_frames, source_hashes = {}, {}
    for split, filename in split_file.items():
        raw, copied = raw_root / filename, ddu_root / "dataset" / "Weibo" / filename
        if not raw.is_file() or not copied.is_file() or sha256(raw) != sha256(copied):
            raise ValueError(f"Raw CSV mismatch for {split}; evidence cache would not map safely")
        raw_frames[split] = pd.read_csv(raw, encoding="utf-8")
        source_hashes[split] = sha256(raw)
    current = {split: valid_rows(report, split) for split in split_file}
    train_rows = current["train"]
    train_source = torch.tensor([int(row["source_row_index"]) for row in train_rows], dtype=torch.long)
    raw_train = raw_frames["train"]
    if train_source.max().item() >= len(raw_train):
        raise ValueError("Audit source row is outside raw train CSV")
    feature_root = ddu_root / "data" / "Weibo" / "features"
    all_features = {}
    for split, alias in ddu_split.items():
        text = torch.load(feature_root / f"{alias}_clip_text_feats.pt", map_location="cpu", weights_only=False)["features"].float()
        image = torch.load(feature_root / f"{alias}_clip_image_feats.pt", map_location="cpu", weights_only=False)["features"].float()
        if len(text) != len(raw_frames[split]) or len(image) != len(raw_frames[split]):
            raise ValueError(f"DDU feature/CSV row mismatch for {split}")
        all_features[split] = (F.normalize(text, dim=-1), F.normalize(image, dim=-1))
    train_text, train_image = all_features["train"]
    memory_text, memory_image = train_text[train_source], train_image[train_source]
    jieba.initialize()
    units = [entities(raw_train.iloc[int(source)]["content"]) for source in train_source]
    df = Counter(unit for sample in units for unit in sample)
    inverted: dict[str, list[int]] = defaultdict(list)
    for memory_index, sample in enumerate(units):
        for unit in sample:
            if args.min_df <= df[unit] <= args.max_df_ratio * len(units):
                inverted[unit].append(memory_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"version": 1, "label_free": True, "candidate_memory": "current valid train rows only",
               "source_hashes": source_hashes, "top_k": args.top_k, "max_entities": args.max_entities,
               "min_df": args.min_df, "max_df_ratio": args.max_df_ratio,
               "entity_rank_weight": args.entity_rank_weight, "splits": {}}
    for split in split_file:
        rows = current[split]
        raw = raw_frames[split]
        q_text_all, q_image_all = all_features[split]
        out = {key: [] for key in ("source_row_indices", "entity_scores", "text_sims", "image_sims", "reliability", "valid_mask", "entity_counts")}
        for row in rows:
            source = int(row["source_row_index"])
            unit_map = entities(raw.iloc[source]["content"])
            selected_entities = []
            for entity, priority in unit_map.items():
                count = df.get(entity, 0)
                if args.min_df <= count <= args.max_df_ratio * len(units):
                    idf = math.log((len(units) + 1) / (count + 1)) + 1.
                    selected_entities.append((idf * (1.35 if priority > 1 else 1.), entity, idf))
            selected_entities.sort(reverse=True)
            selected_entities = selected_entities[:args.max_entities]
            support: dict[int, float] = defaultdict(float)
            total_idf = sum(item[2] for item in selected_entities)
            for _, entity, idf in selected_entities:
                for memory_index in inverted[entity]:
                    if split == "train" and int(train_source[memory_index]) == source:
                        continue
                    support[memory_index] += idf
            q_text, q_image = q_text_all[source], q_image_all[source]
            text_similarity = q_text @ memory_text.T
            if support:
                ids = torch.tensor(list(support), dtype=torch.long)
                coverage = torch.tensor([support[int(i)] / max(total_idf, 1e-8) for i in ids], dtype=torch.float32)
                rank = args.entity_rank_weight * coverage + (1. - args.entity_rank_weight) * ((text_similarity[ids] + 1.) / 2.).clamp(0., 1.)
                order = rank.argsort(descending=True)[:args.top_k]
                chosen, scores = ids[order], coverage[order]
            else:
                chosen, scores = torch.empty(0, dtype=torch.long), torch.empty(0)
            # Padding candidates are included only for fixed tensor shape and
            # are masked out: they cannot influence the evidence representation.
            if len(chosen) < args.top_k:
                excluded = int((train_source == source).nonzero()[0]) if split == "train" and (train_source == source).any() else None
                for fallback in top_indices(text_similarity, args.top_k + 1, excluded):
                    if fallback not in chosen.tolist():
                        chosen = torch.cat((chosen, torch.tensor([fallback])))
                        scores = torch.cat((scores, torch.zeros(1)))
                    if len(chosen) == args.top_k:
                        break
            chosen = chosen[:args.top_k]
            text_sims, image_sims = text_similarity[chosen], q_image @ memory_image[chosen].T
            reliability = scores * (.5 + .5 * ((text_sims + 1.) / 2.).clamp(0., 1.))
            out["source_row_indices"].append(source)
            out["entity_scores"].append(scores)
            out["text_sims"].append(text_sims)
            out["image_sims"].append(image_sims)
            out["reliability"].append(reliability)
            out["valid_mask"].append(scores > 0)
            out["entity_counts"].append(len(selected_entities))
        cache = {
            key: (torch.tensor(value, dtype=torch.long) if key in {"source_row_indices", "entity_counts"}
                  else torch.stack(value))
            for key, value in out.items()
        }
        cache["metadata"] = dict(summary, split=split, selected_rows=len(rows), raw_rows=len(raw))
        torch.save(cache, args.output_dir / f"{split}.pt")
        summary["splits"][split] = {"selected_rows": len(rows),
                                     "entity_covered_rows": int(cache["valid_mask"].any(1).sum()),
                                     "coverage": float(cache["valid_mask"].any(1).float().mean())}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
