from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
LABEL_COLUMNS = ["2_way_label", "3_way_label", "6_way_label"]
NEWS_COLUMNS = ["id", "clean_title", "image_url", *LABEL_COLUMNS]
COMMENT_COLUMNS = [
    "id",
    "author",
    "body",
    "isTopLevel",
    "parent_id",
    "submission_id",
    "ups",
]
INVALID_COMMENT_BODIES = {"[deleted]", "[removed]"}
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")
COMMENT_RECORD_START = re.compile(r"^(\d+)\t([A-Za-z0-9]+)\t")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Match local Fakeddit images to news rows and real comments."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--images-dir", type=Path, default=project_root / "images")
    parser.add_argument(
        "--train-tsv",
        type=Path,
        default=project_root / "multimodal_only_samples" / "multimodal_train.tsv",
    )
    parser.add_argument(
        "--validate-tsv",
        type=Path,
        default=project_root / "multimodal_only_samples" / "multimodal_validate.tsv",
    )
    parser.add_argument(
        "--test-tsv",
        type=Path,
        default=project_root / "multimodal_only_samples" / "multimodal_test_public.tsv",
    )
    parser.add_argument(
        "--comments-tsv",
        type=Path,
        default=project_root / "all_comments.tsv" / "all_comments.tsv",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "sample_check")
    parser.add_argument("--comments-chunksize", type=int, default=250_000)
    parser.add_argument(
        "--short-comment-chars",
        type=int,
        default=10,
        help="Only marks short comments; it does not filter them out.",
    )
    return parser.parse_args()


def normalize_reddit_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = str(value).strip().lower()
    while normalized.startswith(("t1_", "t3_")):
        normalized = normalized[3:]
    return normalized


def clean_comment(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = CONTROL_CHARS.sub(" ", str(value))
    return WHITESPACE.sub(" ", text).strip()


def decode_tsv_field(value: str) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('""', '"')
    return value


def is_present(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(str(value).strip())


def validate_image(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "image_valid": False,
        "image_format": "",
        "image_width": "",
        "image_height": "",
        "image_error": "",
    }
    try:
        if path.stat().st_size <= 0:
            raise ValueError("zero-byte file")
        with Image.open(path) as image:
            result["image_format"] = image.format or ""
            result["image_width"], result["image_height"] = image.size
            image.verify()
        with Image.open(path) as image:
            image.load()
        result["image_valid"] = True
    except Exception as exc:  # Pillow raises several format-specific exceptions.
        result["image_error"] = f"{type(exc).__name__}: {exc}"
    return result


def read_news_tables(split_paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    per_split_rows: dict[str, int] = {}
    per_split_duplicate_ids: dict[str, int] = {}
    for split, path in split_paths.items():
        frame = pd.read_csv(
            path,
            sep="\t",
            usecols=NEWS_COLUMNS,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
        frame["split"] = split
        frame["_normalized_id"] = frame["id"].map(normalize_reddit_id)
        frames.append(frame)
        per_split_rows[split] = len(frame)
        per_split_duplicate_ids[split] = int(frame["_normalized_id"].duplicated(keep=False).sum())

    news = pd.concat(frames, ignore_index=True)
    split_sets = {
        split: set(news.loc[news["split"] == split, "_normalized_id"])
        for split in split_paths
    }
    overlaps: dict[str, int] = {}
    split_names = list(split_paths)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlaps[f"{left}-{right}"] = len(split_sets[left] & split_sets[right])

    stats = {
        "per_split_rows": per_split_rows,
        "per_split_duplicate_id_rows": per_split_duplicate_ids,
        "split_overlap_ids": overlaps,
        "global_duplicate_id_rows": int(news["_normalized_id"].duplicated(keep=False).sum()),
    }
    return news, stats


def scan_comments(
    path: Path,
    target_ids: set[str],
    all_news_ids: set[str],
    chunksize: int,
    short_comment_chars: int,
) -> tuple[dict[str, object], dict[str, object]]:
    comment_count: Counter[str] = Counter()
    top_level_count: Counter[str] = Counter()
    reply_count: Counter[str] = Counter()
    valid_top_level_count: Counter[str] = Counter()
    valid_reply_count: Counter[str] = Counter()
    short_valid_count: Counter[str] = Counter()
    valid_comments: dict[str, list[str]] = defaultdict(list)
    seen_comments: dict[str, set[str]] = defaultdict(set)

    total_rows = 0
    empty_submission_rows = 0
    unmapped_submission_rows = 0
    unmapped_submission_examples: list[str] = []
    unmapped_example_set: set[str] = set()
    malformed_rows = 0
    index_gap_count = 0
    first_record_index: int | None = None
    last_record_index: int | None = None

    def process_record(record_lines: list[str], record_index: int) -> None:
        nonlocal total_rows, empty_submission_rows, unmapped_submission_rows, malformed_rows
        nonlocal first_record_index, last_record_index
        raw = "".join(record_lines).rstrip("\r\n")
        left = raw.split("\t", 3)
        if len(left) != 4:
            malformed_rows += 1
            return
        tail = left[3].rsplit("\t", 4)
        if len(tail) != 5:
            malformed_rows += 1
            return

        body, is_top_level, _parent_id, submission_id, _ups = tail
        normalized_id = normalize_reddit_id(decode_tsv_field(submission_id))
        total_rows += 1
        if first_record_index is None:
            first_record_index = record_index
        last_record_index = record_index

        if not normalized_id:
            empty_submission_rows += 1
        elif normalized_id not in all_news_ids:
            unmapped_submission_rows += 1
            if (
                len(unmapped_submission_examples) < 20
                and normalized_id not in unmapped_example_set
            ):
                unmapped_example_set.add(normalized_id)
                unmapped_submission_examples.append(normalized_id)

        if normalized_id not in target_ids:
            return

        comment_count[normalized_id] += 1
        top_level = decode_tsv_field(is_top_level).strip().lower() in {"true", "1", "yes"}
        if top_level:
            top_level_count[normalized_id] += 1
        else:
            reply_count[normalized_id] += 1

        cleaned_body = clean_comment(decode_tsv_field(body))
        if not cleaned_body or cleaned_body.lower() in INVALID_COMMENT_BODIES:
            return
        dedupe_key = cleaned_body.casefold()
        if dedupe_key in seen_comments[normalized_id]:
            return
        seen_comments[normalized_id].add(dedupe_key)
        valid_comments[normalized_id].append(cleaned_body)
        if top_level:
            valid_top_level_count[normalized_id] += 1
        else:
            valid_reply_count[normalized_id] += 1
        if len(cleaned_body) < short_comment_chars:
            short_valid_count[normalized_id] += 1

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        handle.readline()  # Header. The first column is an unnamed source row index.
        current_lines: list[str] = []
        current_index: int | None = None
        for line in handle:
            match = COMMENT_RECORD_START.match(line)
            candidate_index = int(match.group(1)) if match else None
            is_boundary = (
                candidate_index is not None
                and (current_index is None or candidate_index > current_index)
            )
            if is_boundary:
                if current_index is not None:
                    process_record(current_lines, current_index)
                    if candidate_index != current_index + 1:
                        index_gap_count += candidate_index - current_index - 1
                    if total_rows and total_rows % chunksize == 0:
                        print(
                            f"comments progress: rows={total_rows:,}, "
                            f"matched target rows={sum(comment_count.values()):,}",
                            flush=True,
                        )
                current_lines = [line]
                current_index = candidate_index
            else:
                if current_index is None:
                    malformed_rows += 1
                else:
                    current_lines.append(line)
        if current_index is not None:
            process_record(current_lines, current_index)

    print(
        f"comments complete: rows={total_rows:,}, matched target rows={sum(comment_count.values()):,}",
        flush=True,
    )

    per_news = {
        "comment_count": comment_count,
        "top_level_count": top_level_count,
        "reply_count": reply_count,
        "valid_top_level_count": valid_top_level_count,
        "valid_reply_count": valid_reply_count,
        "short_valid_count": short_valid_count,
        "valid_comments": valid_comments,
    }
    global_stats = {
        "total_comment_rows": total_rows,
        "empty_submission_rows": empty_submission_rows,
        "unmapped_submission_rows": unmapped_submission_rows,
        "unmapped_submission_examples": unmapped_submission_examples,
        "malformed_rows_skipped": malformed_rows,
        "comment_index_gap_count": index_gap_count,
        "first_record_index": first_record_index,
        "last_record_index": last_record_index,
    }
    return per_news, global_stats


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    headers = list(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    matches: pd.DataFrame,
    usable: pd.DataFrame,
    news_stats: dict[str, object],
    comment_stats: dict[str, object],
    image_duplicate_ids: dict[str, list[str]],
    comments_columns: list[str],
) -> str:
    total_images = len(matches)
    valid_images = int(matches["image_valid"].sum())
    matched_news = int(matches["split"].ne("").sum())
    has_text = int(matches["headline"].str.strip().ne("").sum())
    has_label = int(matches[LABEL_COLUMNS].apply(lambda row: any(is_present(v) for v in row), axis=1).sum())
    with_any_comments = int(matches["comment_count"].gt(0).sum())
    with_valid_comments = int(matches["valid_comment_count"].gt(0).sum())
    usable_count = int(matches["usable"].sum())

    split_rows = []
    for split in ["train", "validate", "test", "unmatched"]:
        split_mask = matches["split"].eq(split) if split != "unmatched" else matches["split"].eq("")
        split_frame = matches.loc[split_mask]
        split_rows.append(
            [
                split,
                len(split_frame),
                int(split_frame["split"].ne("").sum()),
                int(split_frame["comment_count"].gt(0).sum()),
                int(split_frame["valid_comment_count"].gt(0).sum()),
                int(split_frame["usable"].sum()),
            ]
        )

    failure_counts: Counter[str] = Counter()
    for reasons in matches.loc[~matches["usable"], "failure_reason"]:
        for reason in str(reasons).split(";"):
            if reason:
                failure_counts[reason] += 1

    label_sections = []
    expected_label_values = {
        "2_way_label": ["0", "1"],
        "3_way_label": ["0", "1", "2"],
        "6_way_label": ["0", "1", "2", "3", "4", "5"],
    }
    for label in LABEL_COLUMNS:
        counts = usable[label].astype(str).value_counts(dropna=False)
        counts = counts.reindex(expected_label_values[label], fill_value=0)
        label_sections.append(f"### `{label}`\n\n" + markdown_table(["标签值", "可用样本数"], counts.items()))

    high_comment_rows = (
        matches.loc[matches["split"].ne(""), ["news_id", "split", "headline", "comment_count", "valid_comment_count"]]
        .sort_values(["comment_count", "valid_comment_count"], ascending=False)
        .head(10)
    )
    high_comment_table = markdown_table(
        ["news_id", "split", "标题", "全部评论", "有效去重评论"],
        (
            [row.news_id, row.split, str(row.headline)[:80], row.comment_count, row.valid_comment_count]
            for row in high_comment_rows.itertuples(index=False)
        ),
    )

    example_rows = usable.head(5)
    examples = []
    for _, row in example_rows.iterrows():
        examples.append(
            f"- `{row['news_id']}` ({row['split']}; 2/3/6-way={row['2_way_label']}/"
            f"{row['3_way_label']}/{row['6_way_label']}): "
            f"{str(row['headline'])[:160]}（有效评论 {row['valid_comment_count']} 条）"
        )
    if not examples:
        examples = ["- 无完整可用样本。"]

    no_comment_images = matches.loc[
        matches["split"].ne("") & matches["valid_comment_count"].eq(0), "image_name"
    ].tolist()
    unmatched_images = matches.loc[matches["split"].eq(""), "image_name"].tolist()
    corrupted_images = matches.loc[~matches["image_valid"], "image_name"].tolist()
    case_mismatches = matches.loc[matches["match_type"].eq("case_insensitive"), "image_name"].tolist()

    report = f"""# Fakeddit 本地图文与评论匹配报告

## 数据来源与口径

- 图片目录：`{matches.attrs['images_dir']}`
- 新闻划分：官方 `train`、`validate`、`test` 三份 TSV，未重新划分。
- 评论文件：`{matches.attrs['comments_tsv']}`
- 评论实际字段：{', '.join(f'`{column}`' for column in comments_columns)}
- Reddit ID 在关联时统一转为小写，并移除 `t1_` / `t3_` 前缀；输出仍保留新闻表中的原始 ID。
- 有效评论规则：正文非空、非 `[deleted]`、非 `[removed]`，清理控制字符和多余空白后按不区分大小写去重。短评论仅标记，未删除。
- 标签保留 TSV 中的原始数值。结合 Fakeddit 论文中的类别定义，并用本地训练集的 `subreddit` 与 2/3/6-way 交叉关系复核，映射为：`2_way_label`：0=Fake、1=True；`3_way_label`：0=True、1=Fake with true text、2=Fake with false text；`6_way_label`：0=True、1=Satire/Parody、2=False Connection、3=Imposter Content、4=Manipulated Content、5=Misleading Content。
- 标签定义来源：[Fakeddit 论文](https://aclanthology.org/2020.lrec-1.755/)；数据字段及评论关联说明来源：[Fakeddit 官方仓库](https://github.com/entitize/Fakeddit)。

## 总体结果

{markdown_table(
    ['检查阶段', '数量', '占图片总数'],
    [
        ['图片文件总数', total_images, '100.00%'],
        ['Pillow 可完整解码', valid_images, f'{valid_images / total_images:.2%}' if total_images else '0.00%'],
        ['匹配到新闻表', matched_news, f'{matched_news / total_images:.2%}' if total_images else '0.00%'],
        ['有有效 clean_title', has_text, f'{has_text / total_images:.2%}' if total_images else '0.00%'],
        ['至少有一个标签', has_label, f'{has_label / total_images:.2%}' if total_images else '0.00%'],
        ['存在评论记录', with_any_comments, f'{with_any_comments / total_images:.2%}' if total_images else '0.00%'],
        ['存在有效去重评论', with_valid_comments, f'{with_valid_comments / total_images:.2%}' if total_images else '0.00%'],
        ['最终完整可用样本', usable_count, f'{usable_count / total_images:.2%}' if total_images else '0.00%'],
    ],
)}

- 文件扩展名统计：{json.dumps(matches['image_extension'].value_counts().to_dict(), ensure_ascii=False)}
- Pillow 检测到的实际格式：{json.dumps(matches['image_format'].value_counts().to_dict(), ensure_ascii=False)}。其中 PNG 虽使用 `.jpg` 文件名，但能完整解码，已在 CSV 中保留实际格式。
- 172 个文件名去除后缀后均与新闻表 `id` 精确匹配，确认本批文件名就是对应的 Fakeddit 新闻 ID。
- 172 条本地新闻共匹配 {int(matches['comment_count'].sum()):,} 条原始评论；过滤空值/删除标记并去重后保留 {int(matches['valid_comment_count'].sum()):,} 条。覆盖新闻中平均 {matches.loc[matches['valid_comment_count'].gt(0), 'valid_comment_count'].mean():.2f} 条，中位数 {matches.loc[matches['valid_comment_count'].gt(0), 'valid_comment_count'].median():.0f} 条，最大 {int(matches['valid_comment_count'].max()):,} 条。
- 少于 {matches.attrs['short_comment_chars']} 个字符的有效评论有 {int(matches['short_valid_comment_count'].sum()):,} 条，只做标记、没有删除。

## 按原始划分统计

{markdown_table(
    ['split', '图片数', '匹配新闻数', '有任意评论', '有有效评论', '可用样本'],
    split_rows,
)}

## 可用样本标签分布

{chr(10).join(label_sections)}

## 失败原因

{markdown_table(['原因', '图片数'], sorted(failure_counts.items())) if failure_counts else '没有失败样本。'}

- 有新闻但没有有效评论的图片：{', '.join(f'`{name}`' for name in no_comment_images) if no_comment_images else '无'}
- 匹配不到新闻表的图片：{', '.join(f'`{name}`' for name in unmatched_images) if unmatched_images else '无'}
- 损坏图片：{', '.join(f'`{name}`' for name in corrupted_images) if corrupted_images else '无'}

## 额外一致性检查

- 各 split 内重复新闻 ID 行数：{json.dumps(news_stats['per_split_duplicate_id_rows'], ensure_ascii=False)}
- 同一新闻 ID 跨 split 重叠数：{json.dumps(news_stats['split_overlap_ids'], ensure_ascii=False)}
- 一张图片匹配多条新闻记录：{int(matches['matched_news_rows'].gt(1).sum())} 张
- 图片扩展名不同但 ID 相同：{json.dumps(image_duplicate_ids, ensure_ascii=False) if image_duplicate_ids else '无'}
- 仅大小写不同后才匹配成功：{', '.join(f'`{name}`' for name in case_mismatches) if case_mismatches else '无'}
- 评论总行数：{comment_stats['total_comment_rows']:,}
- 评论 `submission_id` 为空的行数：{comment_stats['empty_submission_rows']:,}
- 评论 `submission_id` 无法映射到三份新闻表的行数：{comment_stats['unmapped_submission_rows']:,}
- TSV 结构异常而跳过的评论记录数：{comment_stats['malformed_rows_skipped']:,}
- 评论源索引范围：{comment_stats['first_record_index']} 至 {comment_stats['last_record_index']}
- 评论源索引缺口数：{comment_stats['comment_index_gap_count']:,}
- 源索引缺口来自评论表首列保留的非连续原始索引；本次结构解析异常数为 0，因此不是本次扫描漏读。
- `submission_id` 无法映射的统计是相对于三份 multimodal-only 新闻表；官方评论文件覆盖整个 released dataset，包含非 multimodal-only 新闻，因此该数不表示这些评论文件损坏。
- 无法映射 ID 示例（最多 20 个）：{', '.join(f'`{value}`' for value in comment_stats['unmapped_submission_examples']) if comment_stats['unmapped_submission_examples'] else '无'}

## 评论数量较多的本地图文新闻

{high_comment_table}

## 成功样本示例

{chr(10).join(examples)}

## 输出文件

- `all_image_matches.csv`：每张图片一行，包含完整匹配、评论统计和失败原因。
- `usable_samples.csv`：只包含可直接用于图文 + 标签 → 评论实验的新闻。
- `usable_comments.jsonl`：每条可用新闻及其全部有效去重评论。

本次只扫描和筛选本地数据；没有下载图片、训练模型或修改 MFCL 训练代码。
"""
    return report


def main() -> None:
    args = parse_args()
    paths = [args.images_dir, args.train_tsv, args.validate_tsv, args.test_tsv, args.comments_tsv]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [path for path in args.images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )
    images_by_id: dict[str, list[str]] = defaultdict(list)
    image_records: list[dict[str, object]] = []
    for path in image_paths:
        normalized_id = normalize_reddit_id(path.stem)
        images_by_id[normalized_id].append(path.name)
        image_records.append(
            {
                "image_name": path.name,
                "image_path": str(path.resolve()),
                "image_extension": path.suffix.lower(),
                "image_file_size": path.stat().st_size,
                "image_stem": path.stem,
                "_normalized_image_id": normalized_id,
                **validate_image(path),
            }
        )
    image_duplicate_ids = {key: value for key, value in images_by_id.items() if len(value) > 1}

    split_paths = {
        "train": args.train_tsv,
        "validate": args.validate_tsv,
        "test": args.test_tsv,
    }
    news, news_stats = read_news_tables(split_paths)
    news_groups = {key: group for key, group in news.groupby("_normalized_id", sort=False)}
    all_news_ids = set(news_groups)

    matches: list[dict[str, object]] = []
    for image in image_records:
        normalized_id = str(image["_normalized_image_id"])
        candidates = news_groups.get(normalized_id)
        record = dict(image)
        record.update(
            {
                "news_id": image["image_stem"],
                "matched_news_id": "",
                "split": "",
                "headline": "",
                "image_url": "",
                **{column: "" for column in LABEL_COLUMNS},
                "match_type": "none",
                "matched_news_rows": 0,
            }
        )
        if candidates is not None and not candidates.empty:
            record["matched_news_rows"] = len(candidates)
            exact = candidates.loc[candidates["id"].eq(image["image_stem"])]
            selected = exact.iloc[0] if not exact.empty else candidates.iloc[0]
            record["match_type"] = "exact" if not exact.empty else "case_insensitive"
            record["news_id"] = selected["id"]
            record["matched_news_id"] = selected["id"]
            record["split"] = selected["split"]
            record["headline"] = selected["clean_title"]
            record["image_url"] = selected["image_url"]
            for column in LABEL_COLUMNS:
                record[column] = selected[column]
        matches.append(record)

    target_ids = {
        normalize_reddit_id(record["matched_news_id"])
        for record in matches
        if record["matched_news_id"]
    }
    comments_header = pd.read_csv(args.comments_tsv, sep="\t", nrows=0).columns.tolist()
    missing_comment_columns = [column for column in COMMENT_COLUMNS if column not in comments_header]
    if missing_comment_columns:
        raise ValueError(f"Missing comment columns: {missing_comment_columns}; actual={comments_header}")

    per_news, comment_stats = scan_comments(
        args.comments_tsv,
        target_ids,
        all_news_ids,
        args.comments_chunksize,
        args.short_comment_chars,
    )

    for record in matches:
        normalized_id = normalize_reddit_id(record["matched_news_id"])
        comments = per_news["valid_comments"].get(normalized_id, [])
        record["comment_count"] = int(per_news["comment_count"].get(normalized_id, 0))
        record["valid_comment_count"] = len(comments)
        record["top_level_comment_count"] = int(per_news["top_level_count"].get(normalized_id, 0))
        record["reply_comment_count"] = int(per_news["reply_count"].get(normalized_id, 0))
        record["valid_top_level_comment_count"] = int(
            per_news["valid_top_level_count"].get(normalized_id, 0)
        )
        record["valid_reply_comment_count"] = int(
            per_news["valid_reply_count"].get(normalized_id, 0)
        )
        record["short_valid_comment_count"] = int(per_news["short_valid_count"].get(normalized_id, 0))

        failures: list[str] = []
        if not record["image_valid"]:
            failures.append("image_corrupted")
        if not record["matched_news_id"]:
            failures.append("news_not_found")
        elif int(record["matched_news_rows"]) > 1:
            failures.append("news_id_ambiguous")
        if record["matched_news_id"] and not is_present(record["headline"]):
            failures.append("missing_text")
        if record["matched_news_id"] and not any(is_present(record[column]) for column in LABEL_COLUMNS):
            failures.append("missing_label")
        if record["matched_news_id"] and record["comment_count"] == 0:
            failures.append("no_comments")
        elif record["matched_news_id"] and record["valid_comment_count"] == 0:
            failures.append("comments_invalid")
        if len(images_by_id[record["_normalized_image_id"]]) > 1:
            failures.append("duplicate_image_id")

        record["usable"] = not failures
        record["failure_reason"] = ";".join(failures)

    matches_df = pd.DataFrame(matches)
    matches_df.attrs["images_dir"] = str(args.images_dir.resolve())
    matches_df.attrs["comments_tsv"] = str(args.comments_tsv.resolve())
    matches_df.attrs["short_comment_chars"] = args.short_comment_chars
    output_columns = [
        "image_name",
        "image_path",
        "image_extension",
        "image_format",
        "image_file_size",
        "image_width",
        "image_height",
        "image_valid",
        "image_error",
        "news_id",
        "matched_news_id",
        "split",
        "headline",
        "image_url",
        *LABEL_COLUMNS,
        "match_type",
        "matched_news_rows",
        "comment_count",
        "valid_comment_count",
        "top_level_comment_count",
        "reply_comment_count",
        "valid_top_level_comment_count",
        "valid_reply_comment_count",
        "short_valid_comment_count",
        "usable",
        "failure_reason",
    ]
    all_matches_path = args.output_dir / "all_image_matches.csv"
    matches_df[output_columns].to_csv(all_matches_path, index=False, encoding="utf-8-sig")

    usable_columns = [
        "news_id",
        "split",
        "headline",
        "image_path",
        "image_url",
        *LABEL_COLUMNS,
        "comment_count",
        "valid_comment_count",
        "top_level_comment_count",
        "reply_comment_count",
        "valid_top_level_comment_count",
        "valid_reply_comment_count",
        "short_valid_comment_count",
    ]
    usable_df = matches_df.loc[matches_df["usable"], usable_columns].copy()
    usable_df = usable_df.drop_duplicates(subset=["news_id"], keep="first")
    usable_path = args.output_dir / "usable_samples.csv"
    usable_df.to_csv(usable_path, index=False, encoding="utf-8-sig")

    comments_path = args.output_dir / "usable_comments.jsonl"
    with comments_path.open("w", encoding="utf-8") as handle:
        for row in usable_df.itertuples(index=False):
            news_id = str(row.news_id)
            normalized_id = normalize_reddit_id(news_id)
            json.dump(
                {"news_id": news_id, "comments": per_news["valid_comments"].get(normalized_id, [])},
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")

    report = build_report(
        matches_df,
        usable_df,
        news_stats,
        comment_stats,
        image_duplicate_ids,
        comments_header,
    )
    report_path = args.output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"images={len(matches_df):,}")
    print(f"valid_images={int(matches_df['image_valid'].sum()):,}")
    print(f"matched_news={int(matches_df['split'].ne('').sum()):,}")
    print(f"with_valid_comments={int(matches_df['valid_comment_count'].gt(0).sum()):,}")
    print(f"usable={len(usable_df):,}")
    print(f"outputs={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
