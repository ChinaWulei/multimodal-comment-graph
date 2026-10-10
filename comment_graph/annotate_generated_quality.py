"""Resumable human annotation of generated comment relevance and parent fit.

The interface never displays news veracity labels or model proxy scores.
It writes a separate output JSONL after every reviewed case, so Ctrl+C is
safe. This script does not auto-fill labels or train any model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.validate_generated_quality_annotations import IMMUTABLE, load_unique  # noqa: E402

CONTENT = {"r": "relevant", "p": "partly_relevant", "o": "off_topic", "?": None}
PARENT = {"g": "good", "u": "uncertain", "w": "wrong", "?": None}
CLAIM = {"y": True, "n": False, "?": None}


def ask(prompt: str, choices: dict[str, object]) -> object:
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return choices[answer]
        print("请输入其中一个选项 / Please choose one listed option.")


def save_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=path.stem + ".", suffix=".tmp",
                                         delete=False)
    temp = Path(handle.name)
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Maximum cases this session; 0 means all remaining")
    parser.add_argument("--revisit-unresolved", action="store_true")
    parser.add_argument("--open-image", action="store_true",
                        help="On Windows, open each image in the default viewer (opt-in).")
    args = parser.parse_args()
    if args.limit < 0 or args.reference_jsonl.resolve() == args.output_jsonl.resolve():
        raise ValueError("limit must be nonnegative and output must differ from the reference queue")
    original = load_unique(args.reference_jsonl)
    if any(row["official_split"] != "train" for row in original.values()):
        raise ValueError("Human annotation must use official training news only")
    if args.output_jsonl.exists():
        saved = load_unique(args.output_jsonl)
        if set(saved) != set(original):
            raise ValueError("Saved progress IDs differ from reference queue")
        for key, row in saved.items():
            if any(row.get(field) != original[key][field] for field in IMMUTABLE):
                raise ValueError(f"Saved progress changed source fields: {key}")
        rows = [saved[key] for key in original]
    else:
        rows = [dict(row) for row in original.values()]
    reviewed_this_session = 0
    try:
        for number, row in enumerate(rows, 1):
            status = row.get("annotation_status")
            if (status is None and row.get("content_relevance") is not None
                    and row.get("parent_fit") is not None):
                # Respect labels entered directly in JSONL before this CLI
                # was used; never prompt to overwrite them by default.
                status = "reviewed"
            if status == "reviewed" or (status == "unresolved" and not args.revisit_unresolved):
                continue
            if args.limit and reviewed_this_session >= args.limit:
                break
            print("\n" + "=" * 72)
            print(f"{number}/{len(rows)}  {row['annotation_id']}")
            print(f"新闻 / News: {row['headline']}")
            print(f"图片 / Image: {row['image_path']}")
            if args.open_image:
                if not hasattr(os, "startfile"):
                    raise RuntimeError("--open-image is supported only on Windows")
                os.startfile(row["image_path"])  # type: ignore[attr-defined]
            print(f"拟连接父节点 / Proposed parent ({row['parent_node_type']}): {row['parent_content']}")
            for ancestor in row["recent_ancestors"][1:]:
                print(f"上层讨论 / Earlier context ({ancestor['node_type']}): {ancestor['content']}")
            print(f"生成评论 / Generated: {row['generated_comment']}")
            row["content_relevance"] = ask(
                "内容相关性 r=相关 p=部分相关 o=偏题 ?=无法判断: ", CONTENT)
            row["parent_fit"] = ask(
                "父节点位置 g=合适 u=不确定 w=明显不合适 ?=无法判断: ", PARENT)
            row["unsupported_specific_claim"] = ask(
                "是否有无依据的具体事实/身份/来源 y=有 n=未见 ?=无法判断: ", CLAIM)
            row["review_notes"] = input("备注（可直接回车）/ Notes: ").strip()
            row["annotation_status"] = ("reviewed" if row["content_relevance"] is not None
                                        and row["parent_fit"] is not None else "unresolved")
            save_atomic(args.output_jsonl, rows)
            reviewed_this_session += 1
            print(f"已保存 {reviewed_this_session} 条（本次）/ Saved.")
    except KeyboardInterrupt:
        print("\n已安全停止；此前完成的标注已逐条保存。")
    complete = sum(row.get("annotation_status") == "reviewed" or
                   (row.get("content_relevance") is not None and row.get("parent_fit") is not None)
                   for row in rows)
    unresolved = sum(row.get("annotation_status") == "unresolved" for row in rows)
    print(f"完成 {complete}/{len(rows)}，待复核 {unresolved}；文件：{args.output_jsonl}")


if __name__ == "__main__":
    main()
