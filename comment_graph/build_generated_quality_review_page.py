"""Build a self-contained, offline human review page for the train-only queue.

The page never calls a server. It embeds verified local news images, but
exports only the original annotation records and human-entered fields.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-jsonl", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate this derived HTML page from the immutable reference queue")
    args = parser.parse_args()
    if args.output_html.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite review page: {args.output_html}")
    rows = [json.loads(line) for line in args.queue_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [str(row["annotation_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(row["official_split"] != "train" for row in rows):
        raise ValueError("Review page requires a unique official-train annotation queue")
    if any(row.get("content_relevance") is not None or row.get("parent_fit") is not None
           for row in rows):
        raise ValueError("Embed only the unannotated reference queue")
    forbidden = {"2_way_label", "3_way_label", "6_way_label", "label", "veracity",
                 "reranker_news_comment", "reranker_parent_comment"}
    if any(forbidden & set(row) for row in rows):
        raise ValueError("Reference queue unexpectedly contains news labels or model scores")
    photos = {}
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}
    for row in rows:
        path = Path(row["image_path"])
        data = path.read_bytes()
        with Image.open(path) as image:
            image.verify()
            image_format = image.format
        if image_format not in mime:
            raise ValueError(f"Unsupported verified image format: {path}/{image_format}")
        photos[row["annotation_id"]] = (
            f"data:{mime[image_format]};base64,{base64.b64encode(data).decode('ascii')}")
    queue_sha = hashlib.sha256(args.queue_jsonl.read_bytes()).hexdigest()
    template = Path(__file__).with_name("quality_review_template.html").read_text(encoding="utf-8")
    queue_b64 = base64.b64encode(json.dumps(rows, ensure_ascii=False).encode("utf-8")).decode("ascii")
    photos_b64 = base64.b64encode(json.dumps(photos).encode("utf-8")).decode("ascii")
    result = template.replace("__QUEUE_B64__", queue_b64).replace(
        "__PHOTOS_B64__", photos_b64).replace("__QUEUE_HASH__", queue_sha)
    if "__QUEUE_B64__" in result or "__PHOTOS_B64__" in result or "__QUEUE_HASH__" in result:
        raise RuntimeError("Template substitution did not finish")
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(result, encoding="utf-8")
    print(json.dumps({"page": str(args.output_html.resolve()), "cases": len(rows),
                      "embedded_images": len(photos), "queue_sha256": queue_sha,
                      "page_mib": round(args.output_html.stat().st_size / (1024 * 1024), 2),
                      "network_requests_required": False,
                      "news_labels_embedded": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
