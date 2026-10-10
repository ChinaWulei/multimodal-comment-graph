"""Static offline-page audit; does not automate or bypass a browser."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image


class Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.active = None
        self.scripts: dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            attributes = dict(attrs)
            if "src" in attributes:
                raise ValueError("Offline page unexpectedly loads an external script")
            self.active = attributes.get("id", "application")
            self.scripts.setdefault(self.active, "")

    def handle_data(self, data):
        if self.active is not None:
            self.scripts[self.active] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.active = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-jsonl", type=Path, required=True)
    parser.add_argument("--page-html", type=Path, required=True)
    args = parser.parse_args()
    page = args.page_html.read_text(encoding="utf-8")
    if "__QUEUE_B64__" in page or "__PHOTOS_B64__" in page or "__QUEUE_HASH__" in page:
        raise ValueError("Unsubstituted template marker")
    if "http://" in page or "https://" in page:
        raise ValueError("Unexpected network URL in offline review page")
    parsed = Scripts()
    parsed.feed(page)
    queue = json.loads(base64.b64decode(parsed.scripts["queue-data"]))
    photos = json.loads(base64.b64decode(parsed.scripts["photo-data"]))
    expected = [json.loads(line) for line in args.queue_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if queue != expected or set(photos) != {str(row["annotation_id"]) for row in expected}:
        raise ValueError("Embedded queue or image coverage differs from reference")
    for value in photos.values():
        prefix, encoded = value.split(",", 1)
        if not prefix.startswith("data:image/"):
            raise ValueError("Non-image embedded payload")
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            image.verify()
    expected_hash = hashlib.sha256(args.queue_jsonl.read_bytes()).hexdigest()
    if f'const queueHash = "{expected_hash}";' not in parsed.scripts["application"]:
        raise ValueError("Queue hash in application differs from source")
    node = shutil.which("node")
    syntax_check = "skipped: Node.js not installed"
    if node:
        with tempfile.TemporaryDirectory(prefix="quality_page_syntax_") as directory:
            script = Path(directory) / "review.js"
            script.write_text(parsed.scripts["application"], encoding="utf-8")
            subprocess.run([node, "--check", str(script)], check=True)
        syntax_check = "passed"
    print(json.dumps({"cases": len(queue), "verified_embedded_images": len(photos),
                      "queue_hash_matches": True, "external_urls": 0,
                      "javascript_syntax": syntax_check,
                      "browser_render_test": "not run: file URL blocked by browser policy"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
