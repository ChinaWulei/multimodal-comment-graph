"""Synthetic resumable-annotation UI smoke; no research labels are created."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.annotate_generated_quality import main  # noqa: E402
from comment_graph.validate_generated_quality_annotations import load_unique  # noqa: E402


def case(index: int) -> dict:
    return {
        "annotation_id": f"fixture_{index}", "news_id": f"fixture_news_{index}",
        "official_split": "train", "source_group": "fixture", "generated_position": 1,
        "headline": f"Synthetic headline {index}",
        "image_path": f"synthetic_image_{index}.png", "parent_node_index": 1,
        "parent_node_type": "real_comment",
        "parent_content": "Synthetic parent", "recent_ancestors": [],
        "generated_comment": "Synthetic generated comment",
        "content_relevance": None, "parent_fit": None,
        "unsupported_specific_claim": None,
    }


def invoke(reference: Path, output: Path, answers: list[str], limit: int) -> None:
    arguments = ["annotate_generated_quality.py", "--reference-jsonl", str(reference),
                 "--output-jsonl", str(output), "--limit", str(limit)]
    with patch.object(sys, "argv", arguments), patch("builtins.input", side_effect=answers):
        main()


def main_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="quality_annotation_smoke_") as directory:
        reference = Path(directory) / "reference.jsonl"
        output = Path(directory) / "filled.jsonl"
        reference.write_text("\n".join(json.dumps(case(i)) for i in (1, 2)) + "\n",
                             encoding="utf-8")
        invoke(reference, output, ["r", "g", "n", "first"], 1)
        first = load_unique(output)
        assert first["fixture_1"]["annotation_status"] == "reviewed"
        assert first["fixture_2"]["content_relevance"] is None
        invoke(reference, output, ["o", "w", "y", "second"], 1)
        second = load_unique(output)
        assert second["fixture_1"] == first["fixture_1"]  # Resume did not overwrite.
        assert second["fixture_2"]["annotation_status"] == "reviewed"
        assert second["fixture_2"]["unsupported_specific_claim"] is True
        assert load_unique(reference)["fixture_1"]["content_relevance"] is None
    print("Synthetic annotation save/resume smoke passed; no human labels were created.")


if __name__ == "__main__":
    main_smoke()
