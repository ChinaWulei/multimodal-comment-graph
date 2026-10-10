from __future__ import annotations

import json
import string
from typing import Any


STRUCTURE_INPUT_FIELDS = {"image_path", "news_text", "observed_nodes"}
STRUCTURE_SYSTEM_PROMPT = (
    "Generate one plausible Reddit reply-tree node for the supplied news post and image. "
    "Return only the required JSON object. Do not mention labels, fact-checking, or this task."
)


def load_structure_template(path) -> str:
    template = path.read_text(encoding="utf-8").strip()
    fields = {field for _, field, _, _ in string.Formatter().parse(template) if field}
    if fields != {"news_text", "observed_nodes"}:
        raise ValueError(
            "Structured prompt placeholders must be exactly ['news_text', 'observed_nodes']; "
            f"found {sorted(fields)}"
        )
    lowered = template.lower()
    forbidden = ["{label", "2_way_label", "3_way_label", "6_way_label", "authenticity"]
    if any(token in lowered for token in forbidden):
        raise ValueError("Structured prompt contains a forbidden label condition")
    return template


def validate_structure_input(model_input: dict[str, Any]) -> None:
    if set(model_input) != STRUCTURE_INPUT_FIELDS:
        raise ValueError("Structured model_input contains fields outside image, news text, and nodes")
    if not str(model_input["image_path"]).strip() or not str(model_input["news_text"]).strip():
        raise ValueError("Structured model_input is missing image path or news text")
    nodes = model_input["observed_nodes"]
    if not isinstance(nodes, list):
        raise ValueError("observed_nodes must be a list")
    indices = []
    for node in nodes:
        if set(node) != {"node_index", "parent_node_index", "content"}:
            raise ValueError("Observed nodes must contain only index, parent index, and content")
        node_index = int(node["node_index"])
        parent_index = int(node["parent_node_index"])
        if node_index < 1 or parent_index < 0 or parent_index >= node_index:
            raise ValueError("Observed tree has an invalid parent-before-child edge")
        if not str(node["content"]).strip():
            raise ValueError("Observed tree contains an empty comment")
        indices.append(node_index)
    if indices != list(range(1, len(nodes) + 1)):
        raise ValueError("Observed node indices must be contiguous from 1")


def render_structure_prompt(template: str, model_input: dict[str, Any]) -> str:
    validate_structure_input(model_input)
    nodes_json = json.dumps(
        model_input["observed_nodes"], ensure_ascii=False, separators=(",", ":")
    )
    return template.format(news_text=str(model_input["news_text"]).strip(), observed_nodes=nodes_json)


def build_structure_messages(
    template: str,
    model_input: dict[str, Any],
    target_json: str | None = None,
    system_prompt: str = STRUCTURE_SYSTEM_PROMPT,
    allowed_parent_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    prompt = render_structure_prompt(template, model_input)
    if allowed_parent_indices is not None:
        allowed = sorted({int(index) for index in allowed_parent_indices})
        if not allowed or any(index < 0 or index > len(model_input["observed_nodes"]) for index in allowed):
            raise ValueError("Invalid allowed parent-node index set")
        # This is an inference-time decoding constraint, not an added data
        # field: it exposes only structural depth feasibility and never a news
        # label.  The model still predicts the attachment among these nodes.
        prompt += (
            "\n\nFor this generation step, select parent_node_index from this "
            f"allowed set only: {allowed}."
        )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    if target_json is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target_json.strip()}]}
        )
    return messages


def parse_structure_output(text: str, max_parent_index: int) -> dict[str, Any]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json: {exc.msg}") from exc
    if not isinstance(value, dict) or set(value) != {"parent_node_index", "content"}:
        raise ValueError("schema_mismatch")
    parent = value["parent_node_index"]
    if isinstance(parent, bool) or not isinstance(parent, int) or parent < 0 or parent > max_parent_index:
        raise ValueError("invalid_parent_node_index")
    if not isinstance(value["content"], str):
        raise ValueError("content_must_be_string")
    content = value["content"].strip()
    if not content:
        raise ValueError("empty_content")
    return {"parent_node_index": parent, "content": content}
