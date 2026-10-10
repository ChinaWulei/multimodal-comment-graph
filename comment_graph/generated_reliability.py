"""Label-blind reliability controls for synthetic discussion nodes.

This is an experimental *augmentation* rule, not ground-truth recovery of a
reply tree. It uses frozen online text features and the supplied parent edge;
neither class labels nor validation/test outcomes enter its decisions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


RELIABILITY_MODES = {"none", "quality", "placement", "both", "confidence", "quality_confidence",
                     "ancestry_anchor"}


def refine_generated_graph(
    node_features: torch.Tensor,
    node_type: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    batch: torch.Tensor,
    mode: str,
    *,
    predicted_parent_bonus: float = 0.10,
    min_parent_similarity: float = 0.45,
    relevance_floor: float = 0.20,
    relevance_low: float = 0.35,
    relevance_high: float = 0.70,
    duplicate_start: float = 0.90,
    max_generated_depth: int = 3,
    confidence_floor: float = 0.25,
    root_fallback_percentile: float = 0.05,
    anchor_low: float = 0.75,
    anchor_high: float = 0.84,
    anchor_floor: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Reattach dubious generated edges and/or downweight dubious nodes.

    ``quality`` and ``placement`` are separate ablations. Similarities are
    normalized cosine scores in [0, 1], computed from the *same* text encoder
    used for real and generated comments. All generated parents must precede
    the child, so rewiring cannot introduce a cycle or a self-loop.
    """
    if mode not in RELIABILITY_MODES:
        raise ValueError(f"Unknown generated reliability mode: {mode}")
    if not 0 <= relevance_floor <= 1 or not 0 <= min_parent_similarity <= 1:
        raise ValueError("Reliability floors and thresholds must be in [0, 1]")
    if not 0 <= relevance_low < relevance_high <= 1:
        raise ValueError("Require 0 <= relevance_low < relevance_high <= 1")
    if not 0 <= duplicate_start < 1 or max_generated_depth < 1:
        raise ValueError("Invalid duplicate threshold or maximum generated depth")
    if not 0 <= confidence_floor <= 1 or not 0 <= root_fallback_percentile <= 1:
        raise ValueError("Invalid confidence floor or root fallback percentile")
    if not 0 <= anchor_low < anchor_high <= 1 or not 0 <= anchor_floor <= 1:
        raise ValueError("Invalid train-calibrated news-anchor settings")
    if mode == "none":
        return edge_index, edge_type, node_features.new_ones(node_features.size(0)), []
    if edge_index.shape[0] != 2 or edge_index.shape[1] != edge_type.numel():
        raise ValueError("Inconsistent edge_index and edge_type")

    normalized = F.normalize(node_features.detach().float(), dim=-1)
    reliability = node_features.new_ones(node_features.size(0))
    # Reconstruct only generated edges. Real observed edges remain untouched.
    original_parents: dict[int, int] = {}
    for edge in range(edge_type.numel()):
        parent, child = (int(edge_index[0, edge]), int(edge_index[1, edge]))
        if int(edge_type[edge]) == 1 and int(node_type[child]) == 2 and parent < child:
            if child in original_parents and original_parents[child] != parent:
                raise ValueError(f"Generated node {child} has multiple parents")
            original_parents[child] = parent
    generated = torch.nonzero(node_type == 2, as_tuple=False).flatten().tolist()
    if set(generated) != set(original_parents):
        raise ValueError("Every generated node must have one earlier parent edge")

    chosen_parents: dict[int, int] = {}
    generated_depth: dict[int, int] = {}
    chain_anchor: dict[int, float] = {}
    chain_anchor_source: dict[int, int] = {}
    diagnostics: list[dict] = []
    for child in generated:
        graph_id = int(batch[child])
        roots = torch.nonzero((batch == graph_id) & (node_type == 0), as_tuple=False).flatten().tolist()
        if len(roots) != 1:
            raise ValueError("Every graph must have exactly one news root")
        root = roots[0]
        original_parent = original_parents[child]
        if int(batch[original_parent]) != graph_id:
            raise ValueError("Generated parent crosses a news graph boundary")
        candidates = [i for i in range(root, child) if int(batch[i]) == graph_id
                      and (int(node_type[i]) != 2 or generated_depth.get(i, 0) < max_generated_depth
                           or i == original_parent)]
        if root not in candidates:
            raise ValueError("News root is not a valid parent")
        similarities = torch.matmul(normalized[candidates], normalized[child])
        scores = (similarities + 1.0) / 2.0
        original_score = float(scores[candidates.index(original_parent)]) if original_parent in candidates else None
        other_scores = scores[[i for i, candidate in enumerate(candidates) if candidate != original_parent]]
        parent_percentile = (float((other_scores <= original_score).float().mean())
                             if other_scores.numel() else 1.0)
        edge_confidence = confidence_floor + (1.0 - confidence_floor) * parent_percentile
        if mode in {"placement", "both"}:
            # Give the generator's own parent prediction a modest prior, but
            # let a substantially better semantic candidate override it.
            ranked = scores.clone()
            if original_parent in candidates:
                ranked[candidates.index(original_parent)] += predicted_parent_bonus
            selected = candidates[int(ranked.argmax())]
            if float(scores[candidates.index(selected)]) < min_parent_similarity:
                selected = root
        else:
            selected = original_parent
        if mode in {"confidence", "quality_confidence", "ancestry_anchor"} and selected != root:
            if parent_percentile <= root_fallback_percentile:
                selected = root
        chosen_parents[child] = selected
        generated_depth[child] = generated_depth.get(selected, 0) + 1

        parent_score = float(((normalized[child] * normalized[selected]).sum() + 1) / 2)
        news_score = float(((normalized[child] * normalized[root]).sum() + 1) / 2)
        previous_comments = [i for i in range(root + 1, child) if int(batch[i]) == graph_id]
        duplicate_score = (
            float(((normalized[previous_comments] @ normalized[child]).max() + 1) / 2)
            if previous_comments else 0.0
        )
        if mode in {"quality", "both", "quality_confidence", "ancestry_anchor"}:
            topical = max(news_score, parent_score)
            relevance = relevance_floor + (1 - relevance_floor) * min(
                1.0, max(0.0, (topical - relevance_low) / (relevance_high - relevance_low))
            )
            novelty = relevance_floor + (1 - relevance_floor) * min(
                1.0, max(0.0, (1.0 - duplicate_score) / (1.0 - duplicate_start))
            ) if duplicate_score > duplicate_start else 1.0
            reliability[child] = min(relevance, novelty)
        if mode in {"confidence", "quality_confidence", "ancestry_anchor"}:
            reliability[child] *= edge_confidence
        anchor_factor = 1.0
        anchor_source = None
        if mode == "ancestry_anchor":
            # Only the first generated node of a synthetic reply chain is
            # required to anchor to the news. Later short, elliptical replies
            # inherit that trust instead of each being judged against a title.
            if int(node_type[selected]) == 2:
                anchor_factor = chain_anchor[selected]
                anchor_source = chain_anchor_source[selected]
            else:
                anchor_factor = anchor_floor + (1.0 - anchor_floor) * min(
                    1.0, max(0.0, (news_score - anchor_low) / (anchor_high - anchor_low))
                )
                anchor_source = child
            chain_anchor[child] = anchor_factor
            chain_anchor_source[child] = anchor_source
            reliability[child] *= anchor_factor
        diagnostics.append({
            "node_index": child, "original_parent": original_parent,
            "selected_parent": selected, "rewired": selected != original_parent,
            "original_parent_similarity": original_score,
            "selected_parent_similarity": parent_score,
            "news_similarity": news_score,
            "max_previous_comment_similarity": duplicate_score,
            "original_parent_percentile": parent_percentile,
            "edge_confidence": edge_confidence,
            "news_anchor_factor": anchor_factor,
            "anchor_source_node": anchor_source,
            "text_reliability": float(reliability[child]),
        })

    if mode not in {"placement", "both", "confidence", "quality_confidence", "ancestry_anchor"}:
        return edge_index, edge_type, reliability, diagnostics
    new_pairs: list[tuple[int, int]] = []
    new_types: list[int] = []
    for edge in range(edge_type.numel()):
        source, target = (int(edge_index[0, edge]), int(edge_index[1, edge]))
        relation = int(edge_type[edge])
        if relation == 0:
            new_pairs.append((source, target))
            new_types.append(0)
    for child in generated:
        parent = chosen_parents[child]
        new_pairs.extend(((parent, child), (child, parent)))
        new_types.extend((1, 1))
    output_edges = torch.tensor(new_pairs, dtype=torch.long, device=edge_index.device).t().contiguous()
    output_types = torch.tensor(new_types, dtype=edge_type.dtype, device=edge_type.device)
    return output_edges, output_types, reliability, diagnostics
