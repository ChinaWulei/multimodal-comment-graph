"""Label-blind, within-batch mismatched-comment fallback regularization.

Inspired by comment robustness research, NOT a reproduction of CogniDir
(called AdComment in its initial preprint).
Random news mismatches are weak perturbations, not certified irrelevant comments
or an adaptive adversarial attack. No new text or truth labels are generated.
"""
import torch
import torch.nn.functional as F


ROBUSTNESS_MODES = {"none", "mismatch_fallback"}


def mismatch_generated_features(features, view, generator):
    """Swap only generated embeddings, preserving roots, types and node counts.

    The RNG is a dedicated CPU generator. Donors must have a different news ID;
    there are no class/label-dependent choices. If no donor exists, skip the news.
    """
    size = view.root_input_ids.size(0)
    rows = [(view.batch.eq(i) & view.node_type.eq(2)).nonzero().flatten() for i in range(size)]
    available = [i for i in range(size) if rows[i].numel()]
    source = torch.arange(features.size(0), device=features.device)
    donors = [-1] * size
    for target in available:
        candidates = [i for i in available if str(view.news_id[i]) != str(view.news_id[target])]
        if not candidates:
            continue
        donor = candidates[int(torch.randint(len(candidates), (1,), generator=generator))]
        order = torch.randperm(rows[donor].numel(), generator=generator).to(features.device)
        positions = torch.arange(rows[target].numel(), device=features.device) % order.numel()
        source[rows[target]] = rows[donor][order[positions]]
        donors[target] = donor
    changed = features[source]
    valid = torch.tensor([i >= 0 for i in donors], device=features.device, dtype=torch.bool)
    return changed, valid, donors


def mismatch_forward(model, output, generator):
    """Reuse encoded content: no second base pass, BN updates or image encoding."""
    context = output.get("generated_context")
    if context is None:
        return None
    features, valid, donors = mismatch_generated_features(context["features"], context["view"], generator)
    if not valid.any():
        return None
    # Treat the base as a fixed fallback target for this auxiliary loss. Even
    # with unfrozen encoders, the mismatch branch cannot change its teacher.
    with torch.autocast(device_type=features.device.type, enabled=False):
        correction = model.generated_branch(
            features.detach().float(), context["view"], context["base_features"].detach().float(),
            context["text"].detach().float(), context["image"].detach().float(),
            model.graph_structure, apply_dropout=False,
        )
        base = output["global_logits"].detach().float()
        noisy = base + correction["logit_delta"]
        divergence = F.kl_div(noisy.log_softmax(-1), base.softmax(-1), reduction="none").sum(-1)
        loss = divergence[valid].mean().clamp_min(0)
    return {"loss": loss, "logits": noisy, "valid": valid, "donors": donors,
            "correction": correction}


def add_mismatch_loss(model, output, clean_loss, generator, weight):
    if weight < 0:
        raise ValueError("mismatch_weight must be nonnegative")
    zero = clean_loss.new_zeros(())
    if weight == 0:
        return clean_loss, zero, 0
    noisy = mismatch_forward(model, output, generator)
    if noisy is None:
        return clean_loss, zero, 0
    return clean_loss + weight * noisy["loss"], noisy["loss"], int(noisy["valid"].sum())
