# Comment-guided local correction inspired by GAHR

Source: Shen et al., *Rethinking Multimodal Fake News Detection in the Generative AI
Era*, arXiv:2609.36850v1, supplied PDF, especially pp. 6-10, 12, 14, 16.

## What the paper contributes

GAHR uses RoBERTa, DINOv2 and CLIP representations. Global bidirectional attention
produces a global veracity logit; query-guided pooling extracts local evidence.
Its relation operator is R(u,v)=[u,v,|u-v|,u*v] (Eq.25). A sample-specific gate
adds a local logit correction (Eqs.29-30). Generativity supervision is an optional
auxiliary task when modality-specific annotations exist. The reported veracity
comparison in section 5.4 does not use generativity labels or predictions.

The paper reports five-seed experiments and event-cluster splits when no official
split exists. Table 3 supports local correction on the paper's datasets, not CHECKED.
No improvement in our project follows automatically from those results.

## Our adaptation and differences

- Preserve the current MFCL image/text + dynamic comment GNN global classifier.
- Use individual real/generated comments to query actual BERT news token vectors
  and ResNet50 spatial cells (49 cells at the current 224-pixel input size).
- Project tokens, regions and comments to 128 dimensions with Linear/LayerNorm/GELU.
- Mask padded tokens; temperature controls attention concentration.
- Form comment/text, comment/image, and attended text/image relation features.
- Compute separate real/generated reliability scores and source summaries.
  Pool with sum(weight*evidence)/source_count, so rejecting every comment does not
  cancel out through division by the sum of weights. These scores measure learned
  task utility, not calibrated factual correctness or human-rated quality.
- Fuse both source summaries into local evidence. A per-news gate, initialized
  near 0.1, controls a two-class residual correction:
  `final_logits = global_logits + sigmoid(gate) * local_logits`.
- Train with the existing weighted two-class cross entropy on final logits.
  There is no added contrastive loss, AIGC loss, retrieval, or generator training.
- Root-only samples have exactly zero local correction, including in mixed batches.

This is a GAHR-inspired adaptation, not a reproduction. It does not use CLIP,
DINOv2, the paper's bidirectional global stage, or pretrained cross-modal alignment.
CHECKED does not provide news-text/image generativity labels. Known generated
comment provenance must not be substituted for news/image generativity labels.
There is no claim that semantic agreement establishes factual truth.

## Training (PowerShell, project root)

Activate `multimodal-fakenews`. No new pretrained downloads are required.
The recommended first comparison holds graph and pooling fixed at dynamic + mean.

```powershell
python .\comment_graph\smoke_local_correction.py --config .\comment_graph\config.yaml

# Full training, validation-selected checkpoint, then test metrics.
python .\comment_graph\train.py `
  --config .\comment_graph\config.yaml `
  --mode real_generated `
  --graph-structure dynamic_cross_modal `
  --pooling-mode mean `
  --decision-mode global_local_correction `
  --seed 42 `
  --allow-formal-training `
  --run-test `
  --summary-json .\comment_graph\outputs\local_correction_seed42_results.json

# Matched control without local correction, using the same current runner.
python .\comment_graph\train.py `
  --config .\comment_graph\config.yaml `
  --mode real_generated `
  --graph-structure dynamic_cross_modal `
  --pooling-mode mean `
  --decision-mode concat `
  --seed 42 `
  --allow-formal-training `
  --run-test `
  --summary-json .\comment_graph\outputs\concat_seed42_results.json
```

The new checkpoint is
`outputs/checked_mfcl_real_generated_dynamic_cross_modal_mean_global_local_correction_seed42.pt`.
Mode/seed suffixes preserve earlier checkpoints; rerunning the exact same configuration
and seed writes to the same named checkpoint. Existing trained checkpoints are not used
to initialize this run. Qwen comments are read from existing JSONLs.

Use `--mode real_only` with the same decision mode for the generated-comment control.
`--pooling-mode quality_gated_attention` is compatible too, but it is an additional
combination experiment rather than the clean first comparison. Keep both switches
explicit. Options are documented in `config.yaml` under `local_correction`.

Summary JSON includes per-news final/global probabilities, correction gate, two logit
deltas, AUC and confusion matrix, enabling inspection beyond rounded F1 scores.
The model also returns token/region attention maps and comment reliability values.
Labels do not enter the local head. Checkpoint metadata now records the resolved config,
seed and decision mode.

## Evaluation changes and limitations

The runner now computes dataset loss as sum(weighted CE)/sum(target class weights).
Previously it averaged batch weighted-mean losses by batch size, which did not equal
the dataset weighted CE. Historical loss values therefore are not directly comparable.
The objective used for backpropagation and the class prediction metrics are unchanged.
Data-loader RNG is now independent of model initialization; rerun the matched control
with this runner for a fair comparison rather than only comparing old logs.

CHECKED currently has 118 test news with only four fake samples. The same test set has
already informed architecture changes in this conversation, so subsequent scores are
exploratory. Select settings on validation data, run multiple seeds, and use a fresh
held-out/event-grouped evaluation before making a generalization claim. The current
split is preserved for this implementation; it is not the paper's event-group split.

## Executed verification

`reports/smoke_local_correction.json` records eight actual train news (4 real, 4 fake),
two AMP optimizer steps, finite nonzero GNN/edge/attention/correction gradients,
parameter updates and evaluation. No test news were used for implementation tuning.
Checks cover padding invariance, node permutation, source rejection amplitude, missing
comments, image/text sensitivity, local checkpoint round-trip, and label independence.
The synthetic checks validate mechanics only; they do not substitute for real samples.

Technical smoke-test loss or synthetic sensitivity is not evidence of improved detection.
