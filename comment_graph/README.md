# CHECKED comment graph for MFCL

## Source-isolated generated-comment augmentation

New optional `--decision-mode source_isolated --pooling-mode mean` keeps the
news/real-comment GCN separate from the news/generated-comment GCN, with supervised
base classification and a bounded, droppable generated residual. Old modes and
checkpoints remain available. See [SOURCE_ISOLATED.md](SOURCE_ISOLATED.md) for
the architecture, checks, limitations, and complete PowerShell commands.

## GAHR-inspired local correction

An optional `--decision-mode global_local_correction` adds comment-guided attention
over news tokens and image regions, followed by a gated logit correction. See
[GAHR_NOTES.md](GAHR_NOTES.md) for the paper mapping, differences, smoke test and
training/test commands. `concat` preserves the existing decision path.

This module leaves the original MFCL scripts unchanged. It reads CHECKED news/images,
real comments, and already-generated comments. It supports both the original undirected
star graph and a feature-driven dynamic cross-modal graph.

Node types are `0=news`, `1=real_comment`, and `2=generated_comment`. Every comment is
connected directly to the news root in both directions in `star` mode. The existing MFCL
online text encoder encodes all node texts, the existing two-layer GCN produces a 768-dimensional
graph vector, and the existing MFCL text/image module produces a 1536-dimensional
multimodal vector. Their concatenation is classified with cross entropy.

Supported modes:

- `baseline`: text/image only; graph vector is zero.
- `real_only`: news plus real comments.
- `generated_only`: news plus generated comments.
- `real_generated`: news plus both comment sources.

Graph switches are independent of the four data modes:

- `graph.structure_mode`: `star` or `dynamic_cross_modal`.
- `graph.pooling_mode`: `mean`, `dual_modal_attention`, or
  `quality_gated_attention`.

`dynamic_cross_modal` always preserves news-comment connectivity and additionally selects
the top-k comment neighbours. It predicts differentiable edge weights from the current
node state, news-text feature and image feature, then rebuilds the graph before each GCN
layer. `dual_modal_attention` independently pools comments with news-text and image
queries and learns a gate between both summaries. Truth labels are never inputs to graph
construction or attention.

`quality_gated_attention` assigns a learnable 0--1 quality score only to generated
comments. Real comments and the news root keep quality 1. The score gates generated node
features, dynamic edge weights, quality-weighted graph pooling, and a low-rank
source-separated residual attention. It is trained only through the downstream
classification loss; test labels are never model inputs.

Run the structural smoke test:

```powershell
python .\comment_graph\smoke_test.py `
  --config .\comment_graph\config.yaml `
  --graph-structure dynamic_cross_modal `
  --pooling-mode dual_modal_attention
```

The completed local smoke-test results are documented in `reports/smoke_test.md` and
`reports/smoke_test.json`. This is not an accuracy result: the currently generated subset
contains only real validation samples.

Generated-comment coverage currently available for model development:

- `generated_train.jsonl`: 939/939 news, five comments per news.
- `generated_validate.jsonl`: 118/118 news, five comments per news.
- `generated_test.jsonl`: intentionally deferred until model selection is frozen.

After smoke-test review, train one ablation at a time:

```powershell
python .\comment_graph\train.py --config .\comment_graph\config.yaml --mode real_only --allow-formal-training
python .\comment_graph\train.py --config .\comment_graph\config.yaml --mode real_generated --allow-formal-training
```

Train the new full model explicitly:

```powershell
python .\comment_graph\train.py `
  --config .\comment_graph\config.yaml `
  --mode real_generated `
  --graph-structure dynamic_cross_modal `
  --pooling-mode dual_modal_attention `
  --allow-formal-training
```

Graph and pooling names are included in new checkpoint filenames, so the prior star-graph
checkpoints are not overwritten.

Recommended controlled ablations (run separately):

```powershell
# Original graph pooling control
python .\comment_graph\train.py --config .\comment_graph\config.yaml --mode real_generated --graph-structure star --pooling-mode mean --allow-formal-training

# Add only text/image-aware comment pooling
python .\comment_graph\train.py --config .\comment_graph\config.yaml --mode real_generated --graph-structure star --pooling-mode dual_modal_attention --allow-formal-training

# Add only dynamic propagation
python .\comment_graph\train.py --config .\comment_graph\config.yaml --mode real_generated --graph-structure dynamic_cross_modal --pooling-mode mean --allow-formal-training

# Full model: attention plus dynamic propagation
python .\comment_graph\train.py --config .\comment_graph\config.yaml --mode real_generated --graph-structure dynamic_cross_modal --pooling-mode dual_modal_attention --allow-formal-training
```

Quality-gated generated-comment training and test:

```powershell
python .\comment_graph\train.py `
  --config .\comment_graph\config.yaml `
  --mode real_generated `
  --graph-structure dynamic_cross_modal `
  --pooling-mode quality_gated_attention `
  --allow-formal-training `
  --run-test `
  --summary-json .\comment_graph\outputs\quality_gated_attention_results.json
```

The corresponding checkpoint is
`outputs/checked_mfcl_real_generated_dynamic_cross_modal_quality_gated_attention.pt`.

## Fakeddit 的真实回复树与生成边（正式子集）

`config_fakeddit_structure.yaml` 是独立入口，不读取 CHECKED 文件，也不改动原始
MFCL 代码。它使用三分类 `3_way_label` 计算分类损失；标签不会输入评论生成器或参与
构图。`observed_tree` 保留 Fakeddit `parent_id` 的真实边与模型生成的 parent 边；`star`
把相同节点全部连到新闻根节点，作为唯一拓扑差异的对照。

先在正式图片、树和生成图齐全后运行结构不变量检查：

```powershell
python .\comment_graph\smoke_fakeddit_structure.py
```

模型选择阶段只用官方验证集。下面两组 `real_only` 对照使用相同节点和随机种子，
仅比较真实树边与根连接边；不要在此阶段附加 `--run-test`：

```powershell
python .\comment_graph\train.py `
  --config .\comment_graph\config_fakeddit_structure.yaml `
  --mode real_only --graph-structure star --seed 42 `
  --allow-formal-training

python .\comment_graph\train.py `
  --config .\comment_graph\config_fakeddit_structure.yaml `
  --mode real_only --graph-structure observed_tree --seed 42 `
  --allow-formal-training
```

确定结构与超参数后，才将生成图文件配置为完整覆盖 train/validate/test 并执行一次
预先固定的测试评估。这样测试集不会被反复用于挑选方法。

Do not pass `--run-test` while tuning. Use it only once the configuration is frozen.

### Relation-aware generated-edge ablation

`typed_rgcn` keeps the identical node set and topology as `observed_tree`, but
uses a separate message-passing relation for Fakeddit observed reply edges
(`edge_type=0`) and generator-predicted attachment edges (`edge_type=1`). It
does not introduce labels into the graph. Run it only after the structured
generator has produced complete train/validate graph files; choose against the
`legacy_gcn` control on validation before one frozen test evaluation.

```powershell
python .\comment_graph\smoke_typed_graph.py

python .\comment_graph\train.py `
  --config .\comment_graph\config_fakeddit_structure.yaml `
  --mode real_generated `
  --graph-structure observed_tree `
  --message-passing typed_rgcn `
  --pooling-mode mean `
  --seed 42 `
  --allow-formal-training
```

## Optional comment-mismatch robustness pilot (2026-10-04)

See [recent research, implementation, actual pilot results and full commands](RECENT_RESEARCH_20261004.md).
The new `--robustness-mode mismatch_fallback` requires `--decision-mode source_isolated`
and `--pooling-mode mean`. It is disabled by default. The first paired pilot did **not**
show a performance gain; keep the control and inspect comment sensitivity, not only F1.

```powershell
python .\comment_graph\smoke_robustness.py
python .\comment_graph\run_robustness_pilot.py --train-size 128 --epochs 2 --seed 42
```

The pilot runs both arms with matching initialization and train order, never loads
the test split, and writes diagnostics to a fresh timestamped output directory.

## Full-train validation comparisons

See [the predeclared comparison protocol](VALIDATION_SEARCH_PROTOCOL.md).
`--sampling balanced` uses replacement sampling on training news and unweighted
training CE; validation retains its original class distribution and common CE
weights. Oversampling does not create independent minority examples.

```powershell
python .\comment_graph\run_validation_search.py --seeds 42 --arms control mismatch balanced --epochs 20 --patience 5
```

All arms, logs, summaries and checkpoints go into a fresh timestamped directory.
This runner never evaluates test. Use `audit_generated_contribution.py` on a
completed source-isolated checkpoint for clean/removal/repetition/mismatch checks.

## MTS fusion reproduction and local Weibo support

See [method fidelity, actual dataset audit, limitations and commands](MTS_REPRODUCTION.md).
`--mm-fusion mts` enables the complete eight-layer polynomial fusion core;
the default `mfcl` path remains unchanged. `config_weibo.yaml` supports the
audited local image/text CSV data in baseline mode only, without invented comments.

```powershell
python .\comment_graph\run_mts_comparison.py --dataset weibo --seeds 42 --epochs 10 --patience 3
```

## Adapted MIMoE interaction routing for Weibo

The authors' public [MIMoE-FND repository](../third_party/MIMoE-FND) is pinned
at commit `002e747829dffcb2925803bb617cd1f9dd2de20c`. Its exact training entry
requires an unavailable MAE checkpoint, ChineseCLIP and a different spreadsheet
dataset interface, so it cannot be applied directly to the audited local Weibo
CSV without changing the backbone and data protocol. `mimoe_fusion.py` instead
adapts its central mechanism to the already-extracted MFCL text/image vectors:

1. text/image heads form JSD-based prediction agreement;
2. paired projections form a semantic-agreement signal;
3. the four signal combinations supervise a four-way router; and
4. route-weighted, condition-specific experts emit a 1536-dimensional `h_mm`.

The downstream MFCL graph placeholder and classifier retain the same 2304-wide
input as the MFCL control. This is an adapted-core experiment, **not** an
end-to-end reproduction of the paper's MAE/ChineseCLIP score. The differentiable
expert-balance regularizer and single-applied router-z weight are intentional
stability corrections relative to the released code and are recorded in the
source and experiment plan.

First verify routing and gradients on eight real training news items (no test
split is read):

```powershell
python .\comment_graph\smoke_mimoe.py `
  --output .\comment_graph\reports\smoke_mimoe_20261005.json
```

Then run the predeclared, paired validation-only comparison. It writes an
immutable experiment plan, code hashes, logs, checkpoints and summaries to a
fresh directory; it never loads the test split:

```powershell
python .\comment_graph\run_mimoe_comparison.py `
  --seeds 42 `
  --epochs 10 `
  --patience 3
```

After the validation configuration and seeds are frozen, evaluate every paired
checkpoint exactly once on test. This command does not retrain and will refuse
a validation summary that already contains test metrics:

```powershell
python .\comment_graph\run_frozen_mimoe_test.py `
  --validation-runs .\comment_graph\outputs\mimoe_weibo_20261005_145720 `
  --output-dir .\comment_graph\outputs\mimoe_weibo_20261005_145720_test_once
```

## Label-free entity evidence with MFCL and MIMoE

`prepare_weibo_entity_evidence.py` verifies that the DDU and MFCL raw Weibo
CSVs are byte-identical, then rebuilds evidence after this project's image
audit. The candidate memory is only current valid training news; a training
query excludes itself; and no cache contains retrieved veracity labels.

```powershell
python .\comment_graph\prepare_weibo_entity_evidence.py `
  --output-dir .\comment_graph\data\weibo_entity_evidence_valid_train_only

python .\comment_graph\smoke_entity_evidence.py `
  --output .\comment_graph\reports\smoke_entity_evidence_20261005.json
```

Five formal Weibo entries use a common protocol: each command selects the
checkpoint by validation Macro-F1 and then evaluates that frozen checkpoint on
test once. `direct` is the no-gate residual ablation; it is deliberately not
the main evidence model.

```powershell
# 1. MFCL control (only use a fresh held-out test; the existing seed 42/43/44
#    controls were already evaluated once on the current test split.)
python .\comment_graph\train.py --config .\comment_graph\config_weibo.yaml --mode baseline --graph-structure star --pooling-mode mean --decision-mode concat --mm-fusion mfcl --evidence-mode none --seed 1234 --allow-formal-training --run-test

# 2. MFCL + MIMoE (same fresh-test restriction as entry 1)
python .\comment_graph\train.py --config .\comment_graph\config_weibo.yaml --mode baseline --graph-structure star --pooling-mode mean --decision-mode concat --mm-fusion mimoe --evidence-mode none --seed 1234 --allow-formal-training --run-test

# 3. MFCL + label-free entity-evidence gate
python .\comment_graph\train.py --config .\comment_graph\config_weibo.yaml --mode baseline --graph-structure star --pooling-mode mean --decision-mode concat --mm-fusion mfcl --evidence-mode gated --seed 1234 --allow-formal-training --run-test

# 4. MFCL + MIMoE + label-free entity-evidence gate (main combined model)
python .\comment_graph\train.py --config .\comment_graph\config_weibo.yaml --mode baseline --graph-structure star --pooling-mode mean --decision-mode concat --mm-fusion mimoe --evidence-mode gated --seed 1234 --allow-formal-training --run-test

# 5. MFCL + MIMoE + direct evidence residual (remove only the reliability gate)
python .\comment_graph\train.py --config .\comment_graph\config_weibo.yaml --mode baseline --graph-structure star --pooling-mode mean --decision-mode concat --mm-fusion mimoe --evidence-mode direct --seed 1234 --allow-formal-training --run-test
```

## Fakeddit three-way comment-guided local correction

The GAHR-inspired local head now supports the configured number of classes.
It queries news tokens and image regions with comment embeddings, pools real
and generated evidence separately, and adds a gated class-logit residual to
the graph-and-image/text prediction. This is an adaptation of the global/local
correction idea, not a reproduction of GAHR's RoBERTa/DINOv2/CLIP architecture.
No news AIGC labels or AIGC auxiliary loss are introduced. Current Fakeddit
parent-first traversal represents reply structure, not chronological dynamics.

Run a CPU integration check while the VLM occupies the GPU. This uses nine
real training news (three per class) and previously generated comments, with
at most eight real comments per graph for the technical check:

```powershell
python .\comment_graph\smoke_local_correction.py `
  --config .\comment_graph\config_fakeddit_structure.yaml `
  --device cpu --graph-structure observed_tree --message-passing typed_rgcn `
  --samples-per-class 3 --max-real-comments 8 `
  --output-json .\comment_graph\reports\smoke_fakeddit_local_correction_3way.json
```

The check exercises two optimizer steps, nonzero GNN/local-head gradients,
label-input invariance, a root-only zero residual and three-class evaluation.
Its training loss is a plumbing diagnostic, not evidence of generalization.

After expanded VLM training exports the validation-selected adapter, generate
and validate all three graph files:

```powershell
.\comment_generator\partial_observation\scripts\generate_official_expanded_structured_graphs.ps1
```

The initial total-depth policy above is retained for reproducibility. A
training-output audit found that it rejects replies to observed real nodes
already deeper than six. The corrected experiment bounds consecutive generated
hops from the observed tree instead. It uses new output files and a separate
config, preserving the original partial diagnostic run:

```powershell
.\comment_generator\partial_observation\scripts\generate_official_expanded_structured_graphs.ps1 `
  -DepthReference observed_boundary -OutputTag expanded_generateddepth

.\comment_graph\run_fakeddit_expanded_gahr_validation.ps1 `
  -Config .\comment_graph\config_fakeddit_structure_expanded_generateddepth.yaml `
  -Seed 42 -RunRoot .\comment_graph\outputs\fakeddit_generateddepth_validation_seed42
```

Resume refuses to mix the two depth policies. Both require existing parents
and append-only edges; the corrected policy limits synthetic-chain growth
without excluding deep existing real replies. The old run's
`protocol_amendment.json` records why generation was restarted before any
classifier validation/test measurement.

The corrected config also sets `data.real_node_selection: observed_context`.
A train-only audit found that prefix-truncating real nodes to 64 could discard
real parents selected by the generator. All paired arms now keep the generator's
original observed real context plus its ancestors, and fill the remaining
64-node budget in original traversal order. Selection never uses generated
content, predicted parent choices or news labels. An oversized/missing context
raises an error rather than silently changing a generated parent. Legacy
configs retain their `prefix` behavior. The live run's
`context_selection_amendment.json` and original YAML snapshot record this change
before classifier training. Frozen evaluation restores this policy from the
saved checkpoint.

After generation completes, verify actual grafting on all three splits:

```powershell
python .\comment_graph\audit_generated_grafts.py `
  --config .\comment_graph\config_fakeddit_structure_expanded_generateddepth.yaml `
  --split train --split validate --split test `
  --output-json .\comment_graph\reports\expanded_generateddepth_graft_complete.json
```

This checks that the real-only arm has identical real nodes and edges, and
reports any predicted parents lost when building the classifier graph. While
generation is in progress, `--split train` checks only persisted training graphs;
it is not a complete-coverage certificate. `validate_generated_structured_graphs.py`
separately checks graph validity and expected split coverage.

Then compare real-only versus real-plus-generated, each with and without the
local residual, using identical news coverage and the same typed R-GCN:

```powershell
.\comment_graph\run_fakeddit_expanded_gahr_validation.ps1 -Seed 42
```

The script saves four validation summaries and separate checkpoints under
`outputs/fakeddit_expanded_gahr_validation_seed42`, and refuses to overwrite
existing arms. Use `-RunRoot` for a fresh run; additional seeds should be chosen
before inspecting their results. Formal settings come from
`config_fakeddit_structure_expanded.yaml` (including the full real-comment
limit, not the smoke-test override).

This runner does not evaluate test. Freeze the experimental settings on
validation, then use `evaluate_frozen_checkpoint.py` with each summary's exact
checkpoint path for paired test reporting. The existing small Fakeddit test
set has already been inspected in earlier experiments; further measurements
on it are exploratory, not a new untouched confirmation set. Report all
predeclared arms and seeds, including negative results.

For an isolated generated-parent placement comparison, use
`--graph-structure generated_root`: it preserves the observed real reply
edges and all comment texts, but attaches only generated nodes to the news
root. Compare it to `observed_tree` with the same generated JSONL, node limits,
seed, message passing, pooling and decision mode. The existing `star` mode
changes real edges too, so star-vs-tree alone does not isolate generated-parent
placement. The expanded four-arm local-correction runner remains unchanged;
this placement control is a separate validation experiment.

For the expanded `generateddepth` run, `finalize_fakeddit_structure.py` provides
the declared serial follow-up. It requires the upstream pipeline's
`validation_complete` status, checks the amended config hash, audits parent
grafting, and trains the additional `generated_root_typed_concat` control for
seeds 42, 13 and 24. It then freezes all 15 validation-selected checkpoint hashes
before any test pass. The script reports all five arms, not just a winning arm:

```powershell
python .\comment_graph\finalize_fakeddit_structure.py `
  --source-run .\comment_graph\outputs\fakeddit_expanded_pipeline_generateddepth_20261006 `
  --output-dir .\comment_graph\outputs\fakeddit_expanded_paired_final_20261006
```

Do not run this command concurrently with the queued follow-up already waiting
on the upstream pipeline. `queue_fakeddit_finalization.ps1` holds an exclusive
queue lock and verifies the upstream process start time to avoid PID reuse.
The finalizer refuses existing output folders. Outputs include `graft_audit.json`,
three placement validation checkpoints, `frozen_test_plan.json`, all per-arm
test JSON files with per-news predictions, `test_results.json`, and
`aggregate_results.json` (mean/sample standard deviation and paired differences).
These tests use the previously inspected small split and remain exploratory.

Windows checkpoint paths are kept below 240 characters: when the descriptive
filename is too long, training uses `ckpt_<stable-hash>.pt` in the same arm
directory. Always read the actual `checkpoint` path from the result JSON.
Best weights are written to a temporary file and atomically replaced only
after a successful save. This changes file handling, not training settings.
The 2026-10-06 run hit a reproducible 263-character path failure and was
recovered with `resume_fakeddit_expanded_validation.ps1`; `recovery_plan.json`
retains the failure details. Completed arms are verified against their saved
weights and skipped. Nonempty incomplete arm directories are not overwritten.
Original failure logs remain alongside separate recovery logs.

### Completed expanded Fakeddit experiment (2026-10-06)

The paired classifier cohort contains 154 official-train, 30 official-validation
and 49 official-test news. All 233 news have five generated comments (1,165 total).
The same observed-context-preserving real-node policy, at most 64 real comments,
typed R-GCN and seeds 42/13/24 are used in every arm. Generation receives no
veracity label. All 15 validation-selected checkpoints were frozen before their
test passes. This is an MFCL-based adaptation, not a full reproduction of either
the original MFCL or GAHR paper's benchmark protocol.

Actual measured percentages (test standard deviation is across three seeds):

| Arm | Validation macro-F1 mean | Test accuracy mean | Test macro-F1 mean ± sample std |
| --- | ---: | ---: | ---: |
| Real tree, concat | 87.84 | 74.15 | 72.08 ± 2.50 |
| Real tree, global/local correction | 88.85 | 82.31 | 81.54 ± 5.22 |
| Real + generated tree, concat | 84.29 | 80.27 | 78.82 ± 4.40 |
| Real + generated tree, global/local correction | 84.60 | 80.27 | 79.27 ± 9.80 |
| Real tree + generated root-only, concat | 90.05 | 79.59 | 78.70 ± 3.93 |

Within concat, generated tree comments improve test macro-F1 by 6.74 percentage
points over real-only; the paired seed differences are +13.03, 0.00 and +7.19.
However, structure-aware versus generated root-only is only +0.12 points on
average, with mixed signs across seeds. In the global/local setting, adding
generated comments reduces the test mean by 2.26 points. Thus the experiment
does **not** establish that predicted attachment is better than root attachment,
or that combining generation and the local module is the strongest method.
The test/validation rankings differ; do not choose new parameters from these
test results or report only a favorable seed.

Primary artifacts are in `outputs/fakeddit_expanded_paired_final_20261006/`:

- `frozen_test_plan.json`: all 15 checkpoint identities and hashes, recorded before test.
- `test_results.json` and `test_*_seed*.json`: per-seed metrics, per-news predictions and within-checkpoint local-correction diagnostics.
- `aggregate_results.json`: means, standard deviations and paired differences.
- `independent_result_audit.json`: independent metric/weight-hash verification and conditional paired-news bootstrap diagnostics.

The audit recomputes accuracy, macro-F1, weighted-F1 and confusion matrices from
all 735 stored news predictions. Its 5,000 label-stratified news bootstrap
replicates use the same sampled news for both compared methods and all three
fixed seeds. The conditional 95% interval for generated-tree minus real-only
concat is approximately [+0.17, +13.18] percentage points; for generated-tree
minus generated-root it is [-4.73, +4.93]. These intervals do not account for
training/model-selection uncertainty or multiple comparisons and are not a
claim of confirmatory significance. Only 49 news are tested, and this test pool
was inspected in previous exploratory work.

A shared generic image was flagged before classifier training across three
different news IDs/splits. A post-hoc sensitivity calculation excluding its
test news `7k89eq` leaves 48 news: real-only concat macro-F1 is 71.03%, generated
tree concat 78.12%, and generated-root concat 78.01%. Primary results remain the
full 49-news results; no data was silently removed or reassigned.

Ten deterministically selected validation examples and qualitative issue notes
are saved under `comment_generator/partial_observation/reports/` as
`expanded_generateddepth_validation_10_cases.json` and
`expanded_generateddepth_validation_review_notes.json`. Unsupported assertions,
near-duplicate replies and topic drift occur. These comments are synthetic
features, not verified factual evidence; causal image use has not been established
by an image-ablation experiment. Local-logit residuals directly correct only
0/1/1 test predictions across seeds in the real-only local models and 0/0/1 in
the generated local models; their whole-model gains cannot be described simply
as direct inference-time correction of every additional correct prediction.

The Windows path failure and its single recovery are documented in
`outputs/fakeddit_expanded_pipeline_generateddepth_20261006/recovery_plan.json`.
The first three completed runs were retained unchanged. No training hyperparameter
was changed during recovery. CUDA graph operations are not configured for strict
bitwise determinism, so setting the same seed alone is not a bitwise replay guarantee.

```powershell
python .\comment_graph\train.py `
  --config .\comment_graph\config_fakeddit_structure_expanded.yaml `
  --mode real_generated --graph-structure generated_root `
  --message-passing typed_rgcn --pooling-mode mean --decision-mode concat `
  --seed 42 --allow-formal-training `
  --checkpoint-dir .\comment_graph\outputs\fakeddit_expanded_placement_seed42 `
  --summary-json .\comment_graph\outputs\fakeddit_expanded_placement_seed42\validation.json
```
