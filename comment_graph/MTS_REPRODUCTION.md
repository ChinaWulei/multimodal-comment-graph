# MFCL + MTS integration (2026-10-04)

## Method choice and scope

Selected: *Multimodal Taylor Series Network for Misinformation Detection*, WWW 2025,
https://doi.org/10.1145/3696410.3714719 .
Author paper: https://yikewu.tech/files/www_taylor.pdf .
Author code: https://github.com/OneForAllSama/MTS , inspected commit
`77f6634b992b81efabbb6125e8797b74b154857e`, notebook `MTS/MTS_ALL_IN_ONE_CODE.ipynb`.

The full fusion core is implemented: two directional recurrences, linear terms,
cross-modal products, self products, eight layers, batch infinity normalization,
identity-plus-Xavier initialization, and final directional sum. No component of
this fusion core is replaced by a generic attention approximation.

This is **MFCL + the reproduced MTS core**, not a reproduction of the paper's
reported end-to-end score. We retain MFCL's existing frozen BERT/ResNet50 and
trainable contextual module, classifier, and available comment GNN. The paper
uses ResNet34, a different feature interface/training setup and dataset split.
The paired control uses the same local data, preprocessing, seed, batch size,
optimizer, and epoch/selection budget. Only the fusion/head input is changed.
Both are the current online image/text training entry, not a re-run of the
original MFCL paper's contrastive pretraining protocol. The old `Weibo/test.py`
expects 1,174/293 precomputed NPZ IDs and uses test during selection; it is not
used here. The newly found CSV data are a different Weibo preparation. We cannot
compare our score directly with either published paper's score.

Source review found the released `MDR.forward` returns tensors collected through
`.data`, disconnecting its trainable recurrence from the classifier loss. Our
implementation follows the differentiable paper equations instead. We do not
execute the notebook. Epsilon protection at all-zero input is another explicit
numerical safety fix. Layer outputs and all coefficient gradients are tested.

Batch infinity normalization is deliberately retained, not silently replaced
with sample normalization. Consequently predictions depend on batch companions.
Keep validation/test order and batch size fixed when comparing checkpoints.
This is a deployment limitation and should be reported, not a reason to tune
test batches. No claim of causal inference is made.

## Actual local data

CHECKED is already prepared in `comment_generator/checked`: 939 train, 118
validation, 118 test; real/generated comments exist. Three prior source-isolated
controls achieved validation F1=1.0. That validation set is saturated.

Weibo source located at `E:/科研/数据集/Weibo`, not the repository's `Weibo/`
code directory or the missing `Chinese/*.npz` format of the original entry point.
It has text, images, original binary labels, `event_label` and `category`, but
**no real comments or reply tree**. These latter metadata are not model inputs.
CSV labels consistently match image folders: 0=nonrumor, 1=rumor.

| Split | Raw rows | Rows with valid image/text/label |
| --- | ---: | ---: |
| train | 5415 | 4766 |
| validate | 843 | 737 |
| test | 1465 | 1331 |

Audit examined 10,754 referenced images with Pillow and SHA256. No identical
normalized full text or selected primary image bytes occur across these splits.
This does not rule out near-duplicates or overlapping events. Original post IDs
have lost precision (only 50 distinct IDs in 5,415 training rows); retain them
for provenance but use full text/image hashes as internal keys. This is not a
reconstruction of the real post ID. Exact same-text/same-primary-image duplicates
are counted and removed within each split; training retains 4,765 rows.
The first valid source-listed image is selected without reading the label.
Missing images are not downloaded or replaced. Source files are never modified.

## Predeclared first comparison

Before training: Weibo, seed 42, MFCL control versus MFCL+MTS, full available
training rows; at most 10 epochs, patience 3. AdamW lr=1e-4, batch 16, frozen
backbones, common train-derived class weights. Both retain the same MFCL
contextual feature module. Primary metric is validation macro-F1; strict best
checkpoint, earliest tie. No test inference in this runner.

This first seed is exploratory. A positive difference must be rechecked using
seeds 43 and 44, including all runs, before calling it a stable improvement.
Do not repeatedly adjust configurations against test. The CHECKED comparison
is separate and must not be represented as Weibo's comment-graph evaluation.

## Commands (PowerShell, multimodal-fakenews)

```powershell
# These report paths must be new; omit if already completed.
python .\comment_graph\audit_weibo.py --data-root 'E:\科研\数据集\Weibo' --output .\comment_graph\reports\weibo_source_audit_20261004.json
python .\comment_graph\smoke_mts.py --output .\comment_graph\reports\smoke_mts_20261004.json

# Full training and validation, both arms, separate checkpoints; no test tuning.
python .\comment_graph\run_mts_comparison.py --dataset weibo --seeds 42 --epochs 10 --patience 3

# Replication seeds after freezing the method, not a parameter sweep.
python .\comment_graph\run_mts_comparison.py --dataset weibo --seeds 43 44 --epochs 10 --patience 3

# Separate CHECKED integration comparison; uses existing generated comments.
python .\comment_graph\run_mts_comparison.py --dataset checked --seeds 42 --epochs 10 --patience 3
```

All comparison outputs have new timestamped directories, complete logs, plans,
source hashes, checkpoints and JSON summaries. No old results are overwritten.
The previously added null-reference experiment stays OFF and is not combined
with MTS. The original MFCL files are not modified by this integration.
