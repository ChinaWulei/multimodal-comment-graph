# CHECKED Qwen-VL comment generator experiment

This experiment is isolated from MFCL and from the Fakeddit adapters. It uses
CHECKED Chinese microblog text, the downloaded main-post image, and a seeded
subset of observed real comments to predict another real comment. The truth
label is retained only in metadata and is never passed to the generator prompt.

CHECKED comments do not expose `parent_id`; they are treated as a flat direct
comment list. Splits are stratified by the binary `real`/`fake` label at the
news-ID level. No comment or news ID is shared across splits.

## Prepare and validate

```powershell
python .\comment_generator\checked\scripts\prepare_dataset.py `
  --config .\comment_generator\checked\config\config.yaml

python .\comment_generator\checked\scripts\validate_dataset.py `
  --config .\comment_generator\checked\config\config.yaml

python .\comment_generator\checked\scripts\check_longest_samples.py `
  --config .\comment_generator\checked\config\config.yaml `
  --limit 100
```

## Processor-only smoke test

```powershell
python .\comment_generator\partial_observation\scripts\train_generator.py `
  --config .\comment_generator\checked\config\config.yaml `
  --check-data-only `
  --processor-smoke-test `
  --limit 20 `
  --local-files-only
```

## Ten-step QLoRA smoke training

```powershell
python .\comment_generator\partial_observation\scripts\train_generator.py `
  --config .\comment_generator\checked\config\config.yaml `
  --smoke-train `
  --local-files-only
```

Do not start the full two-epoch run until the smoke adapter has produced
reasonable Chinese comments on the validation split.

## Full two-epoch QLoRA training

The current dataset contains 4,665 training samples. On the tested RTX A5000,
the 10-step smoke run used 6.56 GiB allocated GPU memory and took 84 seconds.
The full run is expected to take roughly 2.5--3 hours including validation.

```powershell
python .\comment_generator\partial_observation\scripts\train_generator.py `
  --config .\comment_generator\checked\config\config.yaml `
  --local-files-only
```

Generate five comments for ten unseen validation news items:

```powershell
python .\comment_generator\partial_observation\scripts\generate_comments.py `
  --config .\comment_generator\checked\config\config.yaml `
  --adapter-dir .\comment_generator\checked\outputs\qwen2_5_vl_3b_checked_comment_lora `
  --output-jsonl .\comment_generator\checked\outputs\generated_validate.jsonl `
  --split validate `
  --limit 10 `
  --comments-per-news 5 `
  --local-files-only
```

The binary truth label remains metadata for analysis; generation never reads it.
Only the first official image is currently passed to Qwen when a post contains
multiple images. All image paths remain in each record's metadata for a future
multi-image ablation.
