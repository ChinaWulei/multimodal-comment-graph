"""Frozen, label-blind Qwen direct classification; validation only, no training."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
ARMS = {"base_mm": (False, False), "base_real": (False, True),
        "adapter_mm": (True, False), "adapter_real": (True, True)}


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            result.update(chunk)
    return result.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def save(path, value, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value) if lines
                else json.dumps(value, ensure_ascii=False, indent=2))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


@contextmanager
def lock(out):
    import msvcrt
    with (out / "runner.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def parse_label(text):
    text = text.strip()
    return int(text) if text in {"0", "1", "2"} else -1


def messages(item, comments):
    # This allowlist keeps IDs, labels, subreddit metadata and source paths out of the prompt.
    data = {"news_text": item["news_text"]}
    if comments:
        data["observed_real_comments"] = item["comments"]
    return [
        {"role": "system", "content": [{"type": "text", "text": (HERE / "prompt.txt").read_text(encoding="utf-8")}]},
        {"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": json.dumps(data, ensure_ascii=False)}]},
    ]


def clipped(text, tokenizer, maximum):
    """Keep the same source-text span as the MFCL BERT budget, without decoding WordPieces."""
    encoded = tokenizer(text, truncation=True, max_length=maximum, return_offsets_mapping=True)
    offsets = [pair for pair in encoded["offset_mapping"] if pair[1] > pair[0]]
    if not offsets:
        raise ValueError("Empty tokenized source text")
    stop = max(pair[1] for pair in offsets)
    source = text[:stop]
    # The character boundary must preserve exactly the detector's unpadded token IDs.
    assert tokenizer(source, truncation=True, max_length=maximum)["input_ids"] == encoded["input_ids"]
    return source, bool(text[stop:].strip())


def prepare(config, out):
    from PIL import Image
    from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform, resolve_path
    graph, graph_path = load_config(ROOT / config["graph_config"])
    compare = ROOT / config["comparison_dir"]
    original = json.loads((compare / "protocol.json").read_text(encoding="utf-8"))
    graph.data.max_real_comments = config["max_real_comments"]
    graph.data.max_generated_comments = 5
    graph.graph.structure_mode = "observed_tree"
    tokenizer, transform = make_tokenizer_and_transform(graph)
    primary = set(original["primary_validation_news"])
    val_ids = {r["metadata"]["news_id"] for r in read(compare / "contexts_validate.jsonl")}
    smoke_ids = set(original["smoke_news"])
    sft_train = read(ROOT / "comment_generator/partial_observation/data/official_subset/structure_train_expanded.jsonl")
    sft_val = read(ROOT / "comment_generator/partial_observation/data/official_subset/structure_validate_expanded.jsonl")
    fit_ids = {r["metadata"]["news_id"] for r in sft_train}
    selection_ids = {r["metadata"]["news_id"] for r in sft_val}
    assert len(primary) == 44 and len(val_ids) == 77 and len(smoke_ids) == 8
    assert primary == val_ids - selection_ids and not val_ids & fit_ids
    assert not smoke_ids & val_ids
    payload, targets, audit = {}, {}, {}
    for split, ids, name in [("train", smoke_ids, "smoke"), ("validate", val_ids, "validation")]:
        graph.data.generated_comments[split] = str(compare / f"iterative_{split}.jsonl")
        dataset = make_dataset(graph, graph_path, split, "real_only", tokenizer, transform, allowed_news_ids=ids)
        assert {r["news_id"] for r in dataset.records} == ids
        payload[name], targets[name], audit[name] = [], [], []
        for row in sorted(dataset.records, key=lambda r: r["news_id"]):
            nid = row["news_id"]
            real, edges = dataset._real_nodes_and_edges(nid)
            expected_texts, _, _ = dataset._nodes_and_edges(nid)
            assert expected_texts == [str(n["body"]).strip() for n in real]
            parents = {target: source for source, target, _ in edges}
            headline, truncated_head = clipped(str(row["news_text"]).strip(), tokenizer, graph.data.max_news_length)
            comments, truncated = [], []
            for index, node in enumerate(real, 1):
                text, was_truncated = clipped(str(node["body"]).strip(), tokenizer, graph.data.max_comment_length)
                comments.append({"node_index": index, "parent_node_index": parents[index], "content": text})
                truncated.append(was_truncated)
            with Image.open(row["image_path"]) as image:
                image.load()
            item = {"news_id": nid, "image_path": row["image_path"], "news_text": headline, "comments": comments}
            assert set(item) == {'news_id', 'image_path', 'news_text', 'comments'}
            assert all(set(n) == {'node_index', 'parent_node_index', 'content'} for n in comments)
            payload[name].append(item)
            targets[name].append({"news_id": nid, "label": int(row["3_way_label"]), "primary": nid in primary})
            audit[name].append({"news_id": nid, "comment_ids": [n["comment_id"] for n in real],
                                "headline_truncated": truncated_head, "comments_truncated": truncated,
                                "image_sha256": digest(row["image_path"]), "same_selection_as_MFCL": True})
    # Validate numeric labels against the previously audited 2/3/6-way local mapping.
    manifest = resolve_path(graph_path, graph.data.manifest_csv)
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    cross = Counter((r["2_way_label"], r["3_way_label"], r["6_way_label"]) for r in manifest_rows
                    if r['3_way_label'] in {'0', '1', '2'})
    assert all((a == "1") == (b == "0") == (c == "0") for a, b, c in cross)
    # Three-way labels cannot be derived from six-way labels: 6-way=5 spans both fake types.
    import pandas as pd
    training_source = ROOT / 'multimodal_only_samples/multimodal_train.tsv'
    source_labels = pd.read_csv(training_source, sep='\t', dtype=str,
                               usecols=['subreddit', '2_way_label', '3_way_label', '6_way_label'])
    source_cross = source_labels.groupby(['3_way_label', '6_way_label', 'subreddit']).size()
    assert set(source_labels.loc[source_labels['3_way_label'] == '1', 'subreddit']) == {'propagandaposters'}
    sources = [ROOT / config["graph_config"], ROOT / config["generator_config"], manifest,
               resolve_path(graph_path, graph.data.real_trees_jsonl),
               compare / "protocol.json", compare / "comparison.json", training_source,
               compare / "iterative_train.jsonl", compare / "iterative_validate.jsonl",
               ROOT / config["label_definition_reference"], Path(__file__), HERE / "prompt.txt", HERE / "config.json",
               ROOT / "comment_graph/fakeddit_structure_dataset.py", ROOT / "comment_graph/common.py"]
    for folder in [ROOT / config["model_dir"], ROOT / config["adapter_dir"]]:
        sources.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".jinja", ".txt"})
    sources.extend(sorted((compare / 'classification').glob('*.json')))
    protocol = {"config": config, "arms": ARMS, "sources_sha256": {str(p): digest(p) for p in sources},
                "inputs_sha256": {name: object_digest(rows) for name, rows in payload.items()},
                "targets_sha256": {name: object_digest(rows) for name, rows in targets.items()},
                "input_audit_sha256": object_digest(audit),
                "primary_ids": sorted(primary), "validation_ids": sorted(val_ids), "smoke_ids": sorted(smoke_ids),
                "validation_labels": dict(Counter(str(r["label"]) for r in targets["validation"])),
                "label_crosscheck": {"/".join(k): v for k, v in cross.items()},
                "training_subreddit_crosscheck": {'/'.join(k): int(v) for k, v in source_cross.items()},
                "label_definition_primary_source": "https://arxiv.org/html/1911.03854v2#S3.SS3",
                "label_mapping_evidence": "Prior sample_check/report.md audit plus local 2/3/6 crosscheck; no inferred relabeling.",
                "decoding": {"do_sample": False, "primary": "strict digit output; invalid counted as incorrect",
                             "secondary": "highest first-token logit among single-token 0/1/2 (restricted choice diagnostic, not calibrated confidence)"},
                "warnings": ["Previously explored validation; not blind. No test data is evaluated.",
                             "Comment Adapter was trained for generation, not classification.",
                             "Matched selected real comments and BERT-visible source spans; tokenizers and image preprocessing differ.",
                             "Qwen uses full image at 200704-401408 pixels; MFCL uses 224 center crop.",
                             "Original post/real comments can contain source cues; no metadata labels or subreddit field supplied.",
                             "Unknown pretraining overlap with public Fakeddit cannot be ruled out."]}
    frozen = out / "protocol.json"
    # JSON serializes tuples as lists; compare canonical payloads, not Python container types.
    if frozen.exists() and object_digest(json.loads(frozen.read_text(encoding="utf-8"))) != object_digest(protocol):
        raise ValueError("Protocol/source changed. Use a new output directory; do not overwrite previous results.")
    save(frozen, protocol)
    for name in payload:
        save(out / f"{name}_inputs.jsonl", payload[name], True)
        save(out / f"{name}_targets.jsonl", targets[name], True)
    save(out / "input_audit.json", audit)
    print(json.dumps({"prepared": True, "validation": len(val_ids), "primary": len(primary), "smoke": len(smoke_ids)}), flush=True)


def verify_sources(out):
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    for path, expected in protocol["sources_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for group in ['inputs', 'targets']:
        for name, expected in protocol[f'{group}_sha256'].items():
            if object_digest(read(out / f'{name}_{group}.jsonl')) != expected:
                raise ValueError(f'Prepared {group} changed: {name}')
    audit = json.loads((out / 'input_audit.json').read_text(encoding='utf-8'))
    assert object_digest(audit) == protocol['input_audit_sha256']
    for name in ['smoke', 'validation']:
        expected_images = {r['news_id']: r['image_sha256'] for r in audit[name]}
        for row in read(out / f'{name}_inputs.jsonl'):
            assert digest(row['image_path']) == expected_images[row['news_id']]
    return protocol


def evaluate(config, out, arm, cohort):
    import torch
    from PIL import Image, ImageOps
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no long CPU inference")
    use_adapter, with_comments = ARMS[arm]
    inputs_path = out / f"{cohort}_inputs.jsonl"
    records = read(inputs_path)
    protocol_hash = digest(out / "protocol.json")
    path = out / cohort / f"{arm}.jsonl"
    contract = {"protocol_sha256": protocol_hash, "input_sha256": digest(inputs_path), "arm": arm, "cohort": cohort}
    meta_path = path.with_suffix(".run.json")
    done = read(path) if path.exists() else []
    if meta_path.exists():
        assert json.loads(meta_path.read_text(encoding="utf-8"))["contract"] == contract
    elif done:
        raise ValueError("Output exists without resume contract")
    ids = [r["news_id"] for r in done]
    assert len(set(ids)) == len(ids) and set(ids) <= {r["news_id"] for r in records}
    assert all(r["arm"] == arm and r["protocol_sha256"] == protocol_hash for r in done)
    if len(done) == len(records):
        print(f"VERIFIED completed {arm}/{cohort}", flush=True); return
    torch.manual_seed(config["seed"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    processor = AutoProcessor.from_pretrained(str(ROOT / config["model_dir"]), local_files_only=True,
                                             min_pixels=config["min_pixels"], max_pixels=config["max_pixels"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(ROOT / config["model_dir"]), local_files_only=True,
             dtype=dtype, quantization_config=quant, device_map={"": 0}, attn_implementation="sdpa")
    if use_adapter:
        model = PeftModel.from_pretrained(model, str(ROOT / config["adapter_dir"]), is_trainable=False)
    model.eval()
    assert not model.training
    # Single digit tokens allow an unambiguous secondary closed-set score from the same first forward pass.
    token_ids = [processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(3)]
    assert all(len(t) == 1 for t in token_ids)
    token_ids = [t[0] for t in token_ids]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    save(meta_path, {"contract": contract, "state": "running", "completed": len(done)})
    for item in records:
        if item["news_id"] in ids:
            continue
        tick = time.perf_counter()
        with Image.open(item["image_path"]) as picture:
            picture = ImageOps.exif_transpose(picture).convert("RGB").copy()
        chat = messages(item, with_comments)
        prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[prompt], images=[picture], return_tensors="pt", truncation=False).to(model.device)
        assert "pixel_values" in batch and "image_grid_thw" in batch
        length = batch["input_ids"].shape[1]
        result = {"news_id": item["news_id"], "arm": arm, "protocol_sha256": protocol_hash,
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "input_tokens": length,
                  "comment_count": len(item["comments"]) if with_comments else 0,
                  "image_grid_thw": batch["image_grid_thw"].cpu().tolist(), "label_used_as_input": False}
        if length > config["max_input_tokens"]:
            result.update({"raw_output": "", "prediction": -1, "restricted_prediction": -1, "error": "context_overflow_not_dropped"})
        else:
            with torch.inference_mode():
                generated = model.generate(**batch, do_sample=False, max_new_tokens=config["max_new_tokens"],
                    repetition_penalty=1.0, return_dict_in_generate=True, output_scores=True,
                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
            raw = processor.batch_decode(generated.sequences[:, length:], skip_special_tokens=True)[0].strip()
            scores = generated.scores[0][0, token_ids].float()
            result.update({"raw_output": raw, "prediction": parse_label(raw),
                           "restricted_prediction": int(scores.argmax()), "restricted_label_logits": scores.cpu().tolist(),
                           "error": None if parse_label(raw) >= 0 else "invalid_digit_output"})
            del generated
        result["seconds"] = time.perf_counter() - tick
        done.append(result)
        save(path, done, True)
        save(meta_path, {"contract": contract, "state": "running", "completed": len(done),
                        "session_seconds": time.perf_counter() - started,
                        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
        print(f"{arm}/{cohort} {len(done)}/{len(records)} id={item['news_id']} output={result['raw_output']!r}", flush=True)
        del batch
    meta = json.loads(meta_path.read_text(encoding="utf-8")); meta["state"] = "completed"; save(meta_path, meta)


def metrics(rows, truth, field):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    expected = {r["news_id"] for r in rows}
    assert expected == set(truth) and len(rows) == len(truth)
    y = [truth[r["news_id"]] for r in rows]
    p = [r[field] for r in rows]
    return {"n": len(y), "accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0),
            "weighted_f1": f1_score(y, p, labels=[0, 1, 2], average="weighted", zero_division=0),
            "invalid_count": sum(v == -1 for v in p),
            "per_class": classification_report(y, p, labels=[0, 1, 2], output_dict=True, zero_division=0),
            "confusion_matrix_true_012_pred_012invalid": confusion_matrix(y, p, labels=[0, 1, 2, -1])[:3].tolist()}


def summarize(out):
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    truth = read(out / "validation_targets.jsonl")
    results = {}
    for arm in ARMS:
        rows = read(out / "validation" / f"{arm}.jsonl")
        assert {r["news_id"] for r in rows} == set(protocol["validation_ids"])
        assert all(r["protocol_sha256"] == digest(out / "protocol.json") for r in rows)
        results[arm] = {}
        for name, selected in [("primary44", [r for r in truth if r["primary"]]), ("auxiliary77", truth)]:
            targets = {r["news_id"]: r["label"] for r in selected}
            subset = [r for r in rows if r["news_id"] in targets]
            results[arm][name] = {key: metrics(subset, targets, field) for key, field in
                                  [("strict_output", "prediction"), ("restricted_choice_diagnostic", "restricted_prediction")]}
        save(out / "validation" / f"{arm}_errors.jsonl", [dict(r, true_label={t['news_id']: t['label'] for t in truth}[r['news_id']])
             for r in rows if r['prediction'] != {t['news_id']: t['label'] for t in truth}[r['news_id']]], True)
    compare = ROOT / protocol["config"]["comparison_dir"]
    references = json.loads((compare / "comparison.json").read_text(encoding="utf-8"))
    # Check actual per-news reference predictions, not merely matching row counts.
    for arm in ["A_real", "E_observed_parent", "D_observed_root"]:
        for seed in [42, 13, 24]:
            saved = json.loads((compare / "classification" / f"{arm}_seed{seed}.json").read_text(encoding="utf-8"))
            assert {r['news_id'] for r in saved['validation']['prediction_records']} == set(protocol['primary_ids'])
    report = {"qwen": results, "detector_primary44": references['aggregate'],
              "detector_auxiliary77": references['auxiliary_validation77'], "test_evaluated": False,
              "warnings": protocol['warnings'], "reference_gate_passed": references['test_gate_passed']}
    save(out / "comparison.json", report)
    lines = ["# Qwen 直接分类验证对照", "", "只评估已有验证集，不是测试集结果。主比较 44 条，辅助比较 77 条。", "",
             "| 组别 | 主验证 Accuracy | 主验证 Macro-F1 | 非法输出 | 限定类别选择 Macro-F1（辅助） |",
             "|---|---:|---:|---:|---:|"]
    for arm, row in results.items():
        main = row['primary44']['strict_output']; aux = row['primary44']['restricted_choice_diagnostic']
        lines.append(f"| {arm} | {main['accuracy']:.2%} | {main['macro_f1']:.2%} | {main['invalid_count']}/44 | {aux['macro_f1']:.2%} |")
    for arm in ['A_real', 'D_observed_root', 'E_observed_parent']:
        row = references['aggregate'][arm]
        lines.append(f"| {arm}（3 seeds 均值±标准差） | {row['accuracy']['mean']:.2%} ± {row['accuracy']['std']:.2%} | {row['macro_f1']['mean']:.2%} ± {row['macro_f1']['std']:.2%} | — | — |")
    lines.extend(['', '## 解释限制', '', '- Qwen 四组为固定贪心解码单次运行；不能把重复同一确定性输出称为三个独立种子。',
                  '- 主指标严格接受单个类别数字，非法输出计错，不删样本；辅助指标比较首 token 的三个类别得分，不是校准概率。',
                  '- 若评论 Adapter 不遵循分类指令，不能据此断言它没有类别信息；需同时查看辅助指标。',
                  '- 没有训练分类专用 Qwen LoRA，也不能据此说明经过分类训练的 Qwen 上限。',
                  '- 同一新闻/评论与文字范围，图像分辨率和模型预训练资源不同，属于实用系统对照，不是完全等计算量架构消融。',
                  '- 小规模、已探索验证集，不保证泛化；公开数据的预训练污染未知。',
                  '- 原 A–E 结构增强实验未通过结构优势门槛，本次不绕过规则评估测试集。'])
    (out / "report.md").write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({a: r['primary44']['strict_output'] for a, r in results.items()}, ensure_ascii=False), flush=True)


def child(config_path, out, arm, cohort):
    command = [sys.executable, '-u', str(Path(__file__)), '--config', str(config_path), '--output-dir', str(out),
               '--stage', 'evaluate', '--arm', arm, '--cohort', cohort]
    save(out / 'status.json', {'state': 'running', 'arm': arm, 'cohort': cohort})
    (out / 'logs').mkdir(exist_ok=True)
    with (out / 'logs' / f'{arm}_{cohort}.log').open('a', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{arm}/{cohort} failed, see logs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'evaluate', 'full', 'summarize'], default='prepare')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--cohort', choices=['smoke', 'validation'], default='validation')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    out = args.output_dir.resolve() if args.output_dir else ROOT / config['output_dir']
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'evaluate':
        if not args.arm:
            raise ValueError('--arm required')
        verify_sources(out)
        evaluate(config, out, args.arm, args.cohort)
        return
    with lock(out):
        try:
            prepare(config, out)
            if args.stage in {'smoke', 'full'}:
                for arm in ARMS:
                    child(args.config, out, arm, 'smoke')
                smoke = {a: read(out / 'smoke' / f'{a}.jsonl') for a in ARMS}
                assert all(len(rows) == 8 and all(r['image_grid_thw'] and r['input_tokens'] > 0 for r in rows) for rows in smoke.values())
                assert all(r['restricted_prediction'] in {0, 1, 2} for rows in smoke.values() for r in rows)
                # Malformed free outputs are measured, not used to tune or reject an arm.
                same_prompts = all([r['prompt_sha256'] for r in smoke[b]] == [r['prompt_sha256'] for r in smoke[t]]
                                   for b, t in [('base_mm', 'adapter_mm'), ('base_real', 'adapter_real')])
                assert same_prompts
                save(out / 'smoke_report.json', {'passed': True, 'actual_images': True,
                     'invalid_by_arm': {a: sum(r['prediction'] == -1 for r in rows) for a, rows in smoke.items()},
                     'prompt_hashes_match_between_models': same_prompts})
            if args.stage == 'full':
                for arm in ARMS:
                    child(args.config, out, arm, 'validation')
            if args.stage in {'full', 'summarize'}:
                summarize(out)
            save(out / 'status.json', {'state': 'completed', 'stage': args.stage})
        except BaseException as exc:
            save(out / 'status.json', {'state': 'failed', 'stage': args.stage, 'error': str(exc)})
            raise


if __name__ == '__main__':
    main()
