"""Paired re-evaluation of existing quality adapters, no retraining or test data.

Reproduce original BF16-head predictions before trusting FP32-head differences.
Both heads consume the SAME final hidden state; FP32 computes only No/Yes rows.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.train_quality_lora_cv import SYSTEM, content_target, request, sha
from comment_graph.train_generated_quality_gate import precision_threshold


def metrics(y, p):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    result = {"n": len(y), "unique_probabilities": len(np.unique(p)),
              "oof_roc_auc": float(roc_auc_score(y, p)),
              "oof_average_precision": float(average_precision_score(y, p)), "passed": False}
    try:
        if result["oof_roc_auc"] <= .6:
            raise ValueError("AUC must exceed .60")
        result.update(precision_threshold(p, y), passed=True)
    except ValueError as error:
        result["failure_reason"] = str(error)
    return result


def main():
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, default=ROOT / "comment_graph/outputs/ai_quality_lora_cv_train140")
    parser.add_argument("--fit-dir", type=Path, default=ROOT / "comment_graph/outputs/ai_quality_gate_train140")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier precision audits")
    prior = json.loads((args.cv_dir / "report.json").read_text(encoding="utf-8"))
    protocol = prior["protocol"]
    if protocol["mode"] != "cv" or protocol["system_prompt"] != SYSTEM:
        raise ValueError("Expected the same completed cross-validation protocol")
    if (sha(args.fit_dir / "annotations_ai.jsonl") != protocol["annotation_sha256"] or
            sha(args.fit_dir / "reference_fit.jsonl") != protocol["reference_sha256"]):
        raise ValueError("Reviewed input or annotation data changed")
    rows = {r["annotation_id"]: r for r in map(json.loads,
            (args.fit_dir / "annotations_ai.jsonl").read_text(encoding="utf-8").splitlines())}
    model_path = Path(protocol["model_path"])
    if sha(model_path / "config.json") != protocol["model_config_sha256"]:
        raise ValueError("Model config changed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires the original BF16 CUDA environment")
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
                                             min_pixels=3136, max_pixels=protocol["max_pixels"])
    encoded_ids = [processor.tokenizer.encode(word, add_special_tokens=False) for word in ("No", "Yes")]
    if any(len(ids) != 1 for ids in encoded_ids):
        raise ValueError("No/Yes tokenization changed")
    tokens = [ids[0] for ids in encoded_ids]
    predictions = []
    for fold in prior["fold_reports"]:
        if set(fold["train_news"]) & set(fold["heldout_news"]):
            raise ValueError("Fold news overlap")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, str(args.cv_dir / f"fold_{fold['fold']}"),
                                          is_trainable=False).eval()
        model.requires_grad_(False)
        inner = model.get_base_model()
        for old in fold["predictions"]:
            row = rows[old["annotation_id"]]
            if row["news_id"] not in fold["heldout_news"] or content_target(row) != old["target"]:
                raise ValueError("Prediction or target identity changed")
            with Image.open(row["image_path"]) as raw:
                picture = ImageOps.exif_transpose(raw).convert("RGB").copy()
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": request(row)}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[picture], return_tensors="pt").to("cuda")
            if inputs["input_ids"].shape[1] > protocol["max_input_tokens"] or not inputs["pixel_values"].numel():
                raise ValueError("Input too long or missing image")
            with torch.inference_mode():
                hidden = inner.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -1:, :]
                low = inner.lm_head(hidden)[:, -1, tokens].float()
                # Conversion BEFORE projection avoids rounding large, nearly equal logits.
                high = torch.nn.functional.linear(hidden[:, -1].float(), inner.lm_head.weight[tokens].float())
                p_low = float(torch.softmax(low, -1)[0, 1])
                p_high = float(torch.softmax(high, -1)[0, 1])
            predictions.append({"annotation_id": row["annotation_id"], "news_id": row["news_id"],
                                "fold": fold["fold"], "target": old["target"],
                                "original_probability": old["probability"], "bf16_probability": p_low,
                                "fp32_probability": p_high, "bf16_logits": low[0].tolist(),
                                "fp32_logits": high[0].tolist()})
        print(f"audited_fold={fold['fold']} rows={len(predictions)}", flush=True)
        del inner, model, base, hidden, low, high, inputs
        gc.collect()
        torch.cuda.empty_cache()
    if len(predictions) != len(prior["oof_predictions"]) or len({r["annotation_id"] for r in predictions}) != len(predictions):
        raise ValueError("Incomplete or repeated OOF predictions")
    errors = [abs(r["original_probability"]-r["bf16_probability"]) for r in predictions]
    if max(errors) > 1e-5:
        raise ValueError(f"Original predictions not reproduced: maximum absolute difference {max(errors)}")
    y = [r["target"] for r in predictions]
    report = {"original_predictions_reproduced": True, "max_reproduction_absolute_error": max(errors),
              "same_hidden_state_for_both_heads": True, "weights_retrained": False,
              "cv_report_sha256": sha(args.cv_dir / "report.json"),
              "bf16": metrics(y, [r["bf16_probability"] for r in predictions]),
              "fp32": metrics(y, [r["fp32_probability"] for r in predictions]),
              "predictions": predictions, "reserved_audit_used": False,
              "warning": "Development-set AI-label analysis, not independent quality or classifier improvement."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
