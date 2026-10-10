"""Read-only provenance/metric checks, followed by a separately saved audit."""
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from run import HERE, ROOT, ARMS, read, digest, save, verify_sources


def main():
    config = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    out = ROOT / config['output_dir']
    protocol = verify_sources(out)
    source = ROOT / 'multimodal_only_samples/multimodal_validate.tsv'
    data = pd.read_csv(source, sep='\t', dtype=str, usecols=['id', 'clean_title', '3_way_label'])
    assert not data['id'].duplicated().any()
    originals = data.set_index('id').to_dict('index')
    inputs = {r['news_id']: r for r in read(out / 'validation_inputs.jsonl')}
    truth = {r['news_id']: r['label'] for r in read(out / 'validation_targets.jsonl')}
    assert set(inputs) == set(truth) == set(protocol['validation_ids'])
    for nid, row in inputs.items():
        assert int(originals[nid]['3_way_label']) == truth[nid]
        assert originals[nid]['clean_title'].strip().startswith(row['news_text'])
        assert Path(row['image_path']).stem == nid
    predictions = {arm: read(out / 'validation' / f'{arm}.jsonl') for arm in ARMS}
    for rows in predictions.values():
        assert len(rows) == len(inputs) and {r['news_id'] for r in rows} == set(inputs)
        assert all(r['label_used_as_input'] is False and r['image_grid_thw'] for r in rows)
    for base, adapted in [('base_mm', 'adapter_mm'), ('base_real', 'adapter_real')]:
        assert {r['news_id']: r['prompt_sha256'] for r in predictions[base]} == {
            r['news_id']: r['prompt_sha256'] for r in predictions[adapted]}
    summary = json.loads((out / 'comparison.json').read_text(encoding='utf-8'))
    primary = set(protocol['primary_ids'])
    for arm, rows in predictions.items():
        subset = [r for r in rows if r['news_id'] in primary]
        exact = sum(r['prediction'] == truth[r['news_id']] for r in subset)
        assert abs(exact / 44 - summary['qwen'][arm]['primary44']['strict_output']['accuracy']) < 1e-10
        assert sum(sum(row) for row in summary['qwen'][arm]['primary44']['strict_output']['confusion_matrix_true_012_pred_012invalid']) == 44
    audit = {'passed': True, 'original_validation_tsv_sha256': digest(source),
             'protocol_sha256': digest(out / 'protocol.json'), 'original_labels_checked': 77,
             'news_text_prefix_checked': 77, 'image_stems_match_news_ids': 77,
             'identical_prompts_between_base_and_adapter': True, 'input_images_present': True,
             'prediction_histogram_primary44': {
                 a: dict(Counter(str(r['prediction']) for r in rows if r['news_id'] in primary)) for a, rows in predictions.items()},
             'primary_truth_histogram': dict(Counter(str(truth[n]) for n in primary)),
             'strict_and_restricted_agreement_all77': {a: sum(r['prediction'] == r['restricted_prediction'] for r in rows)
                                                     for a, rows in predictions.items()},
             'prediction_sha256': {a: digest(out / 'validation' / f'{a}.jsonl') for a in ARMS}}
    save(out / 'result_audit.json', audit)
    report = out / 'report.md'
    text = report.read_text(encoding='utf-8')
    heading = '\n## 数据与预测复核\n'
    text = text.split(heading)[0]
    text += heading + '\n'
    text += '77 条新闻已逐条与原始官方验证 TSV 核对标签及标题，并核对图片文件名与新闻 ID。四组新闻完全相同，无非法输出导致的扣分。\n\n'
    text += '主验证真实类别数量为 0:14、1:16、2:14。模型预测分布如下：\n\n'
    text += '| 组别 | 预测0 | 预测1 | 预测2 |\n|---|---:|---:|---:|\n'
    for arm, counts in audit['prediction_histogram_primary44'].items():
        text += f"| {arm} | {counts.get('0', 0)} | {counts.get('1', 0)} | {counts.get('2', 0)} |\n"
    text += '\n## 完整77条辅助验证\n\n| 组别 | Accuracy | Macro-F1 |\n|---|---:|---:|\n'
    for arm, row in summary['qwen'].items():
        metric = row['auxiliary77']['strict_output']
        text += f"| {arm} | {metric['accuracy']:.2%} | {metric['macro_f1']:.2%} |\n"
    text += '\n## 本轮能回答什么\n\n'
    text += '在这份预先固定的直接分类提示词和当前验证子集上，没有观察到原始或评论微调后的 Qwen 优于现有检测管线。评论微调没有带来直接三分类的提升。\n\n'
    text += '微调后图文＋评论组在44条主验证上全部预测为2，说明存在明显类别偏置；不能把这一结果推广为Qwen本身不具备真假识别能力。\n\n'
    text += 'MFCL检测管线已使用训练集的新闻类别标签学习分类，而这里的Qwen仅做直接提示分类，评论Adapter也不是按类别标签训练。这是实用系统对照，不是等监督量的架构比较。分类专用Qwen LoRA尚未训练；提示词、类别编号敏感性也尚未系统验证。\n\n'
    text += '生成增强E相比真实评论A提升，不代表结构已胜出：E仍略低于同内容根连接D，原测试门槛未通过。不要把本报告写成结构方法已获最终验证。\n'
    report.write_text(text, encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
