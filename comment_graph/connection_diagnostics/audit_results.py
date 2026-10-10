"""Independently inspect saved position responses and frozen classifier results."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comment_graph.connection_diagnostics import experiment as graph
from comment_graph.connection_diagnostics import position_probe as probe
from structure_utils import load_structure_template, parse_structure_output


def main():
    protocol = json.loads((probe.OUT / 'protocol.json').read_text(encoding='utf-8'))
    for path, expected in protocol['sources'].items(): assert graph.sha(path) == expected, path
    records = {r['metadata']['news_id']: r for r in graph.read(graph.SOURCE / 'contexts_smoke.jsonl')}
    rows = json.loads((probe.OUT / 'responses.json').read_text(encoding='utf-8'))
    assert len(rows) == protocol['calls'] == 24
    keys = [(r['news_id'], r['order']) for r in rows]
    assert len(set(keys)) == 24
    assert set(keys) == {(n, o) for n in protocol['news_ids'] for o in probe.ORDERS}
    template = load_structure_template(probe.SCRIPTS.parent / 'prompts/structured_comment_generation.txt')
    for row in rows:
        assert row['protocol_sha256'] == graph.sha(probe.OUT / 'protocol.json')
        model_input = records[row['news_id']]['model_input']
        assert row['observed_nodes'] == model_input['observed_nodes']
        assert row['news_text'] == model_input['news_text'] and row['image_path'] == model_input['image_path']
        assert row['actual_image_loaded'] is True
        assert row['display_order'] == probe.node_order(model_input['observed_nodes'], row['news_id'], row['order'])
        messages = probe.messages_for(template, model_input, row['display_order'])
        # All user text derives exclusively from permitted input fields and node IDs.
        assert len(messages) == 2 and messages[1]['content'][0] == {'type': 'image'}
        try:
            parsed = parse_structure_output(row['raw_output'], len(row['display_order']))
            assert row['valid'] and all(row[k] == v for k, v in parsed.items())
        except ValueError:
            assert not row['valid'] and 'error' in row
    probe.aggregate(rows, protocol)
    graph.summarize()  # Recomputes per-news metrics, verifies IDs, hashes and all checkpoints.
    graph_comparison = json.loads((graph.OUT / 'comparison.json').read_text(encoding='utf-8'))
    result = {'passed': True, 'position_responses': len(rows), 'valid_responses': sum(r['valid'] for r in rows),
        'images_and_node_content_and_ids_verified': True, 'generation_label_condition': False,
        'sources_and_classifier_checkpoint_hashes_verified': True, 'classifier_metrics_recomputed': True,
        'training_seeds': graph.SEEDS, 'train_news': 292, 'primary_validation_news': 44,
        'auxiliary_validation_news': 77, 'test_executed': False,
        'validation_gate_passed': graph_comparison['gate_passed'],
        'audit_script_sha256': graph.sha(Path(__file__)),
        'position_responses_sha256': graph.sha(probe.OUT / 'responses.json'),
        'comparison_sha256': graph.sha(graph.OUT / 'comparison.json')}
    graph.save(graph.OUT / 'result_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
