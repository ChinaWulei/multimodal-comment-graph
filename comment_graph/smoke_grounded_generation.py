"""Offline regression tests for frozen real-context generation and resume safety."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'comment_generator/partial_observation/scripts'
sys.path.insert(0, str(SCRIPTS))
import generate_structured_comments as generator
from structure_utils import build_structure_messages, parse_structure_output
import validate_generated_structured_graphs as validator


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid input accepted')


def main():
    original = {'news_text': 'News only', 'image_path': 'actual-image.jpg', 'observed_nodes': [
        {'node_index': 1, 'parent_node_index': 0, 'content': 'Real only'}]}
    current = copy.deepcopy(original)
    current['observed_nodes'].append({'node_index': 2, 'parent_node_index': 1, 'content': 'Synthetic secret'})
    frozen = generator.generation_view(current, original, 'observed_only')
    assert frozen == original
    assert generator.generation_view(current, original, 'iterative') == current
    frozen['observed_nodes'][0]['content'] = 'Mutation'
    assert original['observed_nodes'][0]['content'] == 'Real only'
    template = '{news_text}\n{observed_nodes}'
    messages = build_structure_messages(template, original, allowed_parent_indices=[0, 1])
    assert 'Synthetic secret' not in json.dumps(messages)
    assert messages[1]['content'][0]['type'] == 'image'
    rejects(lambda: build_structure_messages(template, original, allowed_parent_indices=[2]))
    for text in ['{}', '{"parent_node_index":true,"content":"x"}',
                 '{"parent_node_index":2,"content":"x"}', '{"parent_node_index":0,"content":null}',
                 '{"parent_node_index":0,"content":42}', '{"parent_node_index":0,"content":"  "}']:
        rejects(lambda: parse_structure_output(text, 1))
    assert parse_structure_output('{"parent_node_index":1,"content":" reply "}', 1)['content'] == 'reply'
    with tempfile.TemporaryDirectory(dir=ROOT / 'comment_graph/outputs') as name:
        folder = Path(name)
        metadata = folder / 'run.json'
        contract = {'policy': 'observed_only', 'input': 'hash'}
        generator.atomic_json(metadata, {'contract': contract})
        records = [{'sample_id': 's', 'metadata': {'news_id': 'n'}}]
        generator.verify_resume([{'news_id': 'n', 'sample_id': 's'}], records, contract, metadata)
        rejects(lambda: generator.verify_resume([], records, {**contract, 'policy': 'iterative'}, metadata))
        rejects(lambda: generator.verify_resume([{'news_id': 'other', 'sample_id': 's'}], records, contract, metadata))
        graph = {'sample_id': 's', 'news_id': 'n', 'official_split': 'train', 'image_path': 'image.jpg',
                 'graph_nodes': [{'node_index': 0, 'node_type': 'news', 'content': 'news'},
                                 {'node_index': 1, 'node_type': 'real_comment', 'content': 'real', 'source_comment_id': 'r'},
                                 {'node_index': 2, 'node_type': 'generated_comment', 'content': 'new',
                                  'generated_node_id': 'g', 'parent_comment_id': 'r'}],
                 'edges': [{'source': 0, 'target': 1, 'edge_type': 0}, {'source': 1, 'target': 2, 'edge_type': 1}],
                 'generated_nodes': [{'node_index': 2, 'parent_node_index': 1, 'content': 'new',
                                      'generated_node_id': 'g', 'parent_comment_id': 'r'}],
                 'generation_failures': [], 'adapter_dir': 'adapter',
                 'generation_config': {'context_policy': 'observed_only', 'depth_reference': 'observed_boundary',
                                       'max_generated_depth': 6},
                 'parent_selection_policy': 'model_predicted_news_or_real_only',
                 'label_used_for_generation': False, 'time_interpretation': 'not time'}
        path = folder / 'graphs.jsonl'
        def check(value):
            path.write_text(json.dumps(value) + '\n', encoding='utf8')
            with patch.object(sys, 'argv', ['validate', '--input-jsonl', str(path), '--expected-split', 'train']):
                validator.main()
        check(graph)
        bad = copy.deepcopy(graph)
        bad['graph_nodes'].append({'node_index': 3, 'node_type': 'generated_comment', 'content': 'bad',
                                  'generated_node_id': 'g2', 'parent_comment_id': 'g'})
        bad['generated_nodes'].append({'node_index': 3, 'parent_node_index': 2, 'content': 'bad',
                                      'generated_node_id': 'g2', 'parent_comment_id': 'g'})
        bad['edges'].append({'source': 2, 'target': 3, 'edge_type': 1})
        rejects(lambda: check(bad))
    print('PASS: isolated real context, real image slot, strict JSON, resume mismatch, forbidden synthetic parents')


if __name__ == '__main__':
    main()
