import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comment_graph.attachment_consistency.probe import build_messages, parse_parent, consensus, real_leaf_case, PROMPT


class ProbeTests(unittest.TestCase):
    def record(self):
        return {'metadata': {'news_id': 'unit', 'observed_node_comment_ids': {'1': 'a', '2': 'b', '3': 'c'}},
            'model_input': {'news_text': 'A post', 'image_path': 'unit.png', 'observed_nodes': [
                {'node_index': 1, 'parent_node_index': 0, 'content': 'real first'},
                {'node_index': 2, 'parent_node_index': 1, 'content': 'real second'},
                {'node_index': 3, 'parent_node_index': 1, 'content': 'real third'}]}}

    def test_leaf_hidden_edges_remapped(self):
        case = real_leaf_case(self.record())
        self.assertEqual(len(case['model_input']['observed_nodes']), 2)
        self.assertNotIn(case['target_content'], [n['content'] for n in case['model_input']['observed_nodes']])
        self.assertEqual(case['expected_parent'], 1)

    def test_no_truth_or_old_prediction_in_prompt(self):
        case = real_leaf_case(self.record()); template = PROMPT.read_text(encoding='utf-8')
        a = build_messages(case, [1, 2], template)
        case['expected_parent'] = 918273; case['original_predicted_parent'] = 817263; case['label'] = 'FORBIDDEN_LABEL'
        self.assertEqual(a, build_messages(case, [1, 2], template))
        self.assertNotEqual(a, build_messages(case, [2, 1], template))

    def test_strict_json(self):
        self.assertEqual(parse_parent('{"parent_node_index": 2}', {0, 1, 2}), 2)
        for value in ['{"parent_node_index":true}', '{"parent_node_index":99}', '{"parent_node_index":1,"content":"x"}', '```json\n{}\n```']:
            with self.assertRaises(ValueError): parse_parent(value, {0, 1, 2})

    def test_abstention_is_not_ground_truth(self):
        self.assertEqual(consensus([1, 1, 2]), (1, 'majority'))
        self.assertEqual(consensus([1, 2, 3]), (0, 'root_abstention'))
        self.assertEqual(consensus([None, None, 1]), (0, 'root_abstention'))
        self.assertEqual(consensus([0, 0, 2]), (0, 'majority'))


if __name__ == '__main__': unittest.main()
