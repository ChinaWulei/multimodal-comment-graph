from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comment_graph.attachment_consistency.augmented_sft import permute_record, canonical_validate, add_root_context, messages


class AugmentationTests(unittest.TestCase):
    def record(self):
        return {'sample_id': 'unit', 'metadata': {'news_id': 'news', 'target_comment_id': 'target'},
            'model_input': {'news_text': 'title', 'image_path': 'image.jpg', 'observed_nodes': [
                {'node_index': 1, 'parent_node_index': 0, 'content': 'one'},
                {'node_index': 2, 'parent_node_index': 1, 'content': 'two'},
                {'node_index': 3, 'parent_node_index': 1, 'content': 'three'}]},
            'target': {'parent_node_index': 2, 'content': 'withheld'}, 'target_json': ''}

    def test_bijection_and_target(self):
        original = self.record(); before = deepcopy(original)
        augmented = permute_record(original)
        self.assertEqual(original, before)
        mapping = augmented['augmentation']['old_to_new_ids']
        self.assertEqual(augmented['target']['parent_node_index'], mapping[2])
        self.assertEqual(augmented, permute_record(original))
        self.assertEqual(augmented['target']['content'], 'withheld')

    def test_cycle_rejected(self):
        source = self.record()['model_input']; source['observed_nodes'][0]['parent_node_index'] = 2
        with self.assertRaises(ValueError): canonical_validate(source)

    def test_root_context_other_branch_only(self):
        record = self.record(); record['target']['parent_node_index'] = 0; record['model_input']['observed_nodes'] = []
        tree = {'roots': [{'comment_id': 'target', 'parent_id': 'news', 'body': 'withheld', 'children': [
                    {'comment_id': 'descendant', 'parent_id': 'target', 'body': 'leaking reply', 'children': []}]},
                {'comment_id': 'other', 'parent_id': 'news', 'body': 'other branch', 'children': []}]}
        changed = add_root_context(record, tree)
        self.assertEqual([n['content'] for n in changed['model_input']['observed_nodes']], ['other branch'])
        self.assertEqual(changed['target']['parent_node_index'], 0)


if __name__ == '__main__': unittest.main()
