"""Synthetic unit tests only; experiment smoke separately uses eight actual news."""
import sys
from pathlib import Path
import unittest
import torch
from torch_geometric.data import Data
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comment_graph.connection_diagnostics.experiment import transform_edges
from comment_graph.connection_diagnostics.position_probe import messages_for, node_order, SCRIPTS
from structure_utils import build_structure_messages, load_structure_template


def item(parents=(1, 2, 1)):
    pairs = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for node, parent in enumerate(parents, 3): pairs += [(parent, node), (node, parent)]
    return Data(news_id='unit', node_type=torch.tensor([0, 1, 1] + [2] * len(parents)),
        edge_index=torch.tensor(pairs).t(), edge_type=torch.tensor([0]*4 + [1]*(2*len(parents))),
        num_nodes=3+len(parents), y=torch.tensor([0]))


class Protocol(unittest.TestCase):
    def test_preserves_counts_degrees_real_edges(self):
        a = item(); b = transform_edges(a)
        self.assertEqual(a.edge_index.shape, b.edge_index.shape)
        self.assertTrue(torch.equal(torch.bincount(a.edge_index[0]), torch.bincount(b.edge_index[0])))
        self.assertGreater(int(b.changed_parent_count), 0)
        self.assertFalse(any(s == 0 and t >= 3 for s, t in b.edge_index.t().tolist()))

    def test_original_root_kept(self):
        a = item((0, 1, 2)); b = transform_edges(a)
        self.assertIn((0, 3), map(tuple, b.edge_index.t().tolist()))
        self.assertEqual(int(b.shortcut_count), 0)

    def test_no_generated_and_same_parents(self):
        for parents in [(), (1, 1, 1)]:
            a = item(parents); b = transform_edges(a)
            self.assertTrue(torch.equal(a.edge_index, b.edge_index))

    def test_deterministic_and_label_blind(self):
        a = item(); b = item(); b.y[0] = 2
        self.assertTrue(torch.equal(transform_edges(a).edge_index, transform_edges(b).edge_index))

    def test_display_order_only(self):
        template = load_structure_template(SCRIPTS.parent / 'prompts/structured_comment_generation.txt')
        data = {'news_text': 'A test title', 'image_path': 'unit.png', 'observed_nodes': [
            {'node_index': 1, 'parent_node_index': 0, 'content': 'one'},
            {'node_index': 2, 'parent_node_index': 1, 'content': 'two'}]}
        original = messages_for(template, data, [1, 2])
        self.assertEqual(original, build_structure_messages(template, data, allowed_parent_indices=[0, 1, 2]))
        reverse = messages_for(template, data, [2, 1])
        self.assertEqual(original[0], reverse[0])
        self.assertIn('"node_index":2,"parent_node_index":1,"content":"two"', reverse[1]['content'][1]['text'])
        self.assertEqual(data['observed_nodes'][0]['node_index'], 1)
        self.assertNotEqual(node_order(data['observed_nodes'], 'unit', 'shuffle')[-1], 2)
        with self.assertRaises(ValueError): messages_for(template, data, [1, 1])


if __name__ == '__main__': unittest.main()
