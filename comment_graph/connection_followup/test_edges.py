"""Unit-only toy graphs. Actual smoke tests use the existing eight real news."""
import unittest
import torch
from torch_geometric.data import Data
from experiment import transform_edges


def item(parents=(1, 2, 1)):
    pairs = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for node, parent in enumerate(parents, 3): pairs.extend([(parent, node), (node, parent)])
    return Data(news_id='unit', node_type=torch.tensor([0, 1, 1] + [2] * len(parents)),
                edge_index=torch.tensor(pairs).t(), edge_type=torch.tensor([0] * 4 + [1] * (2 * len(parents))),
                num_nodes=3 + len(parents), y=torch.tensor([0]))


class Edges(unittest.TestCase):
    def test_root_and_parent_edges(self):
        data = item()
        result = transform_edges(data, 'root_parent')
        self.assertEqual(result.edge_index.size(1), data.edge_index.size(1) + 6)
        pairs = set(map(tuple, result.edge_index.t().tolist()))
        for node in range(3, 6): self.assertIn((0, node), pairs)
        self.assertIn((2, 4), pairs)

    def test_root_already_present_not_duplicated(self):
        result = transform_edges(item((0, 2)), 'root_parent')
        self.assertEqual(int(result.shortcut_count), 1)
        self.assertEqual(len(set(map(tuple, result.edge_index.t().tolist()))), result.edge_index.size(1))

    def test_matched_degrees_and_determinism(self):
        data = item()
        f = transform_edges(data, 'root_parent')
        g = transform_edges(data, 'root_permuted_parent')
        self.assertTrue(torch.equal(torch.bincount(f.edge_index[0]), torch.bincount(g.edge_index[0])))
        self.assertTrue(torch.equal(g.edge_index, transform_edges(data, 'root_permuted_parent').edge_index))
        self.assertGreater(int(g.changed_parent_count), 0)

    def test_no_label_dependence(self):
        first = item(); second = item(); second.y[0] = 2
        self.assertTrue(torch.equal(transform_edges(first, 'root_permuted_parent').edge_index,
                                    transform_edges(second, 'root_permuted_parent').edge_index))

    def test_same_parent_permutation_is_unchanged(self):
        first = transform_edges(item((1, 1, 1)), 'root_parent')
        second = transform_edges(item((1, 1, 1)), 'root_permuted_parent')
        self.assertTrue(torch.equal(first.edge_index, second.edge_index))
        self.assertEqual(int(second.changed_parent_count), 0)


if __name__ == '__main__': unittest.main()
