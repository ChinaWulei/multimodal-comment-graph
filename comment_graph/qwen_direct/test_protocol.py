"""CPU unit tests; mock inputs here are not reported as experimental samples."""
import json
import unittest
from run import messages, metrics, parse_label, object_digest


class ProtocolTests(unittest.TestCase):
    def test_strict_parser(self):
        self.assertEqual(parse_label(' 2\n'), 2)
        for output in ['true', '{"label":1}', '1 because...', '', '3', '-1']:
            self.assertEqual(parse_label(output), -1)

    def test_prompt_allowlist(self):
        row = {'news_id': 'secret-id', 'label': 'secret-label', 'image_path': 'secret-path',
               'subreddit': 'secret-source', 'news_text': 'title',
               'comments': [{'node_index': 1, 'parent_node_index': 0, 'content': 'discussion'}]}
        with_comments = json.dumps(messages(row, True))
        for secret in ['secret-id', 'secret-label', 'secret-path', 'secret-source']:
            self.assertNotIn(secret, with_comments)
        self.assertIn('discussion', with_comments)
        self.assertNotIn('discussion', json.dumps(messages(row, False)))
        self.assertEqual(messages(row, True)[1]['content'][0]['type'], 'image')

    def test_invalid_stays_in_denominator(self):
        rows = [{'news_id': 'a', 'prediction': 0}, {'news_id': 'b', 'prediction': -1},
                {'news_id': 'c', 'prediction': 2}]
        result = metrics(rows, {'a': 0, 'b': 1, 'c': 2}, 'prediction')
        self.assertAlmostEqual(result['accuracy'], 2/3)
        self.assertAlmostEqual(result['macro_f1'], 2/3)
        self.assertEqual(result['confusion_matrix_true_012_pred_012invalid'][1][3], 1)

    def test_metrics_require_exact_cohort(self):
        with self.assertRaises(AssertionError):
            metrics([{'news_id': 'a', 'prediction': 0}], {'a': 0, 'b': 1}, 'prediction')

    def test_digest_sensitive_to_payload(self):
        self.assertNotEqual(object_digest({'image': 'a'}), object_digest({'image': 'b'}))

    def test_protocol_roundtrip(self):
        value = {'arms': {'base_mm': (False, False)}}
        self.assertEqual(object_digest(value), object_digest(json.loads(json.dumps(value))))


if __name__ == '__main__':
    unittest.main()
