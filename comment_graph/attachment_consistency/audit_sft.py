"""Quantify index shortcuts in existing generator SFT, without altering any sample."""
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comment_graph.attachment_consistency.probe import SCRIPTS, HERE, read, sha, save


def main():
    path = SCRIPTS.parent / 'data/official_subset/structure_train_expanded.jsonl'
    rows = read(path)
    counts, sizes = Counter(), Counter()
    for r in rows:
        assert r['metadata']['official_split'] == 'train'
        n = len(r['model_input']['observed_nodes']); parent = r['target']['parent_node_index']
        assert 0 <= parent <= n
        counts['rows'] += 1; sizes[n] += 1
        counts['nonempty_context'] += n > 0
        counts['root_parent'] += parent == 0
        counts['root_parent_with_nonempty_context'] += parent == 0 and n > 0
        counts['last_parent_nonempty'] += n > 0 and parent == n
        counts['other_parent_nonempty'] += n > 0 and parent != n
    result = {'source': str(path), 'sha256': sha(path), 'counts': dict(counts), 'context_sizes': dict(sizes),
        'news': len({r['metadata']['news_id'] for r in rows}),
        'last_index_rule_accuracy_nonempty': counts['last_parent_nonempty']/counts['nonempty_context'],
        'last_index_else_root_rule_accuracy_all': (counts['last_parent_nonempty'] + sizes[0])/len(rows),
        'interpretation': 'Training association only, not causal proof or held-out parent accuracy.',
        'potential_confounds': ['Tree traversal creates many immediately preceding parents.',
                               'Root targets have no competing observed comments in these SFT examples.'],
        'planned_controls': ['Freeze target content.', 'Separate node ID bias from display-order bias.',
                             'Use real parent IDs for evaluation; never pretend generated parents are truth.']}
    save(HERE / 'outputs/sft_distribution_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
