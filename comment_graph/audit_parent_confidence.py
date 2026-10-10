"""Check whether real-parent scores distinguish observed from random edges.

An edge discriminator can be useful as a *gate* even when exact parent ranking
is too weak to justify reattaching a generated comment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from comment_graph.common import load_config
from comment_graph.reply_parent_ranker import ReplyParentRanker
from comment_graph.train_reply_parent_ranker import encode_graphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = encode_graphs(config, config_path, "validate", device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ReplyParentRanker(input_dim=int(checkpoint["input_dim"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    count = pair_count = 0
    ranker_wins = cosine_wins = 0
    ranker_pairwise_wins = cosine_pairwise_wins = 0
    ranker_margin = cosine_margin = 0.0
    with torch.inference_mode():
        for graph in graphs:
            features = graph["features"].to(device)
            parents = graph["parents"].to(device)
            ranker = model.real_parent_scores(features)
            normalized = F.normalize(features.float(), dim=-1)
            cosine = normalized[1:] @ normalized.T
            for row, true_parent in enumerate(parents.tolist()):
                child = row + 1
                if child < 2:
                    continue  # no alternative parent exists
                # Compare each true edge with every earlier alternative;
                # also report the hardest-negative result separately.
                for scores, name in ((ranker, "ranker"), (cosine, "cosine")):
                    alternatives = torch.cat((scores[row, :true_parent],
                                              scores[row, true_parent + 1:child]))
                    margin = float(scores[row, true_parent] - alternatives.max())
                    if name == "ranker":
                        ranker_wins += margin > 0
                        ranker_pairwise_wins += int((scores[row, true_parent] > alternatives).sum())
                        ranker_margin += margin
                    else:
                        cosine_wins += margin > 0
                        cosine_pairwise_wins += int((scores[row, true_parent] > alternatives).sum())
                        cosine_margin += margin
                count += 1
                pair_count += child - 1
    result = {
        "split": "validate", "news": len(graphs), "edges_with_alternative": count,
        "positive_negative_pairs": pair_count,
        "ranker_pairwise_win_rate": ranker_pairwise_wins / pair_count,
        "cosine_pairwise_win_rate": cosine_pairwise_wins / pair_count,
        "ranker_true_beats_hardest_negative_rate": ranker_wins / count,
        "cosine_true_beats_hardest_negative_rate": cosine_wins / count,
        "ranker_mean_hard_negative_margin": ranker_margin / count,
        "cosine_mean_hard_negative_margin": cosine_margin / count,
        "warning": "A low hard-negative win rate rules out using this score as a reliable parent selector; it does not label generated edges.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
