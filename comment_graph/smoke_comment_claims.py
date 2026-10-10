"""Parser and input-isolation unit tests; synthetic fixtures, no quality claim."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comment_graph.score_comment_claims import input_digest, parse_claims


def main():
    row = {"headline": "A red car", "parent_content": "It looks red", "parent_node_type": "real_comment",
           "parent_node_index": 1, "recent_ancestors": [], "generated_comment": "It is red and costs $500."}
    data = {"claims": [{"span": "It is red", "status": "supported", "source": "headline", "evidence": "red"},
                       {"span": "costs $500", "status": "unsupported", "source": "none", "evidence": ""}]}
    parsed = parse_claims(json.dumps(data), row)
    assert parsed["claim_count"] == 2 and parsed["claim_unsupported_fraction"] == .5
    assert parse_claims('{"claims":[]}', row)["claim_count"] == 0
    assert input_digest(row) == input_digest({**row, "label": 2, "content_relevance": "off_topic",
                                             "parent_fit": "wrong", "unsupported_specific_claim": True})
    assert input_digest(row) != input_digest({**row, "generated_comment": "different"})
    negatives = ["not json", '```json\n{"claims":[]}\n```']
    for field, value in (("span", "fabricated quotation"), ("source", "unvisited_url"),
                         ("evidence", "the price is $500"), ("status", "true")):
        broken = json.loads(json.dumps(data))
        broken["claims"][0][field] = value
        negatives.append(json.dumps(broken))
    negatives.append('{"claims":[{"span":"It is red","status":"supported","source":"none","evidence":""}]}')
    for raw in negatives:
        try:
            parse_claims(raw, row)
        except ValueError:
            pass
        else:
            raise AssertionError(raw)
    print(f"Claim parser and label-isolation smoke passed; {len(negatives)} invalid outputs rejected.")


if __name__ == "__main__":
    main()
