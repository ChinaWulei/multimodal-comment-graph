"""Protocol/unit checks without generation or training."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from comment_graph.run_grounded_structure import qualifies_gain,pair_audit,save,exclusive_run,contexts


def main():
    assert qualifies_gain([.02,.02,0])
    assert not qualifies_gain([.09,-.01,-.01])
    assert not qualifies_gain([.001,.001,.001])
    assert not qualifies_gain([.04,.04])
    graph={'news_id':'n','sample_id':'s','official_split':'train','image_path':'real.jpg',
           'label_used_for_generation':False,'secret_label':2,
           'graph_nodes':[{'node_index':0,'node_type':'news','content':'news'},
                          {'node_index':1,'node_type':'real_comment','content':'real','source_comment_id':'r'},
                          {'node_index':2,'node_type':'generated_comment','content':'synthetic'}],
           'generated_nodes':[{'node_index':2,'parent_node_index':1,'content':'synthetic'}],
           'edges':[{'source':0,'target':1,'edge_type':0},{'source':1,'target':2,'edge_type':1}],
           'generation_failures':[]}
    context=contexts([graph])[0]
    assert 'secret_label' not in json.dumps(context) and 'synthetic' not in json.dumps(context)
    assert set(context['model_input'])=={'image_path','news_text','observed_nodes'}
    with TemporaryDirectory(dir=ROOT/'comment_graph/outputs') as folder:
        path=Path(folder); a=path/'a.jsonl'; b=path/'b.jsonl'
        save(a,[graph],True); save(b,[graph],True)
        assert pair_audit(a,b)['same_news_images_real_nodes_and_edges']
        different=deepcopy(graph); different['graph_nodes'][1]['content']='changed real input'
        save(b,[different],True)
        try: pair_audit(a,b)
        except AssertionError: pass
        else: raise AssertionError('Changed real comments were accepted')
        with exclusive_run(path):
            try:
                with exclusive_run(path): pass
            except RuntimeError: pass
            else: raise AssertionError('Concurrent run was accepted')
        with exclusive_run(path): pass
    print('PASS: frozen improvement gate, context label isolation, paired real-node audit, concurrent job exclusion')


if __name__=='__main__': main()
