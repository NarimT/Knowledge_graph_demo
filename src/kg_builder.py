"""Build a NetworkX graph from normalized triples and export JSON-LD.

Functions:
- build_graph(entities, relations) -> networkx.MultiDiGraph
- graph_to_jsonld(G) -> dict (JSON-LD structure)
- save_jsonld(outpath, jsonld)
"""
from typing import List, Dict
import networkx as nx
import json


def build_graph(entities: List[Dict], relations: List[Dict]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    # add nodes
    for e in entities:
        nid = e["id"]
        G.add_node(nid, **{"text": e.get("text"), "type": e.get("type")})
    # add edges
    for r in relations:
        subj = r["subj_id"]
        obj = r["obj_id"]
        pred = r.get("pred")
        attrs = {k: v for k, v in r.items() if k not in ("subj_id", "obj_id")}
        G.add_edge(subj, obj, key=pred, **attrs)
    return G


def graph_to_jsonld(G: nx.MultiDiGraph) -> Dict:
    """Simple JSON-LD-ish export: nodes as objects and edges as relationships.

    This is intentionally small and human-readable; for production use rdflib/jsonld libraries.
    """
    nodes = []
    edges = []
    for n, data in G.nodes(data=True):
        nodes.append({"@id": n, "@type": data.get("type"), "label": data.get("text")})
    for u, v, k, data in G.edges(keys=True, data=True):
        edge = {"@type": k, "subj": u, "obj": v}
        edge.update(data)
        edges.append(edge)
    return {"@context": {"ex": "http://example.org/kg#"}, "@graph": {"nodes": nodes, "edges": edges}}


def save_jsonld(outpath: str, jsonld: Dict):
    with open(outpath, "w", encoding="utf8") as f:
        json.dump(jsonld, f, indent=2, ensure_ascii=False)