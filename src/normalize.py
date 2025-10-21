"""Normalization helpers: mapping table and canonicalize() function.

Functions:
- canonicalize_entities(entities): simple normalization of names
- normalize_relation(pred): map surface predicates to canonical ones
- canonicalize(entities, relations): returns normalized entities + relations
"""
from typing import List, Dict, Tuple
import re

# mapping table for predicates (surface -> canonical)
PREDICATE_MAP = {
    "joined": "worksAt",
    "joined_with": "worksAt",
    "worksAt": "worksAt",
    "left": "leftCompany",
    "founded": "founded",
    "promotedTo": "promotedTo",
    "role": "hasRole",
    "occupation": "hasRole",
    "contractedWith": "contractedWith",
    "organizedWith": "organizedWith",
    "consultedFor": "consultedFor",
    "volunteersAt": "volunteersAt",
    "mentorsAt": "mentorsAt",
    "partneredWith": "partneredWith",
    "recruitedFrom": "recruitedFrom",
    "works_at": "worksAt",
}


def _simple_canonical_name(text: str) -> str:
    t = text.strip()
    # remove honorifics and punctuation
    t = re.sub(r"^(Dr\.|Mr\.|Mrs\.|Ms\.)\s+", "", t)
    t = re.sub(r"[^\w\s\-&]", "", t)
    # normalize whitespace and case
    t = re.sub(r"\s+", " ", t)
    return t


def canonicalize_entities(entities: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Return canonicalized entities and a map from original text -> canonical id/name.

    Entities are list of dicts {id, text, type}
    """
    canon_map = {}
    out = []
    for e in entities:
        canon_text = _simple_canonical_name(e["text"])  # preserve case as cleaned
        # create a canonical id if not present
        eid = e.get("id") or f"E_{len(out)+1}"
        out.append({"id": eid, "text": canon_text, "type": e.get("type")})
        canon_map[e["text"]] = canon_text
    return out, canon_map


def normalize_relation(pred: str) -> str:
    return PREDICATE_MAP.get(pred, pred)


def canonicalize(entities: List[Dict], relations: List[Dict]) -> Dict:
    """Main canonicalize function.

    Returns: {entities: [...], relations: [...]}
    """
    ents_canon, cmap = canonicalize_entities(entities)
    # normalize relations
    rels_out = []
    for r in relations:
        pred = r.get("pred")
        pred_norm = normalize_relation(pred)
        subj = r.get("subj_id")
        obj = r.get("obj_id")
        # leave ids as-is but ensure they map to canonical texts
        rels_out.append({"subj_id": subj, "pred": pred_norm, "obj_id": obj})
    return {"entities": ents_canon, "relations": rels_out}