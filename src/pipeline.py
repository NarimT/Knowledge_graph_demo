"""
Pipeline skeleton: orchestrates extractors -> normalization -> KG build -> personality -> artifacts.

This is intentionally minimal: it delegates to the modules you already have:
- src.extractors
- src.normalize
- src.kg_builder
- src.llm_client
- src.personality (if you add it)
- src.eval (for evaluation)

The functions return dictionaries so tests can assert on structure.
"""
from typing import Dict, Any, List
from pathlib import Path
import json
import traceback

from src.extractors import extract_svo_from_doc, spacy_ner, get_nlp
from src.normalize import canonicalize
from src.kg_builder import build_graph, graph_to_jsonld
from src.llm_client import call_llm

# Optional imports (if you implemented them)
try:
    from src import personality as personality_module
except Exception:
    personality_module = None

def run_pipeline_for_doc(doc: Dict[str, Any], prompts: Dict[str, str] = None, use_llm: bool = False) -> Dict[str, Any]:
    """
    Run pipeline for a single document (in-memory).
    Returns a dict with intermediate artifacts:
      { 'doc_id', 'ner', 'svos', 'pred_relations', 'kg_jsonld', 'rule_personality' (if available) }
    """
    doc_id = doc.get("doc_id", "DOC")
    text = doc.get("text", "")

    out = {"doc_id": doc_id}
    # 1) spaCy NER
    try:
        ner = spacy_ner(text)
    except Exception as e:
        # spaCy model may be missing — surface error but continue
        ner = {"error": f"spaCy error: {e}"}
    out["ner"] = ner

    # 2) SVO extraction
    try:
        svo_res = extract_svo_from_doc(text)
    except Exception as e:
        svo_res = {"error": f"SVO extraction error: {e}"}
    out["svos"] = svo_res

    # 3) LLM-based RE (optional)
    llm_re = []
    if use_llm and prompts and "re" in prompts:
        prompt_path = prompts["re"]
        for sent in svo_res.get("sentences", []):
            try:
                resp = call_llm(prompt_path, {"text": sent})
            except Exception as e:
                resp = {"error": str(e)}
            llm_re.append({"sentence": sent, "resp": resp})
    out["llm_re"] = llm_re

    # 4) Produce candidate relations from SVOs (surface form)
    cand_rels = []
    for s in svo_res.get("svos", []):
        cand_rels.append({"subj_id": s.get("subj"), "pred": s.get("verb"), "obj_id": s.get("obj"), "evidence": s.get("sent_text")})
    # also include LLM results if available
    for item in llm_re:
        resp = item.get("resp", {})
        if isinstance(resp, dict) and resp.get("extracted"):
            for ex in resp["extracted"]:
                cand_rels.append({"subj_id": ex.get("subj"), "pred": ex.get("pred"), "obj_id": ex.get("obj"), "evidence": ex.get("evidence")})

    out["candidate_relations_raw"] = cand_rels

    # 5) Normalize (requires doc.gold_entities to map text->id if present)
    gold_entities = doc.get("gold_entities", [])
    # Convert cand_rels into the format canonicalize expects: subj_id may be surface text; we will keep them
    simple_rels = []
    for r in cand_rels:
        simple_rels.append({"subj_id": r["subj_id"], "pred": r["pred"], "obj_id": r["obj_id"]})
    try:
        canon = canonicalize(gold_entities, simple_rels)
        out["normalized"] = canon
    except Exception as e:
        out["normalized"] = {"error": f"canonicalize error: {e}"}

    # 6) Build KG
    try:
        entities = canon.get("entities", []) if isinstance(canon, dict) else gold_entities
        relations = canon.get("relations", []) if isinstance(canon, dict) else simple_rels
        G = build_graph(entities, relations)
        jsonld = graph_to_jsonld(G)
        out["kg_jsonld"] = jsonld
    except Exception as e:
        out["kg_jsonld"] = {"error": f"kg build error: {e}"}

    # 7) Personality (rule-based baseline if module available)
    if personality_module and hasattr(personality_module, "rule_based_personality_for_person"):
        try:
            persons_scores = {}
            for e in gold_entities:
                if e.get("type") == "Person":
                    persons_scores[e["id"]] = {"big5": personality_module.rule_based_personality_for_person(e["text"], text)}
            out["rule_personality"] = persons_scores
        except Exception as e:
            out["rule_personality"] = {"error": f"personality scoring error: {e}"}
    else:
        out["rule_personality"] = {"skipped": "personality module not found"}

    return out

# small CLI helper
def run_and_save(doc: Dict[str, Any], outdir: str = "notebooks_output", use_llm: bool = False, prompts: Dict[str,str] = None):
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    out = run_pipeline_for_doc(doc, prompts=prompts, use_llm=use_llm)
    out_path = p / f"pipeline_{doc.get('doc_id','doc')}.json"
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return str(out_path)
