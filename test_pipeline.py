import pytest
import json
import os
from pathlib import Path
from pprint import pprint

from src.extractors import get_nlp, spacy_ner, extract_svo_from_doc
from src.normalize import canonicalize
from src.kg_builder import build_graph, graph_to_jsonld
from src.llm_client import call_llm

# Import eval functions if available
try:
    from src.eval import evaluate_relations, evaluate_personality
except Exception:
    evaluate_relations = None
    evaluate_personality = None

# small sample doc (D1)
SAMPLE_DOC = {
    "doc_id": "D1",
    "text": "Alice joined Acme Corp as a project manager last month. She organized weekly meetings and consistently met deadlines to keep the team on schedule.",
    "gold_entities": [
        { "id": "D1_E1", "text": "Alice", "type": "Person" },
        { "id": "D1_E2", "text": "Acme Corp", "type": "Organization" }
    ],
    "gold_relations": [
        { "subj_id": "D1_E1", "pred": "worksAt", "obj_id": "D1_E2" },
    ],
    "personality_labels": {
        "D1_E1": {
            "big5": {
                "openness": 0.60,
                "conscientiousness": 0.92,
                "extraversion": 0.70,
                "agreeableness": 0.75,
                "neuroticism": 0.20
            },
            "explanation": "sample"
        }
    }
}

def test_spacy_and_svo_runs_or_skip():
    """
    If spaCy model is missing, skip test politely.
    """
    try:
        nlp = get_nlp()
    except RuntimeError as e:
        pytest.skip(str(e))
    # run NER
    ents = spacy_ner(SAMPLE_DOC["text"])
    assert isinstance(ents, list)
    # run SVO extraction
    svo = extract_svo_from_doc(SAMPLE_DOC["text"])
    assert "sentences" in svo and "svos" in svo

def test_normalize_and_kg_build(tmp_path):
    # canonicalize
    ent_list = SAMPLE_DOC["gold_entities"]
    rel_list = [{"subj_id":"Alice", "pred":"joined", "obj_id":"Acme Corp"}]
    canon = canonicalize(ent_list, rel_list)
    assert "entities" in canon and "relations" in canon
    # build graph
    G = build_graph(canon["entities"], canon["relations"])
    assert G.number_of_nodes() >= 2

def test_llm_client_simulator(tmp_path):
    # create a temp prompt file
    p = tmp_path / "prompt_test.txt"
    p.write_text("Extract: {input_text}")
    resp = call_llm(str(p), {"text": SAMPLE_DOC["text"]})
    assert isinstance(resp, dict)
    # simulator returns 'simulated' True; allow also real LLM dicts
    assert "simulated" in resp or "extracted" in resp or "text" in resp

def test_eval_functions_smoke(tmp_path):
    if evaluate_relations is None or evaluate_personality is None:
        pytest.skip("src.eval not available")
    # prepare simple pred_relations structure
    pred_rel = {"D1": [{"subj_id":"D1_E1","pred":"worksAt","obj_id":"D1_E2"}]}
    rel_res = evaluate_relations([SAMPLE_DOC], pred_rel)
    assert "corpus" in rel_res and "per_doc" in rel_res
    # personality eval: make pred same as gold to get zero errors
    pred_person = {"D1": {"D1_E1": {"big5": SAMPLE_DOC["personality_labels"]["D1_E1"]["big5"]}}}
    pers_res = evaluate_personality([SAMPLE_DOC], pred_person)
    assert "trait_metrics" in pers_res and "worst3_persons" in pers_res
