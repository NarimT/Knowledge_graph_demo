"""
Evaluation utilities for KG + personality pipeline.

Implements:
- relation-level evaluation: precision, recall, F1 (per-doc and corpus)
- error analysis: top 3 FP triples, top 3 FN triples
- personality evaluation: MAE per trait, Pearson r per trait
- personality error analysis: top 3 worst MAE cases (by absolute error sum)

Usage (example):
python -m src.eval --gold data/synthetic_v1.json --pred_artifacts notebooks_output/pipeline_artifacts.json --out report/eval_report.json

This file also contains a commented backup code snippet (from Prompt M) that provides a short alternative implementation.
"""

import json
import argparse
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any
import math
import os
from pprint import pprint

import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# -------------------------
# Helper utilities
# -------------------------
def triple_to_tuple(tr: Dict[str, Any]) -> Tuple[str, str, str]:
    """Normalize relation dict -> tuple (subj, pred, obj)."""
    return (tr.get("subj_id"), tr.get("pred"), tr.get("obj_id"))

def load_gold(gold_path: str) -> List[Dict[str, Any]]:
    with open(gold_path, "r", encoding="utf8") as f:
        return json.load(f)

def load_pred_artifacts(artifacts_path: str) -> Dict[str, Any]:
    """Load pipeline_artifacts.json or similar structure produced in notebook."""
    with open(artifacts_path, "r", encoding="utf8") as f:
        return json.load(f)

# -------------------------
# Relation evaluation
# -------------------------
def evaluate_relations(gold_docs: List[Dict[str,Any]], pred_relations: Dict[str, List[Dict[str,Any]]]):
    """
    gold_docs: list of documents (each with 'doc_id' and 'gold_relations' list of dicts {subj_id,pred,obj_id})
    pred_relations: dict doc_id -> list of relation dicts {subj_id,pred,obj_id}
    Returns: dict with per-doc metrics and corpus-level metrics + FP/FN lists
    """
    corpus_tp = corpus_fp = corpus_fn = 0
    fp_counter = Counter()
    fn_counter = Counter()
    per_doc_metrics = {}

    for doc in gold_docs:
        doc_id = doc["doc_id"]
        gold = doc.get("gold_relations", [])
        gold_set = set((g["subj_id"], g["pred"], g["obj_id"]) for g in gold)
        preds = pred_relations.get(doc_id, [])
        pred_set = set((p["subj_id"], p["pred"], p["obj_id"]) for p in preds)

        tp_set = gold_set & pred_set
        fp_set = pred_set - gold_set
        fn_set = gold_set - pred_set

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        corpus_tp += tp
        corpus_fp += fp
        corpus_fn += fn

        for e in fp_set:
            fp_counter[e] += 1
        for e in fn_set:
            fn_counter[e] += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_doc_metrics[doc_id] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    corpus_precision = corpus_tp / (corpus_tp + corpus_fp) if (corpus_tp + corpus_fp) > 0 else 0.0
    corpus_recall = corpus_tp / (corpus_tp + corpus_fn) if (corpus_tp + corpus_fn) > 0 else 0.0
    corpus_f1 = 2 * corpus_precision * corpus_recall / (corpus_precision + corpus_recall) if (corpus_precision + corpus_recall) > 0 else 0.0

    top3_fp = [ {"triple": t, "count": c} for t, c in fp_counter.most_common(3) ]
    top3_fn = [ {"triple": t, "count": c} for t, c in fn_counter.most_common(3) ]

    return {
        "per_doc": per_doc_metrics,
        "corpus": {"tp": corpus_tp, "fp": corpus_fp, "fn": corpus_fn, "precision": corpus_precision, "recall": corpus_recall, "f1": corpus_f1},
        "top3_fp": top3_fp,
        "top3_fn": top3_fn
    }

# -------------------------
# Personality evaluation
# -------------------------
def evaluate_personality(gold_docs: List[Dict[str,Any]], pred_personality: Dict[str, Dict[str, Any]]):
    """
    gold_docs: list of documents; gold personality is under doc['personality_labels'] mapping person_id -> {big5: {...}}
    pred_personality: dict doc_id -> person_id -> {big5: {...}} (e.g., rule_personality from artifacts)
    Returns: metrics per trait (MAE, Pearson r), and worst-case list (top 3 persons by total abs error)
    """
    traits = ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]
    all_golds = {t: [] for t in traits}
    all_preds = {t: [] for t in traits}
    per_person_errors = []  # tuples (doc_id, person_id, total_abs_error, per_trait_errors)

    for doc in gold_docs:
        doc_id = doc["doc_id"]
        gold_map = doc.get("personality_labels", {})
        pred_map = pred_personality.get(doc_id, {})

        for pid, gold_obj in gold_map.items():
            gold_scores = gold_obj.get("big5", {})
            pred_obj = pred_map.get(pid)
            if pred_obj is None:
                # skip missing predictions
                continue
            pred_scores = pred_obj.get("big5", {})

            total_abs = 0.0
            per_trait_err = {}
            has_pair = False
            for t in traits:
                g = gold_scores.get(t)
                p = pred_scores.get(t)
                if g is None or p is None:
                    continue
                has_pair = True
                all_golds[t].append(g)
                all_preds[t].append(p)
                err = abs(g - p)
                per_trait_err[t] = err
                total_abs += err
            if has_pair:
                per_person_errors.append({"doc_id": doc_id, "person_id": pid, "total_abs_error": round(total_abs, 4), "per_trait_errors": per_trait_err})

    trait_metrics = {}
    for t in traits:
        if len(all_golds[t]) == 0:
            trait_metrics[t] = {"mae": None, "pearson_r": None, "n": 0}
            continue
        mae = float(mean_absolute_error(all_golds[t], all_preds[t]))
        pearson_r = None
        try:
            if len(all_golds[t]) > 1:
                pearson_r = float(pearsonr(all_golds[t], all_preds[t])[0])
        except Exception:
            pearson_r = None
        trait_metrics[t] = {"mae": round(mae,4), "pearson_r": pearson_r, "n": len(all_golds[t])}

    # worst 3 persons by total_abs_error
    per_person_errors_sorted = sorted(per_person_errors, key=lambda x: -x["total_abs_error"])
    worst3 = per_person_errors_sorted[:3]

    return {"trait_metrics": trait_metrics, "worst3_persons": worst3}

# -------------------------
# Main / CLI
# -------------------------
def main(args):
    gold = load_gold(args.gold)
    artifacts = load_pred_artifacts(args.pred_artifacts)

    # expected artifact structure: we saved 'pred_relations' and 'rule_personality' etc.
    pred_rel = artifacts.get("pred_relations", {})
    pred_person = artifacts.get("rule_personality", {})

    rel_eval = evaluate_relations(gold_docs=gold, pred_relations=pred_rel)
    pers_eval = evaluate_personality(gold_docs=gold, pred_personality=pred_person)

    out = {"relation_evaluation": rel_eval, "personality_evaluation": pers_eval}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Saved evaluation report to", args.out)
    print("\nCorpus relation metrics:")
    pprint(rel_eval["corpus"])
    print("\nTop 3 FP triples:")
    pprint(rel_eval["top3_fp"])
    print("\nTop 3 FN triples:")
    pprint(rel_eval["top3_fn"])

    print("\nPersonality trait metrics:")
    pprint(pers_eval["trait_metrics"])
    print("\nWorst 3 persons by total abs error:")
    pprint(pers_eval["worst3_persons"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, default="data/synthetic_v1.json", help="Path to gold JSON file")
    parser.add_argument("--pred_artifacts", type=str, default="notebooks_output/pipeline_artifacts.json", help="Path to pipeline artifacts with predictions")
    parser.add_argument("--out", type=str, default="report/eval_report.json", help="Path to save evaluation JSON")
    args = parser.parse_args()
    main(args)


# -------------------------
# Backup snippet (from Prompt M)
# -------------------------
# The following minimal snippet was produced by Prompt M (included as a backup)
# (it is intentionally short; keep as a reference)
#
# def simple_relation_prf(gold_triples, pred_triples):
#     gold_set = set(gold_triples)
#     pred_set = set(pred_triples)
#     tp = len(gold_set & pred_set)
#     fp = len(pred_set - gold_set)
#     fn = len(gold_set - pred_set)
#     precision = tp/(tp+fp) if (tp+fp)>0 else 0
#     recall = tp/(tp+fn) if (tp+fn)>0 else 0
#     f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
#     return precision, recall, f1
#
# (End of backup snippet.)
