# Design choices for KG + Personality Pipeline

This document summarizes the core design decisions for the reproducible, local KG + personality extraction pipeline.

## 1. Core NLP: spaCy
**Choice:** Use spaCy (`en_core_web_sm` or `en_core_web_md`) for tokenization, sentence splitting, NER, POS and dependency parsing.  
**Why:** Fast, well-documented, easy to run locally; reliable for short text and allows deterministic behavior for reproducibility.  
**Risk:** Off-the-shelf models can miss domain-specific entities; consider fine-tuning or adding custom NER rules where required.

## 2. Relation extraction: hybrid (OpenIE + rule-based + optional LLM)
**Choice:** Combine OpenIE-style extraction for high recall with rule-based patterns for precision, and an optional LLM pass for difficult cases.  
**Why:** Hybrid systems balance recall and precision: OpenIE surfaces many candidates, rules filter high-precision patterns, and LLMs help on ambiguous phrasing.  
**Risk:** OpenIE can be noisy and LLM outputs must be validated; maintain provenance and evidence to avoid introducing hallucinations.

## 3. Entity canonicalization: local mapping + fuzzy matching
**Choice:** Use a small SQLite/CSV local KB and fuzzy matching (`rapidfuzz`) to canonicalize mentions. Avoid remote lookups (Wikidata) to keep reproducibility and no-private-data requirements.  
**Why:** Keeps pipeline local and reproducible; small KB is easier to version-control.  
**Risk:** String matching can produce false merges; ensure conservative thresholds and human-in-the-loop corrections.

## 4. Trait representation: reified TraitAssertion nodes
**Choice:** Represent personality as reified nodes (`TraitAssertion`) linking a Person to a PersonalityTrait, storing `score`, `confidence`, `measuredBy`, `measuredAt`, and `prov:wasDerivedFrom`.  
**Why:** Reification stores numeric scores and provenance in one place, simplifying evaluation and audit.  
**Risk:** RDF reification increases verbosity and query complexity; prefer RDF-star or property graphs if available.

## 5. Personality inference: rule-based baseline + LLM ensemble
**Choice:** Produce interpretable rule-based scores first (lexicon/keyword heuristics) and optionally augment with LLM/regression outputs; store both in the KG.  
**Why:** Rule-based output is transparent and reproducible; LLMs can improve nuance but require strict validation and evidence checks.  
**Risk:** LLMs can hallucinate trait rationales and must be validated against evidence substrings and calibration tests.

## 6. Evaluation & reproducibility
**Choices:** Use Precision/Recall/F1 for relations and MAE + Pearson for trait scores; save all pipeline artifacts, prompts, and raw LLM responses in `llm_session/`. Use `requirements.txt` or Docker to pin environments.  
**Why:** These metrics provide interpretable signal for discrete and continuous tasks and support targeted error analysis. Logging prompts/responses enables audits and reproducibility.  
**Risk:** Small gold sets can produce noisy metrics; track confidence intervals and increase gold data iteratively.

## 7. Practical reproducibility steps
- Pin Python and model versions in `requirements.txt` and document `python -m spacy download en_core_web_sm`.  
- Save prompt templates and raw LLM outputs.  
- Provide small example data and a `Makefile` or script that runs the full demo locally.

