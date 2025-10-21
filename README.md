### 🧠 Knowledge Graph + Personality Extraction Pipeline

A compact, reproducible pipeline that extracts a **document-level Knowledge Graph (KG)** — entities, relations, and inferred **Big-5 personality traits** — from short texts.
Designed for clarity, reproducibility, and to demonstrate LLM-assisted workflows (prompt chains, provenance, and evaluation).

---

### ▶️ Quick links

* Notebook: `notebooks/03_pipeline_demo.ipynb`
* Synthetic data: `data/synthetic_v1.json`
* Prompts: `prompts/`
* Saved LLM outputs: `llm_session/raw_responses.json`
* Outputs: `notebooks_output/`
* Core code: `src/`
* Tests: `tests/test_pipeline.py`

---

### ⚡ Highlights

* Modular pipeline: **chunk → NER → coref → RE → normalize → personality → KG**.
* Works fully offline with synthetic data; optional LLM calls are supported and logged.
* Outputs: normalized triples, canonical entity nodes, per-person Big-5 scores with evidence quotes, NetworkX KG exports and visualizations.
* Focus: clear design, repeatable prompts, rigorous evaluation (P/R/F1 for triples; MAE/Pearson for traits).

---

### 🚀 Quick start (local)

1. Create environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. (Optional) Enable LLM calls:

```bash
export OPENAI_API_KEY="sk-..."   # macOS / Linux
# setx OPENAI_API_KEY "sk-..."   # Windows (PowerShell)
```

3. Run the demo notebook:

```bash
jupyter notebook notebooks/03_pipeline_demo.ipynb
```

---

### 🧩 What the notebook does

* Loads `data/synthetic_v1.json` (gold-labeled synthetic docs).
* Runs **spaCy NER** and a compact **SVO extractor** (OpenIE-lite).
* (Optional) Calls LLM for relation extraction / personality inference (prompt chain saved).
* Normalizes relation verbs to a controlled predicate set, canonicalizes mentions.
* Builds a KG with `networkx`, adds `has_trait` edges with numeric scores.
* Runs evaluation: Precision/Recall/F1 for triples; MAE & Pearson for Big-5 scores.
* Saves artifacts to `notebooks_output/` and records all LLM prompts/responses in `llm_session/`.

---

### 🧪 Run tests

```bash
pytest -q
```

If a test fails because spaCy model is missing:

```bash
python -m spacy download en_core_web_sm
```

---

### 📂 Repository structure (essentials)

```
.
├─ data/                     # synthetic dataset (synthetic_v1.json)
├─ notebooks/                # demo notebooks (03_pipeline_demo.ipynb)
├─ prompts/                  # LLM prompt templates (chunk, NER, RE, normalize, personality)
├─ src/                      # extractors, normalize, llm_client, kg_builder, eval
├─ notebooks_output/         # exported graphs, jsons, evaluation results
├─ llm_session/              # saved prompt inputs + raw LLM outputs (traceability)
├─ tests/                    # pytest unit tests
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

### ✅ Reproducibility & grading checklist

Include these files in your submission so graders can reproduce and inspect your process:

* `data/synthetic_v1.json` (gold labels and personality)
* `notebooks/03_pipeline_demo.ipynb` (run notebook)
* `prompts/` (exact prompts used, few-shot examples)
* `llm_session/raw_responses.json` (raw model outputs + metadata)
* `src/` modules (pipeline code)
* `notebooks_output/` (example KG PNGs, triples JSON)
* `report/design_choices.md` (1–1.5 page design justification; mark “ChatGPT-assisted”)
* `README.md` (this file)



### 📈 Evaluation details (what we measure)

* **Triples**: exact-match on normalized `(subject, predicate, object)` → Precision / Recall / F1.

  * Normalization: lowercasing, punctuation stripping, predicate mapping table.
* **Personality**: per-trait Mean Absolute Error (MAE) & Pearson correlation against gold Big-5 scores.
* **Qualitative**: sample false positives/negatives and evidence-based error analysis.

---

### 🧠 Notes on LLM workflow & ethics

* The pipeline demonstrates a **chain-of-prompts** approach (micro-tasks), not one monolithic prompt: chunk → NER → coref → RE → normalize → personality.
* All LLM outputs are validated: JSON schema checks + evidence substring presence. Retry logic is applied for malformed outputs.
* Ethical note: personality inference from text is noisy and sensitive. This project uses **synthetic data** and records provenance; do **not** apply to real people without consent.


# Knowledge_graph_demo
