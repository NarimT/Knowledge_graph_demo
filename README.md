# 🧠 Knowledge Graph + Personality Extraction Pipeline

A compact, reproducible pipeline that extracts a **document-level Knowledge Graph (KG)** — entities, relations, and inferred **Big-5 personality traits** — from short texts.  
Designed for clarity, reproducibility, and to demonstrate LLM-assisted workflows (prompt chains, provenance, and evaluation).

---

### ▶️ Quick links

* **Demo notebook:** `notebooks/03_pipeline_demo.ipynb`
* **Synthetic data:** `data/synthetic_v1.json`
* **Prompts:** `prompts/`
* **Saved LLM outputs:** `llm_session/raw_responses.json`
* **Outputs:** `notebooks_output/`
* **Core code:** `src/`
* **Tests:** `test_pipeline.py`
* **Shared session walkthrough:** [View explanation on ChatGPT ↗️](https://chatgpt.com/share/68f6e1e9-29e8-8008-aacf-79c786f7106a)

---

### ⚡ Highlights

* Modular pipeline: **chunk → NER → coref → RE → normalize → personality → KG**  
* Works fully offline with synthetic data; optional LLM calls are supported and logged  
* Outputs include: normalized triples, canonical entity nodes, per-person Big-5 scores (with evidence quotes), and NetworkX KG visualizations  
* Focus: clean design, repeatable prompts, and rigorous evaluation (Precision/Recall/F1 for triples; MAE/Pearson for personality)

---

### 🚀 Quick start (local)

1. **Create environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
(Optional) Enable LLM calls

bash
Copy code
export OPENAI_API_KEY="sk-..."   # macOS / Linux
# setx OPENAI_API_KEY "sk-..."   # Windows (PowerShell)
Run the demo notebook

bash
Copy code
jupyter notebook notebooks/03_pipeline_demo.ipynb
🧩 What the notebook does
Loads data/synthetic_v1.json (gold-labeled synthetic docs)

Runs spaCy NER and a compact SVO extractor (OpenIE-lite)

(Optional) Calls LLM for relation extraction or personality inference (prompt chain logged)

Normalizes verbs and canonicalizes mentions

Builds a KG with networkx, adds has_trait edges with numeric scores

Evaluates: Precision/Recall/F1 for triples; MAE & Pearson for personality traits

Saves all artifacts to notebooks_output/ and raw LLM logs to llm_session/

🧪 Run tests
bash
Copy code
pytest -q
If tests fail because the spaCy model is missing:

bash
Copy code
python -m spacy download en_core_web_sm
📂 Repository structure
graphql
Copy code
.
├─ data/                     # synthetic dataset (synthetic_v1.json)
├─ notebooks/                # demo notebooks (03_pipeline_demo.ipynb)
├─ prompts/                  # LLM prompt templates (chunk, NER, RE, normalize, personality)
├─ src/                      # extractors, normalize, llm_client, kg_builder, eval
├─ notebooks_output/         # exported graphs, jsons, evaluation results
├─ llm_session/              # saved prompt inputs + raw LLM outputs (traceability)
├─ test_pipeline.py          # pytest unit tests
├─ report/design_choices.md  # 1–1.5 page design summary
├─ requirements.txt
└─ README.md
✅ Reproducibility & grading checklist
Please include the following in your submission:

data/synthetic_v1.json — gold labels and Big-5 traits

notebooks/03_pipeline_demo.ipynb — main pipeline notebook

prompts/ — exact prompt templates and examples

llm_session/raw_responses.json — raw model outputs with metadata

src/ — code modules for extraction and KG building

notebooks_output/ — generated KG, evaluation results

report/design_choices.md — design justification, “ChatGPT-assisted” noted

README.md — this file

📈 Evaluation details
Triples: normalized (subject, predicate, object) exact-match
→ reports Precision, Recall, and F1

Personality: per-trait Mean Absolute Error (MAE) & Pearson correlation
→ evaluates Big-5 score prediction quality

Qualitative: error samples and evidence-based analysis (optional)

🧠 Notes on LLM workflow & ethics
The pipeline uses a chain-of-prompts strategy — smaller micro-tasks instead of one big model call:
chunk → NER → coref → RE → normalize → personality

All LLM outputs are validated with schema checks and evidence grounding.

Ethical note: this demo uses synthetic data only. Personality inference on real people without consent should be avoided.

