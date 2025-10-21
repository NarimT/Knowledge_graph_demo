"""
Generic LLM client with prompt templating and robust simulation fallback.

Function: call_llm(prompt_path, input_json, *, validate_evidence=True)
- Reads prompt template from prompts/
- Inserts input text (template must contain "{input_text}")
- Attempts an actual LLM call (OpenAI ChatCompletion) if OPENAI_API_KEY is set and `openai` is installed
- Otherwise simulates a structured JSON response
- Appends prompt+response to llm_session/raw_responses.json
- Validates that any evidence substrings in the returned JSON appear in the input text (optional)
"""
import os
import json
from typing import Dict, Any
from datetime import datetime

LLM_LOG = "llm_session/raw_responses.json"


def _ensure_log():
    os.makedirs(os.path.dirname(LLM_LOG), exist_ok=True)
    if not os.path.exists(LLM_LOG):
        with open(LLM_LOG, "w", encoding="utf8") as f:
            json.dump([], f)


def _append_log(entry: Dict[str, Any]):
    _ensure_log()
    try:
        with open(LLM_LOG, "r", encoding="utf8") as f:
            data = json.load(f)
    except Exception:
        data = []
    data.append(entry)
    with open(LLM_LOG, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _simulate_response(input_text: str) -> Dict[str, Any]:
    """
    Small heuristic simulator returning structured JSON similar to an extraction API.
    Attempts to find a capitalized name and a capitalized organization token.
    """
    text = input_text.replace("\n", " ")
    words = text.split()
    person = None
    org = None
    # naive heuristics: first Titlecase token -> person, some suffixes -> org
    for w in words:
        # strip punctuation
        w_stripped = w.strip(".,;:")
        if w_stripped.istitle() and w_stripped.isalpha() and person is None:
            person = w_stripped
        if w_stripped.endswith("Corp") or w_stripped.endswith("Labs") or w_stripped.endswith("Bank") or w_stripped.endswith("Solutions") or w_stripped.endswith("Telecom") or w_stripped.endswith("Studios"):
            org = w_stripped
    if person is None:
        person = "Alice"
    if org is None:
        org = "Acme Corp"
    simulated = {
        "extracted": [
            {
                "subj": person,
                "pred": "worksAt",
                "obj": org,
                "evidence": [text[: min(len(text), 120)]]
            }
        ],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "simulated": True,
    }
    return simulated


def call_llm(prompt_path: str, input_json: Dict[str, Any], *, validate_evidence: bool = True) -> Dict[str, Any]:
    """
    Call an LLM with a prompt template and an input JSON.

    Args:
        prompt_path: path to prompt template file (should contain the string '{input_text}')
        input_json: dict containing at least a 'text' field
        validate_evidence: if True, ensure any evidence substrings returned are present in the input text

    Returns:
        parsed JSON-like response (dict). If model returned plain text, it's wrapped under {'text': ...}.
    """
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf8") as f:
        tpl = f.read()

    input_text = input_json.get("text") or json.dumps(input_json, ensure_ascii=False)
    prompt = tpl.replace("{input_text}", input_text)

    response_text = None
    used_simulator = False
    parsed = None

    # Try OpenAI ChatCompletion (if available)
    try:
        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")
                model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=800,
                )
                response_text = completion.choices[0].message.content.strip()
            except Exception:
                # fall through to simulator
                response_text = None

        if response_text is None:
            raise RuntimeError("No live LLM response; using simulator")
    except Exception:
        used_simulator = True
        parsed = _simulate_response(input_text)
        response_text = json.dumps(parsed, ensure_ascii=False)

    # Prepare log entry and append
    log_entry = {
        "prompt_path": prompt_path,
        "prompt": prompt,
        "response_text": response_text,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "simulated": used_simulator,
    }
    try:
        _append_log(log_entry)
    except Exception:
        # non-fatal logging error
        pass

    # Parse response_text into JSON-like dict if possible
    if parsed is None:
        try:
            parsed = json.loads(response_text)
        except Exception:
            # not JSON — return as raw text dict
            parsed = {"text": response_text}

    # Validate evidence substrings if requested
    if validate_evidence and isinstance(parsed, dict) and "extracted" in parsed:
        for ex in parsed.get("extracted", []):
            evs = ex.get("evidence", [])
            for ev in evs:
                if ev is None:
                    continue
                # use a simple containment check
                if ev not in input_text:
                    raise ValueError(f"Evidence substring not found in input_text: {ev!r}")

    return parsed


# If run as a script, simple smoke-test:
if __name__ == "__main__":
    sample_prompt = "prompts/demo_prompt.txt"
    sample_text = {"text": "Alice joined Acme Corp as a project manager last month."}
    # create a demo prompt file if missing
    if not os.path.exists(sample_prompt):
        os.makedirs(os.path.dirname(sample_prompt), exist_ok=True)
        with open(sample_prompt, "w", encoding="utf8") as f:
            f.write("Extract relations from the following sentence:\n\n{input_text}\n\nReturn JSON with 'extracted' list.")
    resp = call_llm(sample_prompt, sample_text)
    print("LLM client test response:", json.dumps(resp, indent=2, ensure_ascii=False))
