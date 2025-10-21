from typing import List, Dict, Tuple
import spacy
from spacy.tokens import Doc, Span, Token

_NLP = None


def get_nlp(model: str = "en_core_web_sm") -> spacy.language.Language:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load(model)
        except OSError as e:
            raise RuntimeError(f"spaCy model '{model}' not found. Install it (python -m spacy download en_core_web_sm)") from e
    return _NLP


def spacy_ner(text: str, model: str = "en_core_web_sm") -> List[Dict]:
    """Return list of entities: {text, start_char, end_char, label}.
    Keeps surface text as in the document.
    """
    nlp = get_nlp(model)
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({
            "text": ent.text,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "label": ent.label_
        })
    return ents


def _find_subject(tok: Token) -> Token:
    # walk left for nsubj relations
    for child in tok.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child
    # try ancestors
    for anc in tok.ancestors:
        for child in anc.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return child
    return None


def _find_object(tok: Token) -> Token:
    # direct object or object of preposition
    for child in tok.children:
        if child.dep_ in ("dobj", "obj", "pobj"):
            return child
    # look deeper
    for child in tok.children:
        for gc in child.children:
            if gc.dep_ in ("dobj", "obj", "pobj"):
                return gc
    return None


def extract_svo_from_sent(sent: Span) -> List[Dict]:
    """Given a spaCy sentence span, return list of SVO dicts: {subj, verb, obj, subj_span, obj_span}

    This is a heuristic extractor suitable for short sentences in our synthetic data.
    """
    svos = []
    # find verbs
    for tok in sent:
        if tok.pos_ == "VERB" or tok.dep_ == "ROOT":
            subj = _find_subject(tok)
            obj = _find_object(tok)
            if subj is not None and obj is not None:
                svos.append({
                    "subj": subj.text,
                    "verb": tok.lemma_,
                    "obj": obj.text,
                    "subj_span": (subj.idx, subj.idx + len(subj.text)),
                    "obj_span": (obj.idx, obj.idx + len(obj.text)),
                    "sent_text": sent.text
                })
    return svos


def extract_svo_from_doc(text: str, model: str = "en_core_web_sm") -> Dict:
    """Runs spaCy pipeline and returns sentences and SVO triples per sentence.

    Returns: {"sentences": [str,...], "svos": [ {sentence_index, subj, verb, obj, subj_span, obj_span} ] }
    """
    nlp = get_nlp(model)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    svos = []
    for i, sent in enumerate(doc.sents):
        for s in extract_svo_from_sent(sent):
            s_out = s.copy()
            s_out["sentence_index"] = i
            svos.append(s_out)
    return {"sentences": sentences, "svos": svos}