"""Validate notebook 04 hybrid classifier matches expected accuracy."""
import os
import re
import certifi
import httpx
import pandas as pd
from pathlib import Path
from huggingface_hub import set_client_factory
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
feedback = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "synthetic_feedback.csv")

ZERO_SHOT_LABELS = {
    "accessibility_issue": (
        "barriers for learners with disabilities such as screen readers, captions, "
        "transcripts, contrast, or assistive technology"
    ),
    "content_quality": (
        "unclear or poor course materials, explanations, or teaching quality"
    ),
    "pacing": (
        "the course pace, workload, or deadlines feel too fast, rushed, or overwhelming"
    ),
    "technical_issue": (
        "platform or system failures such as login problems, uploads, crashes, or website outages"
    ),
    "positive": (
        "the student is praising or expressing satisfaction with the course"
    ),
}
PHRASE_TO_CATEGORY = {phrase: cat for cat, phrase in ZERO_SHOT_LABELS.items()}
CANDIDATE_PHRASES = list(ZERO_SHOT_LABELS.values())
HYPOTHESIS_TEMPLATE = "This student course feedback is about {}."
INPUT_PREFIX = "Online course student feedback: "
CONFIDENCE_THRESHOLD = 0.45
ACCESSIBILITY_GATE = re.compile(
    r"screen reader|caption|captions|transcript|subtitle|assistive|contrast|"
    r"accessib|couldn't access|could not access|tagged properly",
    re.I,
)
KEYWORD_RULES = [
    ("accessibility_issue", ACCESSIBILITY_GATE),
    ("technical_issue", re.compile(
        r"error|bug|broken|crash|crashed|login|upload|platform|quiz closed|website|"
        r"outage|maintenance|logged me out|marked late", re.I)),
    ("pacing", re.compile(
        r"too fast|moving too fast|rushed|deadline|deadlines|too many readings|"
        r"too much|overwhelming|chapters in one week|between tmas|between the midterm", re.I)),
    ("positive", re.compile(
        r"great explanations|really helped|really enjoyed|excellent tutor|"
        r"best online course|very engaging|well structured|well organized", re.I)),
    ("content_quality", re.compile(
        r"unclear|confusing|outdated|poor audio|hard to follow|a lot to read", re.I)),
]

CA_BUNDLE = certifi.where()
for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ[var] = CA_BUNDLE
set_client_factory(
    lambda: httpx.Client(verify=CA_BUNDLE, follow_redirects=True, timeout=60.0)
)

from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=-1,
)


def _classify_keyword(text):
    for category, pattern in KEYWORD_RULES:
        if pattern.search(str(text)):
            return category, 0.75, "keyword_fallback"
    return "content_quality", 0.5, "keyword_fallback"


def classify_feedback(text):
    if ACCESSIBILITY_GATE.search(str(text)):
        return "accessibility_issue", 0.85, "hybrid_accessibility_gate"
    out = classifier(
        f"{INPUT_PREFIX}{text}",
        CANDIDATE_PHRASES,
        multi_label=False,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
    )
    category = PHRASE_TO_CATEGORY[out["labels"][0]]
    score = round(out["scores"][0], 4)
    if score < CONFIDENCE_THRESHOLD:
        kw_category, kw_score, _ = _classify_keyword(text)
        return kw_category, kw_score, "hybrid_low_confidence"
    return category, score, "zero_shot"


results = [classify_feedback(t) for t in feedback["feedback_text"]]
preds = [r[0] for r in results]
y_true = feedback["true_label"]
print(f"Accuracy: {accuracy_score(y_true, preds):.1%}")
print(classification_report(y_true, preds, zero_division=0))
from collections import Counter
print("Methods:", Counter(r[2] for r in results))
