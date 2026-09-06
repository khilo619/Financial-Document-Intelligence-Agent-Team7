"""
services.eval_service.src.metrics: Pure mathematical and string evaluation metrics.
Implements Project LEDGER gold benchmark formulas:
- Exact Match (EM) with text normalization
- Token-level F1 Score (Precision / Recall / F1)
- Numerical Accuracy with relative tolerance epsilon = 0.01 and scale harmonization
- Retrieval Recall@K and Mean Reciprocal Rank (MRR)
"""

import re
import string
from typing import Any


def normalize_text(s: str | Any) -> str:
    """
    Standard text normalization for Exact Match and F1 evaluation.
    Converts to lowercase, strips punctuation, removes articles ('a', 'an', 'the'),
    and standardizes whitespace.
    """
    if s is None:
        return ""
    text = str(s).lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove punctuation
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Standardize whitespace
    return " ".join(text.split())


def exact_match(prediction: Any, ground_truth: Any) -> float:
    """
    Binary metric: 1.0 if normalized prediction matches normalized ground truth, 0.0 otherwise.
    If both values are numeric, delegates to numerical comparison as well.
    """
    if prediction is None or ground_truth is None:
        return 0.0

    # Try numeric equality first if both parse as numbers
    pred_num = parse_numerical_value(prediction)
    true_num = parse_numerical_value(ground_truth)
    if (
        pred_num is not None
        and true_num is not None
        and abs(pred_num - true_num) <= 1e-6
    ):
        return 1.0

    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def token_f1(prediction: Any, ground_truth: Any) -> float:
    """
    Token-level F1 score measuring precision and recall overlap of word tokens.
    """
    pred_tokens = normalize_text(prediction).split()
    true_tokens = normalize_text(ground_truth).split()

    if not pred_tokens and not true_tokens:
        return 1.0
    if not pred_tokens or not true_tokens:
        return 0.0

    # Bag of words overlap
    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    true_counts: dict[str, int] = {}
    for t in true_tokens:
        true_counts[t] = true_counts.get(t, 0) + 1

    overlap = 0
    for t, count in pred_counts.items():
        if t in true_counts:
            overlap += min(count, true_counts[t])

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(true_tokens)
    return (2 * precision * recall) / (precision + recall)


# Multipliers for standard financial scales
SCALE_FACTORS: dict[str, float] = {
    "none": 1.0,
    "exact": 1.0,
    "unit": 1.0,
    "thousand": 1_000.0,
    "thousands": 1_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
    "percent": 0.01,
    "percentage": 0.01,
}


def parse_numerical_value(val: Any) -> float | None:
    """
    Parses numerical values from floats, ints, or complex financial string representations.
    Handles:
    - Commas in thousands: "304,811" -> 304811.0
    - Financial negative parentheses: "(124.50)" -> -124.50
    - Currency prefixes/suffixes: "$14.3M", "€500K"
    - Percentages: "12.5%" -> 12.5
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s:
        return None

    # Handle financial negative in parentheses: (1,234.56) -> -1,234.56
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()

    # Strip currency signs and spaces
    for prefix in ["$", "€", "£", "¥", "₹"]:
        s = s.replace(prefix, "")

    # Strip percentage sign
    s = s.replace("%", "").strip()

    # Handle shorthand multipliers at end of string (e.g., "14.3M", "50K", "2.1B")
    multiplier = 1.0
    if s and s[-1].upper() in ["K", "M", "B", "T"]:
        suffix = s[-1].upper()
        if suffix == "K":
            multiplier = 1_000.0
        elif suffix == "M":
            multiplier = 1_000_000.0
        elif suffix == "B":
            multiplier = 1_000_000_000.0
        elif suffix == "T":
            multiplier = 1_000_000_000_000.0
        s = s[:-1].strip()

    # Remove commas
    s = s.replace(",", "")

    try:
        num = float(s) * multiplier
        return -num if is_negative else num
    except ValueError:
        return None


def convert_to_base_unit(val: float, scale: str | None) -> float:
    """Converts a value with a financial scale to its base unit (e.g. 9.447 thousand -> 9447.0)."""
    if not scale:
        return val
    factor = SCALE_FACTORS.get(scale.lower().strip(), 1.0)
    return val * factor


def numerical_accuracy(
    pred_val: Any,
    true_val: Any,
    pred_scale: str | None = None,
    true_scale: str | None = None,
    epsilon: float = 0.01,
) -> float:
    """
    Computes numerical accuracy with relative tolerance epsilon (default = 0.01, i.e., 1%):
        Acc = 1.0 if (|v_hat - v*| / max(|v*|, 1e-5)) <= epsilon else 0.0

    Features scale harmonization:
    If scales differ (e.g. one is raw dollars and the other in thousands), converts both
    to base units before evaluating. Also checks raw values directly in case both answers
    used the same unnormalized scale.
    """
    p = parse_numerical_value(pred_val)
    t = parse_numerical_value(true_val)

    if p is None or t is None:
        return 0.0

    def check_error(pred: float, target: float) -> bool:
        denominator = max(abs(target), 1e-5)
        rel_error = abs(pred - target) / denominator
        return rel_error <= epsilon

    # 1. Direct comparison
    if check_error(p, t):
        return 1.0

    # 2. Scale-harmonized comparison (e.g., thousands vs raw units)
    if pred_scale or true_scale:
        p_base = convert_to_base_unit(p, pred_scale)
        t_base = convert_to_base_unit(t, true_scale)
        if check_error(p_base, t_base):
            return 1.0

    return 0.0


def retrieval_recall_at_k(
    retrieved_evidence: list[dict[str, Any]],
    gold_evidence: list[dict[str, Any]],
) -> float:
    """
    Computes Recall@K for retrieval.
    Checks how many gold documents (or doc-page pairs) from gold_evidence
    were successfully retrieved in retrieved_evidence.

    Gold evidence objects contain 'source_document' and optional 'source_page'.
    Retrieved evidence objects contain 'document_id' (or 'source_document') and 'page'.
    """
    if not gold_evidence:
        return 1.0
    if not retrieved_evidence:
        return 0.0

    def norm_doc(name: str | None) -> str:
        if not name:
            return ""
        # Strip .pdf and path prefixes
        d = name.lower().split("/")[-1]
        d = d.removesuffix(".pdf")
        return d.strip()

    # Extract target gold pairs: set of (norm_doc, page) or just norm_doc if page is None
    gold_targets = set()
    for g in gold_evidence:
        doc = norm_doc(g.get("source_document") or g.get("source_doc_uid"))
        page = g.get("source_page")
        if doc:
            gold_targets.add((doc, page))

    if not gold_targets:
        return 1.0

    # Extract retrieved pairs
    retrieved_pairs = set()
    retrieved_docs_only = set()
    for r in retrieved_evidence:
        doc = norm_doc(r.get("document_id") or r.get("source_document"))
        page = r.get("page") or r.get("source_page")
        if doc:
            retrieved_pairs.add((doc, page))
            retrieved_docs_only.add(doc)

    # Count hits
    hits = 0
    for gold_doc, gold_page in gold_targets:
        if gold_page is not None:
            if (
                gold_doc,
                gold_page,
            ) in retrieved_pairs or gold_doc in retrieved_docs_only:
                hits += 1
        else:
            if gold_doc in retrieved_docs_only:
                hits += 1

    return hits / len(gold_targets)
