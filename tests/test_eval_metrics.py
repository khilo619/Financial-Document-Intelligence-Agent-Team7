"""
tests/test_eval_metrics.py: Exhaustive test suite for Project LEDGER evaluation metrics.
Verifies Exact Match, Token F1, Numerical Accuracy with epsilon=0.01 tolerance,
scale conversions, and Retrieval Recall@K.
"""

import pytest

from services.eval_service.src.metrics import (
    convert_to_base_unit,
    exact_match,
    normalize_text,
    numerical_accuracy,
    parse_numerical_value,
    retrieval_recall_at_k,
    token_f1,
)

# ==============================================================================
# 1. Text Normalization & Exact Match Tests
# ==============================================================================


def test_normalize_text():
    assert normalize_text("The Operating Income, in 2019!") == "operating income in 2019"
    assert normalize_text("  a net profit of  $500  ") == "net profit of 500"
    assert normalize_text(None) == ""


def test_exact_match_strings():
    # Exact string match after normalization
    assert exact_match("CTS Corporation", "cts corporation") == 1.0
    assert exact_match("The Operating Income", "operating income") == 1.0
    assert exact_match("Different Company", "Other Company") == 0.0
    assert exact_match(None, "Valid") == 0.0


def test_exact_match_numerics():
    # Numbers should match even if formatted differently
    assert exact_match("304,811", 304811) == 1.0
    assert exact_match("$14.50", 14.5) == 1.0
    assert exact_match("100", "200") == 0.0


# ==============================================================================
# 2. Token F1 Score Tests
# ==============================================================================


def test_token_f1_identical():
    assert token_f1("operating cash flow", "operating cash flow") == 1.0
    assert token_f1("", "") == 1.0


def test_token_f1_disjoint():
    assert token_f1("operating income", "net profit") == 0.0


def test_token_f1_partial_overlap():
    # Overlap: 2 common words ("operating", "income")
    # Pred: "operating", "income", "margin" (3 words) -> Precision = 2/3
    # True: "operating", "income" (2 words) -> Recall = 2/2 = 1.0
    # F1 = 2 * (2/3 * 1.0) / (2/3 + 1.0) = (4/3) / (5/3) = 0.8
    score = token_f1("operating income margin", "operating income")
    assert pytest.approx(score, rel=1e-3) == 0.8


# ==============================================================================
# 3. Numerical Parsing & Scale Conversion Tests
# ==============================================================================


def test_parse_numerical_value():
    assert parse_numerical_value(304811) == 304811.0
    assert parse_numerical_value("304,811") == 304811.0
    assert parse_numerical_value("$14.3M") == 14_300_000.0
    assert parse_numerical_value("€500K") == 500_000.0
    assert parse_numerical_value("(1,234.56)") == -1234.56
    assert parse_numerical_value("12.5%") == 12.5
    assert parse_numerical_value("not_a_number") is None


def test_convert_to_base_unit():
    assert convert_to_base_unit(9447.0, "thousand") == 9_447_000.0
    assert convert_to_base_unit(14.3, "million") == 14_300_000.0
    assert convert_to_base_unit(2.5, "billion") == 2_500_000_000.0
    assert convert_to_base_unit(50.0, None) == 50.0


# ==============================================================================
# 4. Numerical Accuracy Relative Tolerance (epsilon = 0.01) Tests
# ==============================================================================


def test_numerical_accuracy_exact():
    assert numerical_accuracy(304811, 304811) == 1.0
    assert numerical_accuracy("304,811", 304811) == 1.0


def test_numerical_accuracy_within_epsilon():
    # Tolerance is 1% (epsilon = 0.01)
    # Target = 1000.0. Boundary: 990.0 to 1010.0
    assert numerical_accuracy(1005.0, 1000.0) == 1.0  # 0.5% error -> pass
    assert numerical_accuracy(995.0, 1000.0) == 1.0  # 0.5% error -> pass
    assert numerical_accuracy(1009.9, 1000.0) == 1.0  # 0.99% error -> pass


def test_numerical_accuracy_outside_epsilon():
    # Target = 1000.0. 1.5% error -> fail
    assert numerical_accuracy(1015.0, 1000.0) == 0.0
    assert numerical_accuracy(985.0, 1000.0) == 0.0


def test_numerical_accuracy_zero_division_protection():
    # Target = 0.0. Formula uses max(|v*|, 1e-5). Epsilon = 0.01 -> boundary is 1e-7
    assert numerical_accuracy(0.0, 0.0) == 1.0
    assert numerical_accuracy(1e-7, 0.0) == 1.0
    assert numerical_accuracy(5.0, 0.0) == 0.0


def test_numerical_accuracy_scale_harmonization():
    # Model returns 9,447,000 in raw dollars; Ground truth is 9447 in thousands
    assert (
        numerical_accuracy(
            pred_val=9_447_000,
            true_val=9447,
            pred_scale=None,
            true_scale="thousand",
        )
        == 1.0
    )

    # Model returns 14.3 with scale "million"; Ground truth is 14,300 with scale "thousand"
    assert (
        numerical_accuracy(
            pred_val=14.3,
            true_val=14300,
            pred_scale="million",
            true_scale="thousand",
        )
        == 1.0
    )


# ==============================================================================
# 5. Retrieval Recall@K Tests
# ==============================================================================


def test_retrieval_recall_at_k():
    gold = [
        {"source_document": "cts-corporation_2019.pdf", "source_page": 1},
        {"source_document": "jabil-circuit-inc_2019.pdf", "source_page": 1},
    ]

    # Perfect retrieval of both docs
    retrieved_full = [
        {"document_id": "cts-corporation_2019.pdf", "page": 1},
        {"document_id": "jabil-circuit-inc_2019.pdf", "page": 1},
        {"document_id": "amcon-distributing-company_2019.pdf", "page": 3},
    ]
    assert retrieval_recall_at_k(retrieved_full, gold) == 1.0

    # Partial retrieval: only 1 of 2 gold docs found
    retrieved_partial = [
        {"document_id": "cts-corporation_2019.pdf", "page": 1},
        {"document_id": "distractor_doc.pdf", "page": 1},
    ]
    assert retrieval_recall_at_k(retrieved_partial, gold) == 0.5

    # Complete retrieval miss
    retrieved_miss = [
        {"document_id": "unrelated_doc_2018.pdf", "page": 5},
    ]
    assert retrieval_recall_at_k(retrieved_miss, gold) == 0.0
