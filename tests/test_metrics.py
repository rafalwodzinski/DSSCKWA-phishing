"""Testy jednostkowe modułu ewaluacji i metryk (src/evaluate.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_metrics,
)


def test_fpr_formula_exact_values():
    """Weryfikacja poprawności wzoru na False Positive Rate: FPR = FP / (FP + TN).

    Scenariusz z predefiniowaną macierzą:
    Rzeczywiste negatywne (klasa 0): 4 próbki.
    Predykcja: 1 jako FP, 3 jako TN.
    Rzeczywiste pozytywne (klasa 1): 2 próbki (obie TP).
    Oczekiwane:
    TN = 3, FP = 1 -> FP + TN = 4 -> FPR = 1/4 = 0.25.
    TP = 2, FN = 0 -> FNR = 0.0.
    """
    y_true = [0, 0, 0, 0, 1, 1]
    # Dla progu 0.5:
    # 0 -> 0.1 (TN)
    # 0 -> 0.2 (TN)
    # 0 -> 0.3 (TN)
    # 0 -> 0.8 (FP)
    # 1 -> 0.9 (TP)
    # 1 -> 0.7 (TP)
    y_score = [0.1, 0.2, 0.3, 0.8, 0.9, 0.7]

    metrics = compute_metrics(y_true, y_score, threshold=0.5)

    assert metrics["fpr"] == pytest.approx(0.25, abs=1e-5)
    assert metrics["fnr"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["recall"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-5)


def test_fnr_formula_exact_values():
    """Weryfikacja poprawności wzoru na False Negative Rate: FNR = FN / (TP + FN).

    Scenariusz:
    Rzeczywiste pozytywne (klasa 1): 4 próbki.
    Predykcja: 1 jako FN (błędnie zaklasyfikowana jako 0), 3 jako TP.
    Oczekiwane: FNR = 1 / 4 = 0.25.
    """
    y_true = [1, 1, 1, 1, 0, 0]
    # Próg 0.5:
    # 1 -> 0.2 (FN)
    # 1 -> 0.7 (TP)
    # 1 -> 0.8 (TP)
    # 1 -> 0.9 (TP)
    # 0 -> 0.1 (TN)
    # 0 -> 0.2 (TN)
    y_score = [0.2, 0.7, 0.8, 0.9, 0.1, 0.2]

    metrics = compute_metrics(y_true, y_score, threshold=0.5)

    assert metrics["fnr"] == pytest.approx(0.25, abs=1e-5)
    assert metrics["recall"] == pytest.approx(0.75, abs=1e-5)
    assert metrics["fpr"] == pytest.approx(0.0, abs=1e-5)


def test_predict_proba_2d_input():
    """Weryfikacja obsługi wyjścia 2D z `model.predict_proba()` (kolumny [P(0), P(1)])."""
    y_true = [0, 1, 0, 1]
    # 2D array z dwiema kolumnami
    y_proba_2d = np.array([
        [0.85, 0.15],  # Pred: 0 (TN)
        [0.20, 0.80],  # Pred: 1 (TP)
        [0.40, 0.60],  # Pred: 1 (FP)
        [0.10, 0.90],  # Pred: 1 (TP)
    ])

    metrics = compute_metrics(y_true, y_proba_2d, threshold=0.5)

    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "fpr" in metrics
    assert "fnr" in metrics
    assert "roc_auc" in metrics
    # TN=1, FP=1 -> FPR = 0.5
    assert metrics["fpr"] == pytest.approx(0.5, abs=1e-5)
    # TP=2, FN=0 -> FNR = 0.0
    assert metrics["fnr"] == pytest.approx(0.0, abs=1e-5)


def test_decision_function_unbounded_scores():
    """Weryfikacja obsługi wartości z `decision_function()` (np. SVM, wartości ujemne i dodatnie)."""
    y_true = [0, 0, 1, 1]
    # Wartości ciągłe odległości od hiperpłaszczyzny
    y_decision = np.array([-2.4, -0.8, 0.5, 3.1])

    # Standardowy próg dla decision_function to 0.0
    metrics = compute_metrics(y_true, y_decision, threshold=0.0)

    # Wszystkie trafione perfekcyjnie: TN=2, FP=0, FN=0, TP=2
    assert metrics["fpr"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["fnr"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["f1"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["roc_auc"] == pytest.approx(1.0, abs=1e-5)


def test_zero_division_safety():
    """Weryfikacja braku ZeroDivisionError w skrajnych przypadkach (np. same 0 lub same 1)."""
    # Same zera w zbiorze
    y_true_zeros = [0, 0, 0]
    y_score_zeros = [0.1, 0.2, 0.3]
    metrics_zeros = compute_metrics(y_true_zeros, y_score_zeros, threshold=0.5)
    assert metrics_zeros["fpr"] == 0.0
    assert metrics_zeros["fnr"] == 0.0

    # Same jedynki w zbiorze
    y_true_ones = [1, 1, 1]
    y_score_ones = [0.8, 0.9, 0.7]
    metrics_ones = compute_metrics(y_true_ones, y_score_ones, threshold=0.5)
    assert metrics_ones["fpr"] == 0.0
    assert metrics_ones["fnr"] == 0.0


def test_plot_confusion_matrix_creation(tmp_path: Path):
    """Weryfikacja tworzenia pliku macierzy pomyłek w rozdzielczości 300 DPI."""
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 0, 1]
    save_file = tmp_path / "confusion_matrix_test.png"

    output_path = plot_confusion_matrix(
        y_true, y_pred, model_name="test_model", save_path=save_file
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Sprawdzenie DPI przez PIL
    with Image.open(output_path) as img:
        dpi = img.info.get("dpi")
        if dpi is not None:
            # 300 DPI z tolerancją zaokrąglenia
            assert round(dpi[0]) == 300


def test_plot_roc_curve_creation(tmp_path: Path):
    """Weryfikacja tworzenia wykresu krzywej ROC w 300 DPI z zaznaczonym progiem."""
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    save_file = tmp_path / "roc_curve_test.png"

    output_path = plot_roc_curve(
        y_true, y_score, model_name="test_model", threshold=0.5, save_path=save_file
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    with Image.open(output_path) as img:
        dpi = img.info.get("dpi")
        if dpi is not None:
            assert round(dpi[0]) == 300


def test_save_metrics_json_and_csv(tmp_path: Path):
    """Weryfikacja zapisu słownika metryk do pliku .json oraz zbiorczego .csv."""
    sample_metrics = {
        "f1": 0.88,
        "precision": 0.90,
        "recall": 0.86,
        "fpr": 0.04,
        "fnr": 0.14,
        "roc_auc": 0.95,
    }

    json_file = save_metrics(
        sample_metrics, name="baseline_nb", output_dir=tmp_path, export_csv=True
    )

    assert json_file.exists()
    with open(json_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["f1"] == 0.88
    assert loaded["fpr"] == 0.04

    # Weryfikacja CSV
    csv_file = tmp_path / "summary_metrics.csv"
    assert csv_file.exists()
    csv_content = csv_file.read_text(encoding="utf-8")
    assert "baseline_nb" in csv_content
    assert "0.88" in csv_content
