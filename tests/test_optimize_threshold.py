"""Testy jednostkowe modułu strojenia hiperparametrów i progu decyzyjnego (src/optimize_threshold.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.optimize_threshold import (
    DEFAULT_MAX_FPR_BUFFER,
    find_optimal_threshold,
    plot_threshold_tuning,
    tune_hyperparameters_and_threshold,
    verify_dod_s3,
)


def test_find_optimal_threshold_respects_max_fpr_and_maximizes_f1():
    """Weryfikuje, że wyznaczony próg tau* spełnia FPR <= max_fpr i maksymalizuje F1."""
    rng = np.random.default_rng(42)
    # 500 próbek klasy 0 i 200 próbek klasy 1 z dobrze rozdzielonymi rozkładami
    neg_scores = rng.normal(loc=-1.2, scale=0.6, size=500)
    pos_scores = rng.normal(loc=1.3, scale=0.6, size=200)

    y_true = np.concatenate([np.zeros(500, dtype=int), np.ones(200, dtype=int)])
    y_score = np.concatenate([neg_scores, pos_scores])

    tau, metrics, curve_df = find_optimal_threshold(
        y_true=y_true,
        y_score=y_score,
        max_fpr=DEFAULT_MAX_FPR_BUFFER,
        num_grid_points=200,
    )

    assert isinstance(tau, float)
    assert metrics["fpr"] <= DEFAULT_MAX_FPR_BUFFER
    assert metrics["f1"] > 0.85
    assert {"threshold", "f1", "precision", "recall", "fpr", "fnr"}.issubset(curve_df.columns)


def test_plot_threshold_tuning_creates_300dpi_png(tmp_path: Path):
    """Weryfikuje generowanie wykresu threshold_tuning.png w rozdzielczości 300 DPI."""
    curve_df = pd.DataFrame({
        "threshold": np.linspace(-1.0, 1.0, 50),
        "f1": np.linspace(0.80, 0.95, 50),
        "precision": np.linspace(0.82, 0.97, 50),
        "recall": np.linspace(0.95, 0.85, 50),
        "fpr": np.linspace(0.05, 0.005, 50),
        "fnr": np.linspace(0.05, 0.15, 50),
    })
    optimal_tau = 0.12
    optimal_metrics = {
        "f1": 0.945,
        "precision": 0.96,
        "recall": 0.93,
        "fpr": 0.015,
        "fnr": 0.07,
        "roc_auc": 0.994,
    }
    out_plot = tmp_path / "threshold_tuning.png"

    saved_path = plot_threshold_tuning(
        curve_df=curve_df,
        optimal_tau=optimal_tau,
        optimal_metrics=optimal_metrics,
        save_path=out_plot,
    )

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0
    with Image.open(saved_path) as img:
        dpi = img.info.get("dpi")
        if dpi is not None:
            assert round(dpi[0]) == 300


def test_tune_rejects_test_set_path(tmp_path: Path):
    """Weryfikuje DoD 1: blokada przekazania pliku test.csv do strojenia hiperparametrów/progu."""
    fake_train = tmp_path / "train.csv"
    fake_test = tmp_path / "test.csv"
    df = pd.DataFrame({"URL": ["example.com", "phish.org"], "Label": [0, 1]})
    df.to_csv(fake_train, index=False)
    df.to_csv(fake_test, index=False)

    with pytest.raises(ValueError, match="DoD 1: Zbiór testowy"):
        tune_hyperparameters_and_threshold(
            train_path=fake_train,
            val_path=fake_test,
            best_model_path=tmp_path / "best_hybrid.joblib",
            threshold_json_path=tmp_path / "threshold.json",
            plot_path=tmp_path / "plot.png",
            results_dir=tmp_path,
        )


def test_tune_hyperparameters_and_threshold_end_to_end_synthetic(tmp_path: Path):
    """Weryfikuje pełny przepływ S3 na syntetycznym zbiorze danych i sprawdza artefakty S3."""
    # Tworzymy wyraźnie separowalny zbiór syntetyczny dla szybkiego testu end-to-end
    benign_urls = [f"trusted-portal-{i}.edu/courses/module-{i}/overview.html" for i in range(120)]
    phish_urls = [
        f"http://192.168.1.{i % 250}/secure-login-paypal-verify-account-{i}.com/auth?user=admin&token=%2f%20{i}"
        for i in range(120)
    ]

    train_df = pd.DataFrame({
        "URL": benign_urls[:80] + phish_urls[:80],
        "Label": [0] * 80 + [1] * 80,
    })
    val_df = pd.DataFrame({
        "URL": benign_urls[80:] + phish_urls[80:],
        "Label": [0] * 40 + [1] * 40,
    })

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    best_model_path = tmp_path / "models" / "best_hybrid.joblib"
    threshold_json_path = tmp_path / "models" / "threshold.json"
    plot_path = tmp_path / "reports" / "figures" / "threshold_tuning.png"
    results_dir = tmp_path / "reports" / "results"

    pipeline, report, dod = tune_hyperparameters_and_threshold(
        train_path=train_csv,
        val_path=val_csv,
        best_model_path=best_model_path,
        threshold_json_path=threshold_json_path,
        plot_path=plot_path,
        results_dir=results_dir,
        max_features_grid=(200,),
        cnb_alpha_grid=(0.5,),
        svm_c_grid=(1.0,),
    )

    assert best_model_path.exists()
    assert threshold_json_path.exists()
    assert plot_path.exists()
    assert all(dod.values())

    with open(threshold_json_path, encoding="utf-8") as f:
        thresh_meta = json.load(f)

    assert "threshold" in thresh_meta
    assert thresh_meta["test_set_used"] is False
    assert thresh_meta["val_metrics_at_optimal_threshold"]["f1"] >= 0.92
    assert thresh_meta["val_metrics_at_optimal_threshold"]["fpr"] <= 0.02
    assert thresh_meta["val_metrics_at_optimal_threshold"]["roc_auc"] >= 0.95

