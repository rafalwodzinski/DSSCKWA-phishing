"""Testy finalnej ewaluacji na tymczasowych artefaktach."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from src.extract_features import build_feature_pipeline
from src.final_eval import run_final_evaluation
from src.train import build_baseline_pipeline


def test_final_evaluation_generates_metrics_figures_and_benchmarks(tmp_path: Path):
    urls = [
        "secure-bank-login.example/login",
        "account-verify.example/update",
        "news.example/article",
        "shop.example/products",
    ]
    labels = [1, 1, 0, 0]
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"URL": urls, "Label": labels}).to_csv(train_path, index=False)
    pd.DataFrame({"URL": urls, "Label": labels}).to_csv(test_path, index=False)

    baseline = build_baseline_pipeline(max_features=100)
    baseline.fit(urls, labels)
    baseline_path = tmp_path / "baseline.joblib"
    joblib.dump(baseline, baseline_path)

    hybrid = Pipeline([
        ("features", build_feature_pipeline(max_features=100)),
        ("classifier", ComplementNB()),
    ])
    hybrid.fit(urls, labels)
    hybrid_path = tmp_path / "best_hybrid.joblib"
    joblib.dump(hybrid, hybrid_path)

    threshold_path = tmp_path / "threshold.json"
    threshold_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    results_path = tmp_path / "results" / "final_test_metrics.json"
    figures_dir = tmp_path / "figures"

    result = run_final_evaluation(
        test_path=test_path,
        train_path=train_path,
        baseline_model_path=baseline_path,
        best_hybrid_model_path=hybrid_path,
        threshold_path=threshold_path,
        results_path=results_path,
        figures_dir=figures_dir,
        benchmark_samples=4,
    )

    assert set(result["models"]) == {"Baseline", "Best Hybrid"}
    assert result["thresholds"]["Best Hybrid"] == 0.5
    assert result["benchmarks"]["inference"]["Best Hybrid"]["sample_count"] == 4
    assert result["benchmarks"]["training"]["sample_count"] == 4
    assert len(result["feature_importance_top_15"]) <= 15
    assert json.loads(results_path.read_text(encoding="utf-8"))["evaluation"] == "final_test"
    assert {
        "confusion_matrices.png",
        "roc_curves.png",
        "feature_importance.png",
    }.issubset({path.name for path in figures_dir.iterdir()})


def test_final_evaluation_rejects_too_few_benchmark_urls(tmp_path: Path):
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"URL": ["a.example", "b.example"], "Label": [0, 1]}).to_csv(
        test_path, index=False
    )

    with pytest.raises(ValueError, match="Benchmark wymaga 1000 URL-i"):
        run_final_evaluation(test_path=test_path, benchmark_samples=1000)
