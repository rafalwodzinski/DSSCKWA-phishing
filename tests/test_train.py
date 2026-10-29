"""Testy jednostkowe modułu trenowania modeli Baseline i hybrydowych (src/train.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.extract_features import LexicalExtractor
from src.train import (
    BASELINE_ALPHA,
    BASELINE_MAX_FEATURES,
    BASELINE_TOKEN_PATTERN,
    CNB_ALPHA,
    SVM_C,
    build_baseline_pipeline,
    build_cnb_pipeline,
    build_svm_pipeline,
    load_model,
    run_stratified_cv,
    save_model,
    train_baseline,
    train_hybrid,
    train_model,
    verify_dod_baseline,
    verify_dod_hybrid,
)


def test_build_baseline_pipeline_structure_and_params():
    """Weryfikuje parametry CountVectorizer i MultinomialNB zgodnie ze specyfikacją Issue."""
    pipeline = build_baseline_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2

    vec = pipeline.named_steps["vectorizer"]
    clf = pipeline.named_steps["classifier"]

    assert isinstance(vec, CountVectorizer)
    assert vec.analyzer == "word"
    assert vec.token_pattern == r"[a-zA-Z0-9]{2,}"
    assert vec.token_pattern == BASELINE_TOKEN_PATTERN
    assert vec.max_features == 20000
    assert vec.max_features == BASELINE_MAX_FEATURES
    assert vec.ngram_range == (1, 1)

    assert isinstance(clf, MultinomialNB)
    assert clf.alpha == 1.0
    assert clf.alpha == BASELINE_ALPHA


def test_baseline_does_not_use_char_ngrams_or_lexical_features():
    """Weryfikuje DoD 1: model Baseline nie korzysta z n-gramów znakowych ani cech leksykalnych."""
    pipeline = build_baseline_pipeline()

    for _, step in pipeline.steps:
        assert not isinstance(step, LexicalExtractor)
        assert not isinstance(step, FeatureUnion)
        assert not isinstance(step, TfidfVectorizer)
        if isinstance(step, CountVectorizer):
            assert step.analyzer == "word"
            assert step.ngram_range == (1, 1)


def test_word_tokenization_behavior():
    """Sprawdza działanie token_pattern=r'[a-zA-Z0-9]{2,}' na adresach URL."""
    pipeline = build_baseline_pipeline()
    vectorizer = pipeline.named_steps["vectorizer"]
    analyzer = vectorizer.build_analyzer()

    # Jednoznakowe elementy (np. 'a', '1') oraz znaki specjalne ('/', '.', '-', '?') są pomijane
    tokens = analyzer("http://sub.bank-login.com/a/1/verify?id=99")
    assert "http" in tokens
    assert "sub" in tokens
    assert "bank" in tokens
    assert "login" in tokens
    assert "com" in tokens
    assert "verify" in tokens
    assert "id" in tokens
    assert "99" in tokens
    assert "a" not in tokens
    assert "1" not in tokens


def test_train_baseline_end_to_end_on_fixtures(tmp_path: Path):
    """Weryfikuje pełny przepływ: trening, ewaluacja, zapis modelu .joblib i pliku val_baseline.json."""
    fixtures_csv = Path("tests/fixtures/sample_urls.csv")
    df = pd.read_csv(fixtures_csv)

    # Podział fixtures na mały train i val do testu
    train_df = df.iloc[:150].reset_index(drop=True)
    val_df = df.iloc[150:].reset_index(drop=True)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    model_out = tmp_path / "models" / "baseline_mnb.joblib"
    results_dir = tmp_path / "reports" / "results"

    pipeline, metrics, dod_status = train_baseline(
        train_path=train_csv,
        val_path=val_csv,
        model_output_path=model_out,
        results_dir=results_dir,
    )

    # Sprawdzenie zapisu modelu i DoD
    assert model_out.exists()
    assert all(dod_status.values())

    # Sprawdzenie wygenerowania pliku val_baseline.json z wszystkimi metrykami z evaluate.py
    val_json = results_dir / "val_baseline.json"
    assert val_json.exists()

    with open(val_json, encoding="utf-8") as f:
        saved_metrics = json.load(f)

    expected_keys = {"f1", "precision", "recall", "fpr", "fnr", "roc_auc"}
    assert expected_keys.issubset(saved_metrics.keys())
    for key in expected_keys:
        assert 0.0 <= saved_metrics[key] <= 1.0
        assert saved_metrics[key] == pytest.approx(metrics[key], abs=1e-6)

    # Sprawdzenie gotowości modelu do Red Teamingu po wczytaniu z pliku
    loaded = load_model(model_out)
    test_urls = ["google.com/search", "login-paypal-security-alert.com/signin"]
    preds = loaded.predict(test_urls)
    probas = loaded.predict_proba(test_urls)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2,)
    assert set(preds).issubset({0, 1})
    assert probas.shape == (2, 2)
    np.testing.assert_allclose(probas.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_verify_dod_baseline_rejects_char_ngrams(tmp_path: Path):
    """Weryfikuje, że verify_dod_baseline odrzuca model używający n-gramów znakowych."""
    bad_pipe = Pipeline([
        ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(3, 5))),
        ("classifier", MultinomialNB()),
    ])
    bad_pipe.fit(["example.com", "phish.org"], [0, 1])
    bad_path = tmp_path / "bad_char.joblib"
    save_model(bad_pipe, bad_path)

    with pytest.raises(AssertionError, match="DoD 1: Model Baseline nie może korzystać z n-gramów znakowych"):
        verify_dod_baseline(bad_path)


def test_train_model_dispatcher_invalid_name():
    """Sprawdza rzucanie wyjątku ValueError dla nieznanego typu modelu."""
    with pytest.raises(ValueError, match="Nieobsługiwany typ modelu"):
        train_model(model_type="unknown_model")


def test_build_hybrid_cnb_and_svm_pipelines():
    """Weryfikuje strukturę hybrydowych potoków ComplementNB oraz LinearSVC (w tym CalibratedClassifierCV)."""
    cnb_pipe = build_cnb_pipeline(max_features=1000)
    assert isinstance(cnb_pipe, Pipeline)
    assert isinstance(cnb_pipe.named_steps["features"], FeatureUnion)
    assert isinstance(cnb_pipe.named_steps["classifier"], ComplementNB)
    assert cnb_pipe.named_steps["classifier"].alpha == CNB_ALPHA

    svm_pipe = build_svm_pipeline(max_features=1000, calibrated=False)
    assert isinstance(svm_pipe, Pipeline)
    assert isinstance(svm_pipe.named_steps["features"], FeatureUnion)
    assert isinstance(svm_pipe.named_steps["classifier"], LinearSVC)
    assert svm_pipe.named_steps["classifier"].C == SVM_C

    cal_svm_pipe = build_svm_pipeline(max_features=1000, calibrated=True)
    assert isinstance(cal_svm_pipe.named_steps["classifier"], CalibratedClassifierCV)


def test_run_stratified_cv_no_leakage_and_stats():
    """Weryfikuje działanie 5-fold StratifiedKFold wewnątrz Pipeline dla cnb i svm."""
    fixtures_csv = Path("tests/fixtures/sample_urls.csv")
    df = pd.read_csv(fixtures_csv)
    urls = df["URL"].astype(str)
    labels = df["Label"].astype(int)

    for model_key in ["cnb", "svm"]:
        cv_res = run_stratified_cv(
            urls=urls,
            labels=labels,
            model_key=model_key,
            n_splits=5,
            max_features=300,
            parallel=False,
        )
        assert cv_res["model"] == model_key
        assert cv_res["n_splits"] == 5
        assert len(cv_res["folds"]) == 5
        assert 0.0 <= cv_res["f1_mean"] <= 1.0
        assert cv_res["f1_std"] >= 0.0
        assert 0.0 <= cv_res["fpr_mean"] <= 1.0
        assert 0.0 <= cv_res["roc_auc_mean"] <= 1.0
        assert cv_res["fold_fit_time_mean_sec"] > 0.0
        assert cv_res["cv_wall_time_sec"] < 90.0


def test_train_hybrid_cnb_and_svm_comparison_table(tmp_path: Path):
    """Weryfikuje pełny trening cnb i svm oraz generowanie tabeli porównawczej w cv_comparison.json."""
    fixtures_csv = Path("tests/fixtures/sample_urls.csv")
    df = pd.read_csv(fixtures_csv)

    train_df = df.iloc[:150].reset_index(drop=True)
    val_df = df.iloc[150:].reset_index(drop=True)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    results_dir = tmp_path / "reports" / "results"
    cnb_out = tmp_path / "models" / "hybrid_cnb.joblib"
    svm_out = tmp_path / "models" / "hybrid_svm.joblib"

    _, cnb_val_metrics, cnb_dod = train_hybrid(
        model_key="cnb",
        train_path=train_csv,
        val_path=val_csv,
        model_output_path=cnb_out,
        results_dir=results_dir,
        max_features=300,
    )
    assert cnb_out.exists()
    assert all(cnb_dod.values())
    assert "f1" in cnb_val_metrics

    _, svm_val_metrics, svm_dod = train_hybrid(
        model_key="svm",
        train_path=train_csv,
        val_path=val_csv,
        model_output_path=svm_out,
        results_dir=results_dir,
        max_features=300,
    )
    assert svm_out.exists()
    assert all(svm_dod.values())
    assert "f1" in svm_val_metrics

    cv_json_path = results_dir / "cv_comparison.json"
    assert cv_json_path.exists()

    with open(cv_json_path, encoding="utf-8") as f:
        cv_data = json.load(f)

    # Tabela porównawcza musi zawierać wszystkie 3 modele: baseline, cnb, svm
    assert {"baseline", "cnb", "svm"}.issubset(cv_data["models"].keys())
    table_keys = [row["model_key"] for row in cv_data["comparison_table"]]
    assert table_keys == ["baseline", "cnb", "svm"]


def test_verify_dod_hybrid_rejects_slow_training(tmp_path: Path):
    """Sprawdza asercję DoD 1 dla przekroczenia limitu 90 sekund na CPU."""
    dummy_json = tmp_path / "cv_comparison.json"
    dummy_json.write_text(
        json.dumps({"models": {"cnb": {}}, "comparison_table": [{"model_key": "cnb"}]}),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="DoD 1 niespełnione"):
        verify_dod_hybrid(
            train_time_sec=95.0,
            cv_wall_time_sec=30.0,
            cv_comparison_path=dummy_json,
            model_key="cnb",
        )
