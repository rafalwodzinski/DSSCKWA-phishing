"""Testy jednostkowe modułu trenowania grupy kontrolnej Baseline (src/train.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from src.extract_features import LexicalExtractor
from src.train import (
    BASELINE_ALPHA,
    BASELINE_MAX_FEATURES,
    BASELINE_TOKEN_PATTERN,
    build_baseline_pipeline,
    load_model,
    save_model,
    train_baseline,
    train_model,
    verify_dod_baseline,
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

