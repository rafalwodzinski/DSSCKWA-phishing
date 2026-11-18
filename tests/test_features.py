"""Testy jednostkowe modułu ekstrakcji cech leksykalnych (src/extract_features.py)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from src.extract_features import (
    FEATURE_NAMES,
    LexicalExtractor,
    build_feature_pipeline,
    extract_features,
    load_feature_pipeline,
    save_feature_pipeline,
    verify_dod_features,
)


@pytest.fixture
def sample_urls() -> list[str]:
    """Zestaw 5 znanych adresów URL sprawdzających kluczowe cechy (w tym @, IP, %2f)."""
    return [
        "https://www.example.com/index.html",
        "http://admin:secret@phishing-target.com/login",
        "http://192.168.1.1/admin/setup.php?user=admin&pass=123",
        "https://bank.com/search?q=%2flogin%20test",
        "sub1.sub2.malicious-site.org/path-with-hyphens/action",
    ]


def test_lexical_extractor_output_shape_and_dtype(sample_urls: list[str]):
    """Weryfikacja wymiarów (N, 12), typu float64 oraz braku wartości NaN/Inf."""
    extractor = LexicalExtractor()
    features = extractor.transform(sample_urls)

    assert isinstance(features, np.ndarray)
    assert features.shape == (5, 12)
    assert features.dtype == np.float64
    assert not np.isnan(features).any(), "Wykryto wartości NaN w macierzy cech"
    assert not np.isinf(features).any(), "Wykryto wartości Inf w macierzy cech"


def test_feature_names_out():
    """Weryfikacja metody get_feature_names_out()."""
    extractor = LexicalExtractor()
    feature_names = extractor.get_feature_names_out()

    assert len(feature_names) == 12
    assert feature_names == [
        "url_length",
        "host_length",
        "path_length",
        "num_dots",
        "has_at",
        "shannon_entropy",
        "num_hyphens",
        "digit_ratio",
        "num_subdomains",
        "num_query_params",
        "has_ip",
        "num_percent_encoded",
    ]


def test_five_known_urls_exact_calculations():
    """Weryfikacja poprawności wyliczeń na 5 sprecyzowanych URL-ach."""
    extractor = LexicalExtractor()

    # URL 1: Standardowy URL
    url_1 = "https://www.example.com/index.html"
    feat_1 = extractor.transform([url_1])[0]
    # url_length = len("https://www.example.com/index.html") = 34
    assert feat_1[0] == 34.0
    # host = "www.example.com" -> 15
    assert feat_1[1] == 15.0
    # path = "/index.html" -> 11
    assert feat_1[2] == 11.0
    # num_dots: "www.example.com/index.html" ma 3 kropki
    assert feat_1[3] == 3.0
    # has_at = 0.0
    assert feat_1[4] == 0.0
    # num_hyphens = 0.0
    assert feat_1[6] == 0.0
    # has_ip = 0.0
    assert feat_1[10] == 0.0

    # URL 2: URL zawierający znak '@'
    url_2 = "http://admin:secret@phishing-target.com/login"
    feat_2 = extractor.transform([url_2])[0]
    # has_at musi być 1.0
    assert feat_2[4] == 1.0
    # num_hyphens: "phishing-target" -> 1
    assert feat_2[6] == 1.0

    # URL 3: URL z adresem IPv4 oraz wieloma parametrami zapytania (?, &, =)
    url_3 = "http://192.168.1.1/admin/setup.php?user=admin&pass=123"
    feat_3 = extractor.transform([url_3])[0]
    # has_ip musi wynosić 1.0
    assert feat_3[10] == 1.0
    # parametry zapytania: '?' (1) + '&' (1) + '=' (2) -> 4
    assert feat_3[9] == 4.0
    # digit_ratio: cyfry 19216811 (8 cyfr) + 123 (3 cyfry) = 11 cyfr
    assert feat_3[7] == pytest.approx(11 / len(url_3), abs=1e-4)

    # URL 4: URL z kodowaniem procentowym (%2f, %20)
    url_4 = "https://bank.com/search?q=%2flogin%20test"
    feat_4 = extractor.transform([url_4])[0]
    # %2f oraz %20 -> 2 sekwencje percent-encoded
    assert feat_4[11] == 2.0

    # URL 5: URL z subdomenami ("sub1.sub2.malicious-site.org")
    url_5 = "sub1.sub2.malicious-site.org/path-with-hyphens/action"
    feat_5 = extractor.transform([url_5])[0]
    # subdomain = "sub1.sub2" -> 2 subdomeny
    assert feat_5[8] == 2.0
    # num_hyphens: "malicious-site" (1) + "path-with-hyphens" (2) = 3
    assert feat_5[6] == 3.0


def test_extraction_speed_per_url():
    """Weryfikacja kryterium wydajnościowego: czas ekstrakcji < 2 ms na URL."""
    extractor = LexicalExtractor()
    urls = [
        "https://www.google.com/search?q=machine+learning&oq=ml",
        "http://user@test-bank-verification.com/auth?token=123456&secure=true",
        "https://sub.domain.website.org/files/doc.pdf",
        "http://10.0.0.1:8080/portal/index.html",
        "https://example.com/%2f%20%3a%40",
    ] * 50  # 250 URL-i

    # Rozgrzewka (JIT/inicjalizacja struktur)
    extractor.transform(urls[:10])

    t0 = time.perf_counter()
    res = extractor.transform(urls)
    total_time_ms = (time.perf_counter() - t0) * 1000
    avg_time_per_url_ms = total_time_ms / len(urls)

    assert res.shape == (250, 12)
    # Wymóg kryterium akceptacji: < 2 ms na URL
    assert (
        avg_time_per_url_ms < 2.0
    ), f"Zbyt wolna ekstrakcja: {avg_time_per_url_ms:.3f} ms/URL"


def test_input_types_compatibility():
    """Weryfikacja obsługi wejścia jako pandas.Series, numpy.ndarray i list."""
    extractor = LexicalExtractor()
    urls = ["https://example.com", "http://test.org/path"]

    res_list = extractor.transform(urls)
    res_series = extractor.transform(pd.Series(urls))
    res_array = extractor.transform(np.array(urls))

    np.testing.assert_array_equal(res_list, res_series)
    np.testing.assert_array_equal(res_list, res_array)


def test_sklearn_pipeline_compatibility():
    """Weryfikacja działania w potoku scikit-learn Pipeline."""
    pipeline = Pipeline([
        ("lexical", LexicalExtractor()),
    ])

    urls = pd.Series(["https://example.com/login", "http://phish.xyz?id=99"])
    output = pipeline.fit_transform(urls)

    assert isinstance(output, np.ndarray)
    assert output.shape == (2, 12)


def test_edge_cases_handling():
    """Weryfikacja odporności na skrajne wartości: pusty string, spacje, brak protokołu."""
    extractor = LexicalExtractor()
    edge_urls = ["", "   ", "just_a_word", "123456"]

    res = extractor.transform(edge_urls)
    assert res.shape == (4, 12)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
    # Pusty string powinien mieć same zera
    assert np.all(res[0] == 0.0)


def test_build_feature_pipeline_structure():
    """Weryfikacja struktury i parametrów hybrydowego FeatureUnion."""
    pipeline = build_feature_pipeline(max_features=15000, ngram_range=(3, 5))

    assert isinstance(pipeline, FeatureUnion)
    transformer_dict = dict(pipeline.transformer_list)

    assert "tfidf" in transformer_dict
    assert "lexical" in transformer_dict

    tfidf = transformer_dict["tfidf"]
    assert isinstance(tfidf, TfidfVectorizer)
    assert tfidf.analyzer == "char"
    assert tfidf.ngram_range == (3, 5)
    assert tfidf.max_features == 15000
    assert tfidf.sublinear_tf is True

    lex_pipe = transformer_dict["lexical"]
    assert isinstance(lex_pipe, Pipeline)
    pipe_steps = dict(lex_pipe.steps)
    assert isinstance(pipe_steps["lexical"], LexicalExtractor)
    assert isinstance(pipe_steps["scaler"], MaxAbsScaler)


def test_hybrid_pipeline_output_format_and_dod(sample_urls: list[str]):
    """Weryfikacja kryterium DoD 1: macierz rzadka scipy.sparse i brak wartości ujemnych."""
    pipeline = build_feature_pipeline(max_features=500)
    X = pipeline.fit_transform(sample_urls)

    # 1. Sprawdzenie rzadkości macierzy
    assert scipy.sparse.issparse(X), "Wyjście nie jest macierzą scipy.sparse"
    assert isinstance(X, scipy.sparse.csr_matrix)

    # 2. Liczba próbek i cech
    assert X.shape[0] == len(sample_urls)
    # 12 cech leksykalnych + n cech TF-IDF
    assert X.shape[1] > 12

    # 3. Brak wartości ujemnych (DoD 1)
    assert X.min() >= 0.0, f"Znaleziono wartości ujemne: {X.min()}"
    if X.data.size > 0:
        assert X.data.min() >= 0.0, f"Wartość ujemna w X.data: {X.data.min()}"

    # 4. Typ danych
    assert X.dtype == np.float64


def test_lexical_pipeline_scaler_bounds(sample_urls: list[str]):
    """Weryfikacja skalowania cech leksykalnych przez MaxAbsScaler do przedziału [0, 1]."""
    lex_pipe = Pipeline([
        ("lexical", LexicalExtractor()),
        ("scaler", MaxAbsScaler()),
    ])

    scaled = lex_pipe.fit_transform(sample_urls)
    assert isinstance(scaled, np.ndarray)
    assert scaled.shape == (len(sample_urls), 12)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0


def test_fit_only_on_train_no_data_leakage():
    """Weryfikacja podziału: fit wyłącznie na train, transform na osobnym zbiorze."""
    fixtures_csv = Path("tests/fixtures/sample_urls.csv")
    if fixtures_csv.exists():
        df = pd.read_csv(fixtures_csv)
    else:
        df = pd.DataFrame({"URL": [f"https://site{i}.com/path" for i in range(100)]})

    train_urls = df["URL"].iloc[:140]
    test_urls = df["URL"].iloc[140:]

    pipeline = build_feature_pipeline(max_features=1000)
    X_train = pipeline.fit_transform(train_urls)
    X_test = pipeline.transform(test_urls)

    assert scipy.sparse.issparse(X_train)
    assert scipy.sparse.issparse(X_test)
    assert X_train.shape[0] == len(train_urls)
    assert X_test.shape[0] == len(test_urls)
    assert X_train.shape[1] == X_test.shape[1]
    assert X_test.min() >= 0.0


def test_feature_names_out_composition(sample_urls: list[str]):
    """Weryfikacja kompletności i prefiksów cech z get_feature_names_out()."""
    pipeline = build_feature_pipeline(max_features=200)
    X = pipeline.fit_transform(sample_urls)
    feature_names = pipeline.get_feature_names_out()

    assert len(feature_names) == X.shape[1]
    assert any(name.startswith("tfidf__") for name in feature_names)
    assert any(name.startswith("lexical__") for name in feature_names)


def test_pipeline_serialization_and_reloading(tmp_path: Path, sample_urls: list[str]):
    """Weryfikacja kryterium DoD 2: serializacja do joblib, poprawny odczyt i weryfikacja DoD."""
    pipeline = build_feature_pipeline(max_features=300)
    X_train = pipeline.fit_transform(sample_urls)

    save_path = tmp_path / "models" / "feature_pipeline.joblib"
    save_feature_pipeline(pipeline, save_path)
    assert save_path.exists()

    # Sprawdzenie funkcji weryfikującej DoD
    dod_status = verify_dod_features(X_train, save_path)
    assert dod_status["is_sparse_matrix"] is True
    assert dod_status["non_negative_values"] is True
    assert dod_status["pipeline_joblib_valid"] is True

    # Weryfikacja załadowanego pipeline'u
    loaded_pipe = load_feature_pipeline(save_path)
    X_loaded = loaded_pipe.transform(sample_urls)

    assert scipy.sparse.issparse(X_loaded)
    np.testing.assert_allclose(X_train.toarray(), X_loaded.toarray(), rtol=1e-5, atol=1e-5)


def test_extract_features_end_to_end(tmp_path: Path):
    """Weryfikacja pełnego wywołania extract_features na małym zbiorze fixtures."""
    fixtures_csv = Path("tests/fixtures/sample_urls.csv")
    out_pipe = tmp_path / "feature_pipeline.joblib"

    pipe, X_tr, X_va, X_te, dod = extract_features(
        train_path=fixtures_csv,
        val_path=fixtures_csv,
        test_path=fixtures_csv,
        output_pipeline_path=out_pipe,
        max_features=500,
    )

    assert scipy.sparse.issparse(X_tr)
    assert scipy.sparse.issparse(X_va)
    assert scipy.sparse.issparse(X_te)
    assert out_pipe.exists()
    assert all(dod.values())
