"""Testy jednostkowe modułu ekstrakcji cech leksykalnych (src/extract_features.py)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.extract_features import FEATURE_NAMES, LexicalExtractor


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
