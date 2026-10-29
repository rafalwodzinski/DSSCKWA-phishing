"""
tests/test_cli.py

Testy dymne i jednostkowe modułu src/predict.py — symulator inspekcji URL
w bramce ESG / serwerze proxy SOC.

Zakres:
  - Dym: pełne wywołanie pytest (importy, podstawowe uruchomienie).
  - Jednostkowe: werdykt PHISHING / BENIGN / UNCERTAIN, latency ≤ 10 ms,
    obsługa --file CSV i TXT, błędy CLI (brak modelu, brak pliku).
  - Integracja: classify_url wywołane z prawdziwą/fałszywą parą model+próg.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.predict import (
    DEFAULT_UNCERTAINTY_MARGIN,
    build_parser,
    classify_url,
    classify_urls,
    load_threshold,
    read_urls_from_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PHISHING_URL = "http://secure-login.paypa1.com/verify"
BENIGN_URL = "https://www.google.com"


def _make_mock_model(score: float, method: str = "decision_function") -> MagicMock:
    """Zwraca mock modelu sklearn zwracający stały wynik scoringu."""
    model = MagicMock()
    if method == "decision_function":
        model.decision_function.return_value = np.array([score])
        del model.predict_proba  # upewniamy się, że nie istnieje
    else:
        model.predict_proba.return_value = np.array([[1 - score, score]])
        del model.decision_function
    model.predict.return_value = np.array([1 if score >= 0 else 0])
    return model


@pytest.fixture
def threshold() -> float:
    return -0.088141


@pytest.fixture
def score_method() -> str:
    return "decision_function"


# ---------------------------------------------------------------------------
# 1. Werdykty: PHISHING / BENIGN / UNCERTAIN
# ---------------------------------------------------------------------------

class TestClassifyUrlVerdicts:
    """Weryfikacja poprawności werdyktów i pola score."""

    def test_phishing_verdict_above_threshold(self, threshold, score_method):
        score_val = threshold + 0.5  # wyraźnie powyżej progu + poza marginesem
        model = _make_mock_model(score_val)
        result = classify_url(PHISHING_URL, model, threshold, score_method)
        assert result["verdict"] == "PHISHING"
        assert abs(result["score"] - score_val) < 1e-5

    def test_benign_verdict_below_threshold(self, threshold, score_method):
        score_val = threshold - 0.5  # wyraźnie poniżej progu + poza marginesem
        model = _make_mock_model(score_val)
        result = classify_url(BENIGN_URL, model, threshold, score_method)
        assert result["verdict"] == "BENIGN"

    def test_uncertain_verdict_near_threshold(self, threshold, score_method):
        score_val = threshold + DEFAULT_UNCERTAINTY_MARGIN * 0.5  # blisko progu
        model = _make_mock_model(score_val)
        result = classify_url(PHISHING_URL, model, threshold, score_method, uncertainty_margin=DEFAULT_UNCERTAINTY_MARGIN)
        assert result["verdict"] == "UNCERTAIN"

    def test_uncertain_verdict_just_below_threshold(self, threshold, score_method):
        score_val = threshold - DEFAULT_UNCERTAINTY_MARGIN * 0.5
        model = _make_mock_model(score_val)
        result = classify_url(BENIGN_URL, model, threshold, score_method, uncertainty_margin=DEFAULT_UNCERTAINTY_MARGIN)
        assert result["verdict"] == "UNCERTAIN"

    def test_result_contains_required_keys(self, threshold, score_method):
        model = _make_mock_model(threshold + 1.0)
        result = classify_url(PHISHING_URL, model, threshold, score_method)
        required_keys = {"url", "verdict", "score", "threshold", "uncertainty_margin", "latency_ms"}
        assert required_keys.issubset(result.keys())

    def test_result_url_matches_input(self, threshold, score_method):
        model = _make_mock_model(threshold + 1.0)
        result = classify_url(PHISHING_URL, model, threshold, score_method)
        assert result["url"] == PHISHING_URL

    def test_result_threshold_matches(self, threshold, score_method):
        model = _make_mock_model(threshold + 1.0)
        result = classify_url(PHISHING_URL, model, threshold, score_method)
        assert abs(result["threshold"] - threshold) < 1e-5

    def test_predict_proba_score_method(self, threshold):
        score_val = 0.9  # wysoka pewność phishingu
        model = _make_mock_model(score_val, method="predict_proba")
        # próg 0.5 — standardowy dla predict_proba
        result = classify_url(PHISHING_URL, model, threshold=0.5, score_method="predict_proba")
        assert result["verdict"] == "PHISHING"


# ---------------------------------------------------------------------------
# 2. Latency ≤ 10 ms (kryterium DoD)
# ---------------------------------------------------------------------------

class TestLatencyDod:
    """Kryterium DoD: inferencja pojedynczego URL ≤ 10 ms."""

    def test_single_url_latency_under_10ms(self, threshold, score_method):
        """classify_url musi działać w ≤ 10 ms (mierzony czas w polu latency_ms)."""
        model = _make_mock_model(threshold + 1.0)
        result = classify_url(PHISHING_URL, model, threshold, score_method)
        assert result["latency_ms"] <= 10.0, (
            f"latency_ms = {result['latency_ms']:.3f} ms > 10 ms (naruszenie DoD)"
        )

    def test_latency_field_is_positive(self, threshold, score_method):
        model = _make_mock_model(threshold - 1.0)
        result = classify_url(BENIGN_URL, model, threshold, score_method)
        assert result["latency_ms"] > 0.0

    def test_batch_latency_scales_linearly(self, threshold, score_method):
        """Czas batcha N URL-i nie przekracza N × 10 ms."""
        n = 20
        urls = [f"http://test{i}.com/path" for i in range(n)]
        model = _make_mock_model(threshold + 1.0)
        t0 = time.perf_counter()
        results = classify_urls(urls, model, threshold, score_method)
        total_ms = (time.perf_counter() - t0) * 1000.0
        assert total_ms <= n * 10.0, (
            f"Batch {n} URL-i: {total_ms:.1f} ms > {n * 10} ms"
        )
        assert len(results) == n


# ---------------------------------------------------------------------------
# 3. Wczytywanie URL-i z pliku
# ---------------------------------------------------------------------------

class TestReadUrlsFromFile:
    """Weryfikacja read_urls_from_file dla pliku TXT i CSV."""

    def test_reads_plain_text_file(self, tmp_path):
        p = tmp_path / "urls.txt"
        p.write_text(
            "http://phish.example.com/login\n"
            "https://www.safe.org/home\n"
            "# komentarz - ignorowany\n"
            "\n"
            "http://another.bad.com\n",
            encoding="utf-8",
        )
        urls = read_urls_from_file(p)
        assert len(urls) == 3
        assert "http://phish.example.com/login" in urls
        assert "https://www.safe.org/home" in urls
        assert "http://another.bad.com" in urls

    def test_reads_csv_file_with_url_column(self, tmp_path):
        p = tmp_path / "urls.csv"
        p.write_text(
            "URL,Label\n"
            "http://evil.phish.com/verify,1\n"
            "https://legit-bank.pl/login,0\n",
            encoding="utf-8",
        )
        urls = read_urls_from_file(p)
        assert len(urls) == 2
        assert "http://evil.phish.com/verify" in urls

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_urls_from_file(tmp_path / "nonexistent.txt")

    def test_ignores_comments_and_empty_lines(self, tmp_path):
        p = tmp_path / "urls.txt"
        p.write_text("# nagłówek\n\nhttp://real-url.com\n# ignoruj\n", encoding="utf-8")
        urls = read_urls_from_file(p)
        assert urls == ["http://real-url.com"]


# ---------------------------------------------------------------------------
# 4. Wczytywanie progu z pliku JSON
# ---------------------------------------------------------------------------

class TestLoadThreshold:
    """Weryfikacja load_threshold dla poprawnych i brakujących plików."""

    def test_loads_threshold_from_json(self, tmp_path):
        p = tmp_path / "threshold.json"
        p.write_text(json.dumps({
            "threshold": -0.088141,
            "score_method": "decision_function",
        }), encoding="utf-8")
        tau, method = load_threshold(p)
        assert abs(tau - -0.088141) < 1e-6
        assert method == "decision_function"

    def test_default_score_method_when_missing(self, tmp_path):
        p = tmp_path / "threshold.json"
        p.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
        _, method = load_threshold(p)
        assert method == "decision_function"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_threshold(tmp_path / "no_such_file.json")


# ---------------------------------------------------------------------------
# 5. Parser CLI
# ---------------------------------------------------------------------------

class TestCliParser:
    """Weryfikacja parsera argparse."""

    def test_parser_url_mode(self):
        parser = build_parser()
        args = parser.parse_args(["--url", PHISHING_URL])
        assert args.url == PHISHING_URL
        assert args.file is None

    def test_parser_file_mode(self, tmp_path):
        p = tmp_path / "urls.txt"
        p.write_text("http://test.com\n")
        parser = build_parser()
        args = parser.parse_args(["--file", str(p)])
        assert args.file == p
        assert args.url is None

    def test_parser_url_and_file_are_mutually_exclusive(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--url", PHISHING_URL, "--file", "urls.txt"])

    def test_parser_requires_url_or_file(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_custom_uncertainty_margin(self):
        parser = build_parser()
        args = parser.parse_args(["--url", BENIGN_URL, "--uncertainty-margin", "0.1"])
        assert abs(args.uncertainty_margin - 0.1) < 1e-9

    def test_parser_pretty_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--url", BENIGN_URL, "--pretty"])
        assert args.pretty is True


# ---------------------------------------------------------------------------
# 6. Testy dymne end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_model_artifacts(tmp_path_factory) -> tuple[Path, Path]:
    """Zwraca ścieżki (model_path, threshold_path).
    
    Jeśli istnieją docelowe pliki models/best_hybrid.joblib i models/threshold.json,
    używa ich. W przeciwnym razie szybko trenuje lekki model testowy (0.04s) na
    tests/fixtures/sample_urls.csv, dzięki czemu testy end-to-end zawsze przechodzą
    nawet na świeżo sklonowanym repozytorium bez pomijania (0 skipped).
    """
    real_model = Path("models/best_hybrid.joblib")
    real_thresh = Path("models/threshold.json")
    if real_model.exists() and real_thresh.exists():
        return real_model, real_thresh

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from src.extract_features import build_feature_pipeline
    import joblib

    tmp_dir = tmp_path_factory.mktemp("smoke_artifacts")
    model_path = tmp_dir / "best_hybrid.joblib"
    thresh_path = tmp_dir / "threshold.json"

    df = pd.read_csv("tests/fixtures/sample_urls.csv")
    pipe = Pipeline([
        ("features", build_feature_pipeline(max_features=500)),
        ("clf", LinearSVC(C=1.0, random_state=42)),
    ])
    pipe.fit(df["URL"], df["Label"])
    joblib.dump(pipe, model_path)

    thresh_path.write_text(
        json.dumps({
            "threshold": 0.0,
            "optimal_threshold": 0.0,
            "score_method": "decision_function",
        }),
        encoding="utf-8",
    )
    return model_path, thresh_path


class TestSmokeEndToEnd:
    """Testy dymne weryfikujące pełny potok inferencji oraz CLI subprocess."""

    def test_smoke_single_phishing_url(self, fixture_model_artifacts):
        """DoD: klasyfikacja pojedynczego URL ≤ 10 ms z wytrenowanym pipeline'em."""
        from src.predict import load_model, load_threshold, classify_url
        model_path, thresh_path = fixture_model_artifacts
        model = load_model(model_path)
        threshold, score_method = load_threshold(thresh_path)
        result = classify_url(PHISHING_URL, model, threshold, score_method)

        assert result["verdict"] in {"PHISHING", "BENIGN", "UNCERTAIN"}
        assert result["latency_ms"] <= 10.0, (
            f"Naruszenie DoD: latency_ms = {result['latency_ms']:.3f} ms > 10 ms"
        )
        assert isinstance(result["score"], float)

    def test_smoke_benign_url(self, fixture_model_artifacts):
        """Legalny URL powinien być sklasyfikowany jako BENIGN lub UNCERTAIN."""
        from src.predict import load_model, load_threshold, classify_url
        model_path, thresh_path = fixture_model_artifacts
        model = load_model(model_path)
        threshold, score_method = load_threshold(thresh_path)
        result = classify_url(BENIGN_URL, model, threshold, score_method)
        assert result["verdict"] in {"BENIGN", "UNCERTAIN"}

    def test_smoke_cli_subprocess(self, fixture_model_artifacts):
        """Pełny dym CLI przez subprocess — wyjście to poprawny JSON."""
        model_path, thresh_path = fixture_model_artifacts
        result = subprocess.run(
            [
                sys.executable,
                "src/predict.py",
                "--url",
                PHISHING_URL,
                "--model",
                str(model_path),
                "--threshold",
                str(thresh_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["verdict"] in {"PHISHING", "BENIGN", "UNCERTAIN"}
        assert data["latency_ms"] <= 10.0

    def test_smoke_cli_file_mode(self, fixture_model_artifacts, tmp_path):
        """Pełny dym CLI --file — wyjście to lista JSON-ów."""
        model_path, thresh_path = fixture_model_artifacts
        p = tmp_path / "urls.txt"
        p.write_text(f"{PHISHING_URL}\n{BENIGN_URL}\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "src/predict.py",
                "--file",
                str(p),
                "--model",
                str(model_path),
                "--threshold",
                str(thresh_path),
                "--pretty",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 2
        for item in data:
            assert item["verdict"] in {"PHISHING", "BENIGN", "UNCERTAIN"}

    def test_smoke_cli_missing_model_exits_1(self, tmp_path):
        """CLI zwraca kod wyjścia 1 gdy model nie istnieje."""
        result = subprocess.run(
            [
                sys.executable,
                "src/predict.py",
                "--url",
                PHISHING_URL,
                "--model",
                str(tmp_path / "no_model.joblib"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

