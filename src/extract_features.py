"""Moduł ekstrakcji cech leksykalnych, entropijnych oraz znakowych TF-IDF adresów URL.

Zgodnie z wymaganiami projektu DSSCKWA-phishing implementuje:
- transformator LexicalExtractor (12 cech leksykalno-entropijnych),
- hybrydowy pipeline FeatureUnion łączący znakowy TF-IDF (3-5) oraz LexicalExtractor
  ze skalowaniem MaxAbsScaler (punkt synchronizacji S2).
"""

from __future__ import annotations

import argparse
import collections
import ipaddress
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

DEFAULT_TRAIN_FILE = Path("data/processed/train.csv")
DEFAULT_VAL_FILE = Path("data/processed/val.csv")
DEFAULT_TEST_FILE = Path("data/processed/test.csv")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_PIPELINE_FILE = DEFAULT_MODELS_DIR / "feature_pipeline.joblib"

TFIDF_MAX_FEATURES = 30000
TFIDF_NGRAM_RANGE = (3, 5)

FEATURE_NAMES: list[str] = [
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

_PERCENT_ENCODED_RE = re.compile(r"%[0-9a-fA-F]{2}")
_IPV4_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def _calculate_shannon_entropy(text: str) -> float:
    """Oblicza entropię Shannona rozkładu znaków w tekście."""
    if not text:
        return 0.0
    length = len(text)
    counts = collections.Counter(text)
    entropy = -sum((c / length) * math.log2(c / length) for c in counts.values())
    return float(entropy)


def _is_ipv4(host: str) -> float:
    """Sprawdza, czy przekazany host jest poprawnym adresem IPv4."""
    if not host or not _IPV4_RE.match(host):
        return 0.0
    try:
        ipaddress.IPv4Address(host)
        return 1.0
    except ValueError:
        return 0.0


class LexicalExtractor(BaseEstimator, TransformerMixin):
    """Transformator scikit-learn wyliczający 12 cech leksykalno-entropijnych adresu URL.

    Cechy:
    1. url_length - długość całkowita URL
    2. host_length - długość hosta (z tldextract)
    3. path_length - długość ścieżki
    4. num_dots - liczba znaków '.'
    5. has_at - obecność znaku '@' (1.0 lub 0.0)
    6. shannon_entropy - entropia Shannona znaków w URL
    7. num_hyphens - liczba myślników '-'
    8. digit_ratio - stosunek liczby cyfr do długości URL
    9. num_subdomains - liczba subdomen wyznaczona przez tldextract
    10. num_query_params - liczba znaków parametrów query ('?', '&', '=')
    11. has_ip - czy host jest adresem IPv4 (1.0 lub 0.0)
    12. num_percent_encoded - liczba znaków/sekwencji kodowanych procentowo (%xx)
    """

    def __init__(self) -> None:
        # Inicjalizacja tldextract w trybie offline (brak zapytań sieciowych HTTP, suffix_list_urls=())
        self._extractor = tldextract.TLDExtract(
            cache_dir=False,
            suffix_list_urls=(),
        )

    def fit(self, X: Any, y: Any = None) -> "LexicalExtractor":
        """Dopasowuje transformator do danych (stateless transformer)."""
        return self

    def _extract_single(self, url: str) -> list[float]:
        """Wylicza wektor 12 cech dla pojedynczego adresu URL."""
        if not isinstance(url, str):
            url = "" if url is None else str(url)

        url_len = len(url)
        if url_len == 0:
            return [0.0] * 12

        # Ekstrakcja ścieżki i hosta z urlsplit
        split_url = url if "://" in url or url.startswith("//") else f"//{url}"
        try:
            parsed = urlsplit(split_url)
            path_len = len(parsed.path)
            parsed_host = parsed.hostname or ""
        except Exception:
            path_len = 0
            parsed_host = ""

        # Ekstrakcja domeny/hosta i subdomeny z tldextract
        ext = self._extractor(url)
        host = ext.fqdn if ext.fqdn else parsed_host
        host_len = len(host)

        # Liczba kropek i myślników
        num_dots = url.count(".")
        num_hyphens = url.count("-")
        has_at = 1.0 if "@" in url else 0.0

        # Entropia Shannona
        entropy = _calculate_shannon_entropy(url)

        # Stosunek cyfr
        digit_count = sum(c.isdigit() for c in url)
        digit_ratio = digit_count / url_len

        # Liczba subdomen
        subdomain = ext.subdomain
        num_subdomains = len(subdomain.split(".")) if subdomain else 0

        # Liczba parametrów zapytania (?, &, =)
        num_query_params = url.count("?") + url.count("&") + url.count("=")

        # Czy host jest adresem IPv4
        has_ip = _is_ipv4(host)

        # Liczba sekwencji kodowanych procentowo (%xx)
        num_percent_encoded = len(_PERCENT_ENCODED_RE.findall(url))

        return [
            float(url_len),
            float(host_len),
            float(path_len),
            float(num_dots),
            float(has_at),
            float(entropy),
            float(num_hyphens),
            float(digit_ratio),
            float(num_subdomains),
            float(num_query_params),
            float(has_ip),
            float(num_percent_encoded),
        ]

    def transform(
        self, X: pd.Series | np.ndarray | Sequence[str] | Iterable[str]
    ) -> np.ndarray:
        """Ekstrahuje 12 cech dla kolekcji adresów URL.

        Args:
            X: Kolekcja adresów URL (Series, ndarray, lista stringów).

        Returns:
            np.ndarray o wymiarach (N, 12) i typie float64 bez NaN/Inf.
        """
        if isinstance(X, pd.Series):
            urls = X.tolist()
        elif isinstance(X, np.ndarray):
            urls = X.ravel().tolist()
        elif isinstance(X, (list, tuple)):
            urls = list(X)
        else:
            urls = list(X)

        features = [self._extract_single(url) for url in urls]
        arr = np.asarray(features, dtype=np.float64)

        # Gwarancja braku wartości NaN/Inf
        if arr.size > 0:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        return arr

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Zwraca nazwy 12 wyliczanych cech."""
        return list(FEATURE_NAMES)


def build_feature_pipeline(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    sublinear_tf: bool = True,
) -> FeatureUnion:
    """Buduje FeatureUnion łączący znakowy TF-IDF oraz LexicalExtractor.

    1. TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=30000, sublinear_tf=True)
    2. Pipeline([('lexical', LexicalExtractor()), ('scaler', MaxAbsScaler())])
       - MaxAbsScaler skaluje cechy do [0, 1] bez niszczenia rzadkości macierzy
         i gwarantuje brak wartości ujemnych (wymóg dla ComplementNB).

    Args:
        max_features: Maksymalna liczba cech dla TF-IDF.
        ngram_range: Zakres n-gramów znakowych (min_n, max_n).
        sublinear_tf: Czy stosować sublinear tf (1 + log(tf)).

    Returns:
        FeatureUnion: Skonstruowany potok ekstrakcji cech.
    """
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )
    lexical_pipe = Pipeline([
        ("lexical", LexicalExtractor()),
        ("scaler", MaxAbsScaler()),
    ])
    return FeatureUnion([
        ("tfidf", tfidf),
        ("lexical", lexical_pipe),
    ])


def save_feature_pipeline(pipeline: FeatureUnion, path: Path | str) -> None:
    """Serializuje pipeline ekstrakcji cech do pliku .joblib."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, dest)


def load_feature_pipeline(path: Path | str) -> FeatureUnion:
    """Wczytuje zserializowany pipeline z pliku .joblib."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku pipeline'u: {source}")
    return joblib.load(source)


def verify_dod_features(
    X_train: scipy.sparse.spmatrix,
    pipeline_path: Path,
) -> dict[str, bool]:
    """Weryfikuje kryteria Definition of Done (DoD) dla Issue i punktu synchronizacji S2.

    Kryteria akceptacji:
    - Wyjście transformatora to macierz rzadka scipy.sparse bez wartości ujemnych (X.min() >= 0).
    - Obiekt models/feature_pipeline.joblib zapisuje się i wczytuje poprawnie.
    """
    # 1. Macierz rzadka scipy.sparse
    is_sparse = scipy.sparse.issparse(X_train)
    assert is_sparse, "DoD 1: Wyjście transformatora nie jest macierzą scipy.sparse"

    # 2. Brak wartości ujemnych
    min_val = float(X_train.min()) if X_train.nnz > 0 else 0.0
    has_no_negative = bool(min_val >= 0.0 and (X_train.data.size == 0 or X_train.data.min() >= 0.0))
    assert has_no_negative, f"DoD 1: Wykryto wartości ujemne w macierzy cech (min={min_val})"

    # 3. Zapis i odczyt z joblib
    file_exists = pipeline_path.exists()
    assert file_exists, f"DoD 2: Plik pipeline'u nie istnieje: {pipeline_path}"

    loaded_pipeline = load_feature_pipeline(pipeline_path)
    assert isinstance(loaded_pipeline, FeatureUnion), (
        f"DoD 2: Wczytany obiekt nie jest FeatureUnion ({type(loaded_pipeline)})"
    )

    # Weryfikacja działania wczytanego modelu
    test_urls = ["https://example.com/verify", "http://test-phishing.org/login"]
    X_test_check = loaded_pipeline.transform(test_urls)
    assert scipy.sparse.issparse(X_test_check), "DoD 2: Wczytany model nie generuje macierzy scipy.sparse"
    assert X_test_check.shape[0] == 2, "DoD 2: Nieprawidłowa liczba próbek po wczytaniu"
    assert (X_test_check.data.size == 0 or X_test_check.data.min() >= 0.0), (
        "DoD 2: Wczytany model generuje wartości ujemne"
    )

    return {
        "is_sparse_matrix": is_sparse,
        "non_negative_values": has_no_negative,
        "pipeline_joblib_valid": True,
    }


def extract_features(
    train_path: Path = DEFAULT_TRAIN_FILE,
    val_path: Path | None = DEFAULT_VAL_FILE,
    test_path: Path | None = DEFAULT_TEST_FILE,
    output_pipeline_path: Path = DEFAULT_PIPELINE_FILE,
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
) -> tuple[FeatureUnion, scipy.sparse.spmatrix, scipy.sparse.spmatrix | None, scipy.sparse.spmatrix | None, dict[str, bool]]:
    """Uruchamia pełną procedurę ekstrakcji cech: fit na train, transformacja val/test, zapis i weryfikacja DoD."""
    print("=== Ekstrakcja cech (punkt synchronizacji S2) ===")
    print(f"1. Wczytywanie danych treningowych: {train_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Brak pliku treningowego: {train_path}")

    train_df = pd.read_csv(train_path)
    if "URL" not in train_df.columns:
        raise ValueError(f"Brak wymaganej kolumny 'URL' w {train_path}")

    print(f"   Liczba próbek train: {len(train_df):,}")

    # 2. Budowanie pipeline'u
    print(f"2. Budowanie FeatureUnion (TF-IDF char {ngram_range}, max_features={max_features} + LexicalExtractor + MaxAbsScaler)...")
    pipeline = build_feature_pipeline(max_features=max_features, ngram_range=ngram_range)

    # 3. Dopasowanie WYŁĄCZNIE na train.csv
    print("3. Dopasowywanie pipeline'u (fit_transform) WYŁĄCZNIE na train.csv...")
    X_train = pipeline.fit_transform(train_df["URL"])
    print(f"   Rozmiar macierzy train: {X_train.shape} (niezerowe: {X_train.nnz:,})")

    # 4. Transformacja zbioru walidacyjnego (jeśli istnieje)
    X_val = None
    if val_path and val_path.exists():
        print(f"4. Transformacja zbioru walidacyjnego: {val_path}")
        val_df = pd.read_csv(val_path)
        if "URL" in val_df.columns:
            X_val = pipeline.transform(val_df["URL"])
            print(f"   Rozmiar macierzy val: {X_val.shape} (niezerowe: {X_val.nnz:,})")

    # 5. Transformacja zbioru testowego (jeśli istnieje)
    X_test = None
    if test_path and test_path.exists():
        print(f"5. Transformacja zbioru testowego: {test_path}")
        test_df = pd.read_csv(test_path)
        if "URL" in test_df.columns:
            X_test = pipeline.transform(test_df["URL"])
            print(f"   Rozmiar macierzy test: {X_test.shape} (niezerowe: {X_test.nnz:,})")

    # 6. Zapis pipeline'u do models/feature_pipeline.joblib
    print(f"6. Zapisywanie pipeline'u do: {output_pipeline_path}")
    save_feature_pipeline(pipeline, output_pipeline_path)

    # 7. Weryfikacja kryteriów DoD
    print("7. Weryfikacja kryteriów DoD (punkt S2)...")
    dod_results = verify_dod_features(X_train, output_pipeline_path)
    for criterion, status in dod_results.items():
        print(f"   - {criterion}: {'[OK]' if status else '[BŁĄD]'}")

    print("\nGotowość punktu S2: Feature pipeline wygenerowany i przetestowany!")
    return pipeline, X_train, X_val, X_test, dod_results


def main() -> None:
    """Punkt wejścia CLI dla skryptu ekstrakcji cech."""
    parser = argparse.ArgumentParser(description="Ekstrakcja cech i budowa pipeline'u S2")
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help="Ścieżka do pliku train.csv",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_VAL_FILE,
        help="Ścieżka do pliku val.csv",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST_FILE,
        help="Ścieżka do pliku test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PIPELINE_FILE,
        help="Ścieżka zapisu pipeline'u .joblib (domyślnie models/feature_pipeline.joblib)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=TFIDF_MAX_FEATURES,
        help=f"Maksymalna liczba cech TF-IDF (domyślnie {TFIDF_MAX_FEATURES})",
    )
    args = parser.parse_args()

    try:
        extract_features(
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            output_pipeline_path=args.output,
            max_features=args.max_features,
        )
    except Exception as e:
        print(f"\nBłąd wykonania ekstrakcji cech: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
