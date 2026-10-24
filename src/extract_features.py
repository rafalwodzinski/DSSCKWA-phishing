"""Moduł ekstrakcji cech leksykalnych i entropijnych adresów URL.

Zgodnie z wymaganiami projektu DSSCKWA-phishing implementuje transformator
LexicalExtractor zgodny ze standardem scikit-learn (BaseEstimator, TransformerMixin).
Wylicza zestaw 12 cech strukturalnych przydatnych w detekcji phishingu.
"""

from __future__ import annotations

import collections
import ipaddress
import math
import re
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin

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
