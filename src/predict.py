"""Moduł inspekcji URL — symulator działania bramki pocztowej ESG / serwera proxy SOC.

Klasyfikuje adresy URL przy użyciu wytrenowanego pipeline'u i optymalnego progu
decyzyjnego tau* (models/threshold.json). Obsługuje dwa tryby wejścia:

  --url <url>         Klasyfikacja pojedynczego adresu URL.
  --file <ścieżka>   Klasyfikacja adresów URL z pliku tekstowego (jeden URL w linii).

Wyjście JSON na stdout:
  {
    "url": "...",
    "verdict": "PHISHING" | "BENIGN" | "UNCERTAIN",
    "score": <float>,
    "threshold": <float>,
    "uncertainty_margin": <float>,
    "latency_ms": <float>
  }

Werdykt UNCERTAIN jest zwracany, gdy wynik modelu mieści się w marginesie
+/-uncertainty_margin wokół progu tau* (domyślnie 0.05).

Przykład użycia:
  python src/predict.py --url "http://secure-login.paypa1.com/verify"
  python src/predict.py --file urls.txt --model models/best_hybrid.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Upewnij się, że katalog główny projektu jest w sys.path (dla unpicklingu klas z modułu src)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np

DEFAULT_MODEL_FILE = Path("models/best_hybrid.joblib")
DEFAULT_THRESHOLD_FILE = Path("models/threshold.json")
DEFAULT_UNCERTAINTY_MARGIN = 0.05


# ---------------------------------------------------------------------------
# Ładowanie artefaktów
# ---------------------------------------------------------------------------

def load_model(model_path: Path) -> Any:
    """Wczytuje wytrenowany pipeline z pliku .joblib."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {model_path}\n"
            "Uruchom najpierw: python src/optimize_threshold.py"
        )
    return joblib.load(model_path)


def load_threshold(threshold_path: Path) -> tuple[float, str]:
    """Wczytuje próg decyzyjny tau* i metodę scoringu z threshold.json.

    Returns:
        (threshold, score_method) — próg tau* oraz metoda ('decision_function' lub 'predict_proba').
    """
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku progu: {threshold_path}\n"
            "Uruchom najpierw: python src/optimize_threshold.py"
        )
    with open(threshold_path, encoding="utf-8") as f:
        data = json.load(f)

    threshold = float(data["threshold"])
    score_method = str(data.get("score_method", "decision_function"))
    return threshold, score_method


# ---------------------------------------------------------------------------
# Inferencja
# ---------------------------------------------------------------------------

def _get_score(model: Any, url: str, score_method: str) -> float:
    """Oblicza ciągły wynik predykcji dla pojedynczego URL.

    Obsługuje decision_function (LinearSVC) i predict_proba (NB).
    """
    if score_method == "decision_function" and hasattr(model, "decision_function"):
        raw = model.decision_function([url])
        arr = np.asarray(raw).ravel()
        return float(arr[0] if arr.ndim == 1 else arr[..., -1].ravel()[0])
    if score_method == "predict_proba" and hasattr(model, "predict_proba"):
        raw = model.predict_proba([url])
        arr = np.asarray(raw)
        # Kolumna 1 = prawdopodobieństwo klasy PHISHING
        return float(arr[0, 1] if arr.ndim == 2 else arr.ravel()[0])
    # Fallback: predict() — zwraca 0 lub 1
    return float(model.predict([url])[0])


def classify_url(
    url: str,
    model: Any,
    threshold: float,
    score_method: str,
    uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> dict[str, Any]:
    """Klasyfikuje pojedynczy URL i zwraca słownik z werdyktem i metadanymi.

    Args:
        url: Adres URL do klasyfikacji.
        model: Wczytany pipeline sklearn.
        threshold: Próg decyzyjny tau*.
        score_method: Metoda scoringu ('decision_function' lub 'predict_proba').
        uncertainty_margin: Margines wokół progu dla werdyktu UNCERTAIN.

    Returns:
        Słownik z kluczami: url, verdict, score, threshold, uncertainty_margin, latency_ms.
    """
    t_start = time.perf_counter()
    score = _get_score(model, url, score_method)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    distance = score - threshold
    if abs(distance) <= uncertainty_margin:
        verdict = "UNCERTAIN"
    elif score >= threshold:
        verdict = "PHISHING"
    else:
        verdict = "BENIGN"

    return {
        "url": url,
        "verdict": verdict,
        "score": round(score, 6),
        "threshold": round(threshold, 6),
        "uncertainty_margin": uncertainty_margin,
        "latency_ms": round(latency_ms, 3),
    }


def classify_urls(
    urls: list[str],
    model: Any,
    threshold: float,
    score_method: str,
    uncertainty_margin: float = DEFAULT_UNCERTAINTY_MARGIN,
) -> list[dict[str, Any]]:
    """Klasyfikuje listę adresów URL i zwraca listę słowników z wynikami.

    Czas latency_ms obejmuje łączny czas inferencji wszystkich URL-i (batch).
    Każdy wynik zawiera własne latency_ms (czas per-URL).
    """
    results = []
    for url in urls:
        result = classify_url(url, model, threshold, score_method, uncertainty_margin)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Wczytywanie URL-i z pliku
# ---------------------------------------------------------------------------

def read_urls_from_file(path: Path) -> list[str]:
    """Wczytuje adresy URL z pliku tekstowego — jeden URL na linię.

    Ignoruje puste linie i linie zaczynające się od '#'.
    Obsługuje pliki CSV z kolumną URL (pierwsza kolumna lub kolumna 'URL'/'url').
    """
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z URL-ami: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Próba wykrycia CSV (pierwsza linia zawiera nagłówek z 'url' lub 'URL')
    if lines and "," in lines[0]:
        header = lines[0].lower()
        url_col_idx = 0
        parts = header.split(",")
        for i, col in enumerate(parts):
            if col.strip() in {"url", "urls", "address"}:
                url_col_idx = i
                break
        urls = []
        for line in lines[1:]:
            if not line.strip() or line.strip().startswith("#"):
                continue
            cols = line.split(",")
            if url_col_idx < len(cols):
                urls.append(cols[url_col_idx].strip().strip('"'))
        return urls

    # Tryb tekstowy — jeden URL na linię
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Buduje parser argumentów CLI."""
    parser = argparse.ArgumentParser(
        prog="predict",
        description=(
            "Inspekcja adresów URL — symulator bramki ESG / serwera proxy SOC.\n"
            "Klasyfikuje URL jako PHISHING / BENIGN / UNCERTAIN na podstawie\n"
            "wytrenowanego pipeline'u i optymalnego progu decyzyjnego tau*."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--url",
        type=str,
        metavar="URL",
        help="Pojedynczy adres URL do sklasyfikowania.",
    )
    input_group.add_argument(
        "--file",
        type=Path,
        metavar="PLIK",
        help="Ścieżka do pliku tekstowego z adresami URL (jeden URL na linię lub CSV).",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        metavar="MODEL",
        help=f"Ścieżka do pliku modelu .joblib (domyślnie: {DEFAULT_MODEL_FILE}).",
    )
    parser.add_argument(
        "--threshold",
        type=Path,
        default=DEFAULT_THRESHOLD_FILE,
        metavar="THRESHOLD",
        help=f"Ścieżka do pliku threshold.json (domyślnie: {DEFAULT_THRESHOLD_FILE}).",
    )
    parser.add_argument(
        "--uncertainty-margin",
        type=float,
        default=DEFAULT_UNCERTAINTY_MARGIN,
        metavar="MARGIN",
        help=(
            f"Margines wokół progu dla werdyktu UNCERTAIN (domyślnie: {DEFAULT_UNCERTAINTY_MARGIN}). "
            "URL ze score w przedziale [tau - margin, tau + margin] jest oznaczany jako UNCERTAIN."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=False,
        help="Wypisz JSON z wcięciami (czytelny dla człowieka).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punkt wejścia CLI. Zwraca kod wyjścia (0 = sukces, 1 = błąd)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Wczytaj model i próg
    try:
        model = load_model(args.model)
        threshold, score_method = load_threshold(args.threshold)
    except FileNotFoundError as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    indent = 2 if args.pretty else None

    # Tryb: pojedynczy URL
    if args.url is not None:
        result = classify_url(
            url=args.url,
            model=model,
            threshold=threshold,
            score_method=score_method,
            uncertainty_margin=args.uncertainty_margin,
        )
        print(json.dumps(result, ensure_ascii=False, indent=indent))
        return 0

    # Tryb: plik z URL-ami
    try:
        urls = read_urls_from_file(args.file)
    except FileNotFoundError as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    if not urls:
        print("Błąd: plik nie zawiera żadnych adresów URL.", file=sys.stderr)
        return 1

    results = classify_urls(
        urls=urls,
        model=model,
        threshold=threshold,
        score_method=score_method,
        uncertainty_margin=args.uncertainty_margin,
    )
    print(json.dumps(results, ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())

