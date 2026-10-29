"""Moduł trenowania modeli uczenia maszynowego dla detekcji phishingowych adresów URL.

Zgodnie z wymaganiami projektu DSSCKWA-phishing moduł obsługuje:
- Trenowanie grupy kontrolnej Baseline (`--model baseline`):
  Pipeline składający się wyłącznie z `CountVectorizer` na prostych tokenach słownych
  (`analyzer="word"`, `token_pattern=r"[a-zA-Z0-9]{2,}"`, `max_features=20000`)
  oraz klasyfikatora `MultinomialNB(alpha=1.0)`, bez analizy znakowej i bez cech leksykalnych.
- Ewaluację wytrenowanego modelu na zbiorze walidacyjnym (`val.csv`) przy użyciu `src/evaluate.py`.
- Zapis modelu do `models/baseline_mnb.joblib` oraz wyników do `reports/results/val_baseline.json`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from src.evaluate import compute_metrics, save_metrics
from src.extract_features import LexicalExtractor

DEFAULT_TRAIN_FILE = Path("data/processed/train.csv")
DEFAULT_VAL_FILE = Path("data/processed/val.csv")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_RESULTS_DIR = Path("reports/results")

BASELINE_MODEL_FILE = DEFAULT_MODELS_DIR / "baseline_mnb.joblib"
BASELINE_RESULT_NAME = "val_baseline"

BASELINE_TOKEN_PATTERN = r"[a-zA-Z0-9]{2,}"
BASELINE_MAX_FEATURES = 20000
BASELINE_ALPHA = 1.0


def build_baseline_pipeline(
    max_features: int = BASELINE_MAX_FEATURES,
    token_pattern: str = BASELINE_TOKEN_PATTERN,
    alpha: float = BASELINE_ALPHA,
) -> Pipeline:
    """Buduje potok grupy kontrolnej Baseline: CountVectorizer (tokeny słowne) + MultinomialNB.

    Zgodnie ze specyfikacją model Baseline opiera się wyłącznie na prostych tokenach
    słownych (bez n-gramów znakowych i bez 12 cech leksykalnych).

    Args:
        max_features: Maksymalna liczba najczęstszych tokenów słownych (domyślnie 20000).
        token_pattern: Wyrażenie regularne definiujące token słowny (domyślnie r"[a-zA-Z0-9]{2,}").
        alpha: Parametr wygładzania Laplace'a/Lidstone'a dla MultinomialNB (domyślnie 1.0).

    Returns:
        Skonfigurowany obiekt `sklearn.pipeline.Pipeline`.
    """
    vectorizer = CountVectorizer(
        analyzer="word",
        token_pattern=token_pattern,
        max_features=max_features,
    )
    classifier = MultinomialNB(alpha=alpha)

    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])


def save_model(model: Pipeline, path: Path | str) -> Path:
    """Serializuje wytrenowany model/pipeline do pliku .joblib."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    return dest


def load_model(path: Path | str) -> Pipeline:
    """Wczytuje zapisany model/pipeline z pliku .joblib."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku modelu: {source}")
    return joblib.load(source)


def verify_dod_baseline(model_path: Path) -> dict[str, bool]:
    """Weryfikuje kryteria Definition of Done (DoD) dla grupy kontrolnej Baseline.

    Kryteria akceptacji:
    1. Model Baseline nie korzysta z n-gramów znakowych ani cech leksykalnych.
    2. Zapisany plik `models/baseline_mnb.joblib` jest gotowy do testów Red Teamingu
       (poprawnie wczytuje się i wykonuje predykcje `predict` oraz `predict_proba`
       bezpośrednio na surowych adresach URL).
    """
    assert model_path.exists(), f"DoD 2: Plik modelu nie istnieje: {model_path}"

    loaded_model = load_model(model_path)
    assert isinstance(loaded_model, Pipeline), (
        f"DoD 2: Wczytany model nie jest obiektem Pipeline ({type(loaded_model)})"
    )

    # DoD 1: Weryfikacja braku n-gramów znakowych oraz braku cech leksykalnych
    uses_char_ngrams = False
    uses_lexical_features = False

    for _, step in loaded_model.steps:
        if isinstance(step, (LexicalExtractor, FeatureUnion)):
            uses_lexical_features = True
        if isinstance(step, (CountVectorizer, TfidfVectorizer)):
            if step.analyzer != "word":
                uses_char_ngrams = True
            if getattr(step, "ngram_range", (1, 1)) != (1, 1):
                uses_char_ngrams = True

    assert not uses_char_ngrams, "DoD 1: Model Baseline nie może korzystać z n-gramów znakowych!"
    assert not uses_lexical_features, "DoD 1: Model Baseline nie może korzystać z cech leksykalnych!"

    vec = loaded_model.named_steps.get("vectorizer")
    clf = loaded_model.named_steps.get("classifier")
    assert isinstance(vec, CountVectorizer), "DoD 1: Krok wektoryzacji musi być CountVectorizer"
    assert vec.analyzer == "word", f"DoD 1: Oczekiwano analyzer='word', otrzymano {vec.analyzer}"
    assert vec.token_pattern == BASELINE_TOKEN_PATTERN, (
        f"DoD 1: Nieprawidłowy token_pattern: {vec.token_pattern}"
    )
    assert isinstance(clf, MultinomialNB), "DoD 1: Klasyfikator musi być MultinomialNB"

    # DoD 2: Gotowość do testów Red Teamingu (predykcja na liście URL-i)
    sample_red_team_urls = [
        "example.com/login",
        "secure-paypal-verification-update.com/account/signin",
    ]
    preds = loaded_model.predict(sample_red_team_urls)
    probas = loaded_model.predict_proba(sample_red_team_urls)

    assert preds.shape == (2,), f"DoD 2: Nieprawidłowy kształt predykcji: {preds.shape}"
    assert probas.shape == (2, 2), f"DoD 2: Nieprawidłowy kształt predict_proba: {probas.shape}"

    return {
        "no_char_ngrams_or_lexical": not (uses_char_ngrams or uses_lexical_features),
        "model_ready_for_red_teaming": True,
    }


def _load_dataset(path: Path) -> pd.DataFrame:
    """Wczytuje zbiór danych CSV i weryfikuje obecność kolumn URL oraz Label."""
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku danych: {path}")
    df = pd.read_csv(path)
    if "URL" not in df.columns or "Label" not in df.columns:
        raise ValueError(
            f"Plik {path} musi zawierać kolumny 'URL' i 'Label'. Znaleziono: {list(df.columns)}"
        )
    return df


def train_baseline(
    train_path: Path = DEFAULT_TRAIN_FILE,
    val_path: Path = DEFAULT_VAL_FILE,
    model_output_path: Path = BASELINE_MODEL_FILE,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    max_features: int = BASELINE_MAX_FEATURES,
) -> tuple[Pipeline, dict[str, float], dict[str, bool]]:
    """Trenuje model grupy kontrolnej Baseline na train.csv i ewaluuje na val.csv.

    Args:
        train_path: Ścieżka do zbioru treningowego CSV.
        val_path: Ścieżka do zbioru walidacyjnego CSV.
        model_output_path: Ścieżka zapisu wytrenowanego modelu (.joblib).
        results_dir: Katalog zapisu wyników ewaluacji.
        max_features: Maksymalna liczba tokenów w słowniku CountVectorizer.

    Returns:
        Krotka: (wytrenowany_pipeline, slownik_metryk_val, status_dod).
    """
    print("=== Trening modelu: Baseline (MultinomialNB na tokenach słownych) ===")
    print(f"1. Wczytywanie danych treningowych ({train_path}) i walidacyjnych ({val_path})...")
    train_df = _load_dataset(train_path)
    val_df = _load_dataset(val_path)
    print(f"   Liczba próbek: train={len(train_df):,}, val={len(val_df):,}")

    print(
        f"2. Budowanie potoku: CountVectorizer(analyzer='word', "
        f"token_pattern=r'{BASELINE_TOKEN_PATTERN}', max_features={max_features}) "
        f"+ MultinomialNB(alpha={BASELINE_ALPHA})..."
    )
    pipeline = build_baseline_pipeline(max_features=max_features)

    print("3. Dopasowywanie modelu (fit) wyłącznie na zbiorze treningowym...")
    X_train = train_df["URL"].astype(str)
    y_train = train_df["Label"].astype(int)
    pipeline.fit(X_train, y_train)

    vocab_size = len(pipeline.named_steps["vectorizer"].vocabulary_)
    print(f"   Rozmiar wyuczonego słownika tokenów: {vocab_size:,}")

    print("4. Ewaluacja modelu na zbiorze walidacyjnym za pomocą src/evaluate.py...")
    X_val = val_df["URL"].astype(str)
    y_val = val_df["Label"].astype(int)
    y_val_proba = pipeline.predict_proba(X_val)

    metrics = compute_metrics(y_val, y_val_proba)
    for metric_name, metric_val in metrics.items():
        print(f"   - {metric_name}: {metric_val:.4f}")

    json_path = save_metrics(
        metrics=metrics,
        name=BASELINE_RESULT_NAME,
        output_dir=results_dir,
        export_csv=True,
    )
    print(f"5. Zapisano wyniki walidacyjne do: {json_path}")

    saved_model_path = save_model(pipeline, model_output_path)
    print(f"6. Zapisano wytrenowany model do: {saved_model_path}")

    print("7. Weryfikacja kryteriów akceptacji (DoD)...")
    dod_status = verify_dod_baseline(saved_model_path)
    for criterion, status in dod_status.items():
        print(f"   - {criterion}: {'[OK]' if status else '[BŁĄD]'}")

    print("\nTrening grupy kontrolnej Baseline zakończony pomyślnie!")
    return pipeline, metrics, dod_status


def train_model(
    model_type: str = "baseline",
    train_path: Path = DEFAULT_TRAIN_FILE,
    val_path: Path = DEFAULT_VAL_FILE,
    model_output_path: Path | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> tuple[Any, dict[str, float], dict[str, bool]]:
    """Uruchamia trening wybranego wariantu modelu."""
    normalized_type = model_type.strip().lower()
    if normalized_type == "baseline":
        out_path = model_output_path if model_output_path is not None else BASELINE_MODEL_FILE
        return train_baseline(
            train_path=train_path,
            val_path=val_path,
            model_output_path=out_path,
            results_dir=results_dir,
        )
    raise ValueError(
        f"Nieobsługiwany typ modelu: '{model_type}'. Dostępne opcje: ['baseline']."
    )


def main() -> None:
    """Punkt wejścia CLI dla skryptu trenowania modeli."""
    parser = argparse.ArgumentParser(
        description="Trenowanie i walidacja modeli klasyfikacji phishingowych URL."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline"],
        help="Wybór architektury modelu do wytrenowania (domyślnie: baseline).",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help="Ścieżka do pliku treningowego CSV (domyślnie: data/processed/train.csv).",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_VAL_FILE,
        help="Ścieżka do pliku walidacyjnego CSV (domyślnie: data/processed/val.csv).",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=None,
        help="Opcjonalna ścieżka zapisu modelu .joblib.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Katalog zapisu metryk (domyślnie: reports/results).",
    )
    args = parser.parse_args()

    try:
        train_model(
            model_type=args.model,
            train_path=args.train,
            val_path=args.val,
            model_output_path=args.output_model,
            results_dir=args.results_dir,
        )
    except Exception as exc:
        print(f"\nBłąd podczas trenowania modelu: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

