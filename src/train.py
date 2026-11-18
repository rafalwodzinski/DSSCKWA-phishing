"""Moduł trenowania modeli uczenia maszynowego dla detekcji phishingowych adresów URL.

Zgodnie z wymaganiami projektu DSSCKWA-phishing moduł obsługuje:
- Trenowanie grupy kontrolnej Baseline (`--model baseline`):
  Pipeline składający się wyłącznie z `CountVectorizer` na prostych tokenach słownych
  (`analyzer="word"`, `token_pattern=r"[a-zA-Z0-9]{2,}"`, `max_features=20000`)
  oraz klasyfikatora `MultinomialNB(alpha=1.0)`, bez analizy znakowej i bez cech leksykalnych.
- Trenowanie i 5-krotną stratyfikowaną walidację krzyżową modeli hybrydowych
  (`--model cnb` oraz `--model svm`) na połączonym wektorze znakowego TF-IDF (3-5)
  i 12 cech leksykalnych z `MaxAbsScaler` (wewnątrz Pipeline, bez przecieku między foldami).
- Porównanie Baseline vs Hybrid ComplementNB vs Hybrid LinearSVC (średnia ± odchylenie
  standardowe dla F1, FPR, ROC-AUC oraz czas treningu ≤ 90 s na CPU) i zapis zestawienia
  do `reports/results/cv_comparison.json`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import compute_metrics, save_metrics
from src.extract_features import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    LexicalExtractor,
    build_feature_pipeline,
)

RANDOM_STATE = 42
CV_SPLITS = 5
MAX_TRAINING_TIME_SECONDS = 90.0

DEFAULT_TRAIN_FILE = Path("data/processed/train.csv")
DEFAULT_VAL_FILE = Path("data/processed/val.csv")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_RESULTS_DIR = Path("reports/results")

BASELINE_MODEL_FILE = DEFAULT_MODELS_DIR / "baseline_mnb.joblib"
CNB_MODEL_FILE = DEFAULT_MODELS_DIR / "hybrid_cnb.joblib"
SVM_MODEL_FILE = DEFAULT_MODELS_DIR / "hybrid_svm.joblib"

BASELINE_RESULT_NAME = "val_baseline"
CNB_RESULT_NAME = "val_cnb"
SVM_RESULT_NAME = "val_svm"
CV_COMPARISON_FILENAME = "cv_comparison.json"

BASELINE_TOKEN_PATTERN = r"[a-zA-Z0-9]{2,}"
BASELINE_MAX_FEATURES = 20000
BASELINE_ALPHA = 1.0

CNB_ALPHA = 1.0
SVM_C = 1.0

# Współdzielony bufor dla procesów potomnych (fork) podczas równoległego CV
_SHARED_CV_URLS: pd.Series | None = None
_SHARED_CV_LABELS: pd.Series | None = None


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


def build_cnb_pipeline(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    alpha: float = CNB_ALPHA,
) -> Pipeline:
    """Buduje hybrydowy potok ComplementNB na znakowym TF-IDF (3-5) i 12 cechach leksykalnych.

    Umieszczenie `build_feature_pipeline()` wewnątrz `Pipeline` gwarantuje brak wycieku
    danych między foldami podczas walidacji krzyżowej.

    Args:
        max_features: Maksymalna liczba cech dla znakowego TF-IDF (domyślnie 30000).
        ngram_range: Zakres n-gramów znakowych (domyślnie (3, 5)).
        alpha: Parametr wygładzania dla ComplementNB (domyślnie 1.0).

    Returns:
        Skonfigurowany obiekt `sklearn.pipeline.Pipeline`.
    """
    features = build_feature_pipeline(max_features=max_features, ngram_range=ngram_range)
    classifier = ComplementNB(alpha=alpha)
    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])


def build_svm_pipeline(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    C: float = SVM_C,
    calibrated: bool = False,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Buduje hybrydowy potok LinearSVC na znakowym TF-IDF (3-5) i 12 cechach leksykalnych.

    Obsługuje zarówno szybkie wyliczanie ocen z `decision_function()`, jak i opcjonalne
    opakowanie w `CalibratedClassifierCV` dla prawdopodobieństw `predict_proba()`.

    Args:
        max_features: Maksymalna liczba cech dla znakowego TF-IDF (domyślnie 30000).
        ngram_range: Zakres n-gramów znakowych (domyślnie (3, 5)).
        C: Parametr regularyzacji dla LinearSVC (domyślnie 1.0).
        calibrated: Czy opakować LinearSVC w CalibratedClassifierCV (domyślnie False).
        random_state: Ziarno losowości (domyślnie 42).

    Returns:
        Skonfigurowany obiekt `sklearn.pipeline.Pipeline`.
    """
    features = build_feature_pipeline(max_features=max_features, ngram_range=ngram_range)
    base_svm = LinearSVC(C=C, random_state=random_state)
    classifier: Any = (
        CalibratedClassifierCV(estimator=base_svm, cv=3) if calibrated else base_svm
    )
    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])


def _predict_continuous_scores(
    pipeline: Pipeline,
    X: pd.Series | list[str] | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Wyznacza ciągłe oceny modelu (predict_proba lub decision_function) oraz domyślny próg."""
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X)
        return scores, 0.5
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        return scores, 0.0
    preds = pipeline.predict(X)
    return np.asarray(preds, dtype=float), 0.5


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


def _evaluate_single_cv_fold(args: tuple[Any, ...]) -> dict[str, Any]:
    """Funkcja robocza wykonująca pojedynczy fold walidacji krzyżowej wewnątrz Pipeline."""
    fold_idx, train_idx, val_idx, model_key, max_features, ngram_range, calibrated = args

    if _SHARED_CV_URLS is None or _SHARED_CV_LABELS is None:
        raise RuntimeError("Dane CV nie zostały zainicjalizowane przed uruchomieniem foldów.")

    X_fold_train = _SHARED_CV_URLS.iloc[train_idx]
    y_fold_train = _SHARED_CV_LABELS.iloc[train_idx]
    X_fold_val = _SHARED_CV_URLS.iloc[val_idx]
    y_fold_val = _SHARED_CV_LABELS.iloc[val_idx]

    if model_key == "baseline":
        pipe = build_baseline_pipeline(max_features=max_features)
    elif model_key == "cnb":
        pipe = build_cnb_pipeline(max_features=max_features, ngram_range=ngram_range)
    elif model_key == "svm":
        pipe = build_svm_pipeline(
            max_features=max_features,
            ngram_range=ngram_range,
            calibrated=calibrated,
        )
    else:
        raise ValueError(f"Nieznany klucz modelu w CV: {model_key}")

    t0 = time.perf_counter()
    pipe.fit(X_fold_train, y_fold_train)
    fit_time = time.perf_counter() - t0

    scores, threshold = _predict_continuous_scores(pipe, X_fold_val)
    fold_metrics = compute_metrics(y_fold_val, scores, threshold=threshold)

    return {
        "fold": int(fold_idx),
        "fit_time_sec": float(fit_time),
        "metrics": fold_metrics,
    }


def run_stratified_cv(
    urls: pd.Series,
    labels: pd.Series,
    model_key: str,
    n_splits: int = CV_SPLITS,
    max_features: int | None = None,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    calibrated: bool = False,
    random_state: int = RANDOM_STATE,
    parallel: bool = True,
) -> dict[str, Any]:
    """Przeprowadza 5-krotną stratyfikowaną walidację krzyżową wewnątrz Pipeline.

    Każdy fold dopasowuje własny `Pipeline` wyłącznie na części treningowej foldu,
    zapobiegając wyciekowi słownika TF-IDF oraz statystyk `MaxAbsScaler` między foldami.

    Args:
        urls: Seria adresów URL ze zbioru treningowego.
        labels: Seria etykiet binarnych (0/1) ze zbioru treningowego.
        model_key: Identyfikator modelu ('baseline', 'cnb', 'svm').
        n_splits: Liczba foldów (domyślnie 5).
        max_features: Maksymalna liczba cech dla wektoryzatora.
        ngram_range: Zakres n-gramów znakowych (dla modeli hybrydowych).
        calibrated: Czy użyć CalibratedClassifierCV dla SVM.
        random_state: Ziarno generatora dla StratifiedKFold (domyślnie 42).
        parallel: Czy zrównoleglić obliczenia foldów na rdzeniach CPU.

    Returns:
        Słownik ze statystykami CV (średnie, odchylenia standardowe, czasy i wyniki foldów).
    """
    global _SHARED_CV_URLS, _SHARED_CV_LABELS

    resolved_max_features = (
        max_features
        if max_features is not None
        else (BASELINE_MAX_FEATURES if model_key == "baseline" else TFIDF_MAX_FEATURES)
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    _SHARED_CV_URLS = urls.reset_index(drop=True).astype(str)
    _SHARED_CV_LABELS = labels.reset_index(drop=True).astype(int)

    tasks = [
        (
            fold_idx,
            tr_idx,
            va_idx,
            model_key,
            resolved_max_features,
            ngram_range,
            calibrated,
        )
        for fold_idx, (tr_idx, va_idx) in enumerate(
            skf.split(_SHARED_CV_URLS, _SHARED_CV_LABELS), start=1
        )
    ]

    t_start = time.perf_counter()
    use_fork_pool = (
        parallel
        and len( _SHARED_CV_URLS) >= 5000
        and hasattr(os, "fork")
        and "fork" in mp.get_all_start_methods()
    )

    try:
        if use_fork_pool:
            max_workers = min(n_splits, os.cpu_count() or 1)
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                fold_results = list(executor.map(_evaluate_single_cv_fold, tasks))
        else:
            fold_results = [_evaluate_single_cv_fold(task) for task in tasks]
    finally:
        _SHARED_CV_URLS = None
        _SHARED_CV_LABELS = None

    cv_wall_time = time.perf_counter() - t_start
    fold_results.sort(key=lambda item: item["fold"])

    metric_names = ["f1", "precision", "recall", "fpr", "fnr", "roc_auc"]
    summary_stats: dict[str, float] = {}

    for m_name in metric_names:
        values = [float(f["metrics"][m_name]) for f in fold_results]
        summary_stats[f"{m_name}_mean"] = float(np.mean(values))
        summary_stats[f"{m_name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    fit_times = [float(f["fit_time_sec"]) for f in fold_results]
    summary_stats["fold_fit_time_mean_sec"] = float(np.mean(fit_times))
    summary_stats["fold_fit_time_std_sec"] = (
        float(np.std(fit_times, ddof=1)) if len(fit_times) > 1 else 0.0
    )
    summary_stats["cv_wall_time_sec"] = float(cv_wall_time)

    return {
        "model": model_key,
        "n_splits": n_splits,
        **summary_stats,
        "folds": fold_results,
    }


def update_cv_comparison(
    model_key: str,
    display_name: str,
    cv_summary: dict[str, Any],
    full_train_fit_time_sec: float,
    val_metrics: dict[str, float] | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Aktualizuje i zapisuje zestawienie porównawcze modeli w reports/results/cv_comparison.json.

    Dodatkowo eksportuje czytelną tabelę porównawczą (Baseline vs Hybrid ComplementNB vs
    Hybrid LinearSVC) do pliku `reports/results/cv_comparison.csv`.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / CV_COMPARISON_FILENAME

    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    if "cv_config" not in data:
        data["cv_config"] = {
            "splitter": "StratifiedKFold",
            "n_splits": cv_summary.get("n_splits", CV_SPLITS),
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "max_training_time_limit_sec": MAX_TRAINING_TIME_SECONDS,
        }
    if "models" not in data or not isinstance(data["models"], dict):
        data["models"] = {}

    entry: dict[str, Any] = {
        "display_name": display_name,
        "f1_mean": round(float(cv_summary["f1_mean"]), 6),
        "f1_std": round(float(cv_summary["f1_std"]), 6),
        "f1_formatted": f"{cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}",
        "fpr_mean": round(float(cv_summary["fpr_mean"]), 6),
        "fpr_std": round(float(cv_summary["fpr_std"]), 6),
        "fpr_formatted": f"{cv_summary['fpr_mean']:.4f} ± {cv_summary['fpr_std']:.4f}",
        "roc_auc_mean": round(float(cv_summary["roc_auc_mean"]), 6),
        "roc_auc_std": round(float(cv_summary["roc_auc_std"]), 6),
        "roc_auc_formatted": f"{cv_summary['roc_auc_mean']:.4f} ± {cv_summary['roc_auc_std']:.4f}",
        "precision_mean": round(float(cv_summary["precision_mean"]), 6),
        "precision_std": round(float(cv_summary["precision_std"]), 6),
        "recall_mean": round(float(cv_summary["recall_mean"]), 6),
        "recall_std": round(float(cv_summary["recall_std"]), 6),
        "fnr_mean": round(float(cv_summary["fnr_mean"]), 6),
        "fnr_std": round(float(cv_summary["fnr_std"]), 6),
        "fold_fit_time_mean_sec": round(float(cv_summary["fold_fit_time_mean_sec"]), 3),
        "fold_fit_time_std_sec": round(float(cv_summary["fold_fit_time_std_sec"]), 3),
        "cv_wall_time_sec": round(float(cv_summary["cv_wall_time_sec"]), 3),
        "train_time_sec": round(float(full_train_fit_time_sec), 3),
        "total_time_sec": round(
            float(cv_summary["cv_wall_time_sec"] + full_train_fit_time_sec), 3
        ),
        "within_90s_limit": bool(full_train_fit_time_sec <= MAX_TRAINING_TIME_SECONDS),
        "folds": cv_summary.get("folds", []),
    }

    if val_metrics is not None:
        entry["val_metrics"] = {k: round(float(v), 6) for k, v in val_metrics.items()}

    data["models"][model_key] = entry

    # Budowa ujednoliconej tabeli porównawczej: Baseline vs Hybrid ComplementNB vs Hybrid LinearSVC
    ordered_keys = ["baseline", "cnb", "svm"]
    comparison_table: list[dict[str, Any]] = []
    for key in ordered_keys:
        if key in data["models"]:
            m_info = data["models"][key]
            comparison_table.append({
                "model_key": key,
                "model": m_info["display_name"],
                "f1_cv": m_info["f1_formatted"],
                "fpr_cv": m_info["fpr_formatted"],
                "roc_auc_cv": m_info["roc_auc_formatted"],
                "f1_mean": m_info["f1_mean"],
                "f1_std": m_info["f1_std"],
                "fpr_mean": m_info["fpr_mean"],
                "fpr_std": m_info["fpr_std"],
                "roc_auc_mean": m_info["roc_auc_mean"],
                "roc_auc_std": m_info["roc_auc_std"],
                "fold_fit_time_mean_sec": m_info["fold_fit_time_mean_sec"],
                "train_time_sec": m_info["train_time_sec"],
                "cv_wall_time_sec": m_info["cv_wall_time_sec"],
            })

    data["comparison_table"] = comparison_table

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if comparison_table:
        csv_path = results_dir / "cv_comparison.csv"
        pd.DataFrame(comparison_table).to_csv(csv_path, index=False)

    return json_path


def verify_dod_hybrid(
    train_time_sec: float,
    cv_wall_time_sec: float,
    cv_comparison_path: Path,
    model_key: str,
) -> dict[str, bool]:
    """Weryfikuje kryteria Definition of Done (DoD) dla modeli hybrydowych.

    Kryteria akceptacji:
    1. Całkowity czas treningu mieści się w limicie <= 90 s na CPU.
    2. Zapisana tabela porównawcza w `reports/results/cv_comparison.json`.
    """
    within_time_limit = bool(
        train_time_sec <= MAX_TRAINING_TIME_SECONDS
        and cv_wall_time_sec <= MAX_TRAINING_TIME_SECONDS
    )
    assert within_time_limit, (
        f"DoD 1 niespełnione: czas treningu ({train_time_sec:.2f}s) lub CV ({cv_wall_time_sec:.2f}s) "
        f"przekroczył limit {MAX_TRAINING_TIME_SECONDS}s na CPU!"
    )

    assert cv_comparison_path.exists(), (
        f"DoD 2 niespełnione: brak pliku {cv_comparison_path}"
    )

    with open(cv_comparison_path, encoding="utf-8") as f:
        cv_data = json.load(f)

    assert "models" in cv_data and model_key in cv_data["models"], (
        f"DoD 2 niespełnione: brak wpisu '{model_key}' w {cv_comparison_path}"
    )
    assert "comparison_table" in cv_data and len(cv_data["comparison_table"]) >= 1, (
        f"DoD 2 niespełnione: brak tabeli porównawczej w {cv_comparison_path}"
    )

    return {
        "training_time_under_90s": within_time_limit,
        "cv_comparison_saved": True,
    }


def _ensure_baseline_in_comparison(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    results_dir: Path,
    n_splits: int = CV_SPLITS,
    max_features: int = BASELINE_MAX_FEATURES,
) -> None:
    """Zapewnia obecność grupy kontrolnej Baseline w tabeli porównawczej cv_comparison.json."""
    json_path = results_dir / CV_COMPARISON_FILENAME
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                existing = json.load(f)
            if "models" in existing and "baseline" in existing["models"]:
                return
        except Exception:
            pass

    print("   [Info] Obliczanie statystyk 5-fold CV dla grupy kontrolnej Baseline do tabeli porównawczej...")
    X_train = train_df["URL"].astype(str)
    y_train = train_df["Label"].astype(int)

    cv_base = run_stratified_cv(
        urls=X_train,
        labels=y_train,
        model_key="baseline",
        n_splits=n_splits,
        max_features=max_features,
    )

    t0 = time.perf_counter()
    base_pipe = build_baseline_pipeline(max_features=max_features)
    base_pipe.fit(X_train, y_train)
    base_fit_time = time.perf_counter() - t0

    val_metrics = None
    val_json = results_dir / f"{BASELINE_RESULT_NAME}.json"
    if val_json.exists():
        try:
            with open(val_json, encoding="utf-8") as f:
                val_metrics = json.load(f)
        except Exception:
            val_metrics = None
    elif val_df is not None:
        scores, thresh = _predict_continuous_scores(base_pipe, val_df["URL"].astype(str))
        val_metrics = compute_metrics(val_df["Label"].astype(int), scores, threshold=thresh)

    update_cv_comparison(
        model_key="baseline",
        display_name="Baseline MultinomialNB (word tokens)",
        cv_summary=cv_base,
        full_train_fit_time_sec=base_fit_time,
        val_metrics=val_metrics,
        results_dir=results_dir,
    )


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
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    vocab_size = len(pipeline.named_steps["vectorizer"].vocabulary_)
    print(f"   Rozmiar wyuczonego słownika tokenów: {vocab_size:,} (czas: {fit_time:.2f}s)")

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


def train_hybrid(
    model_key: str,
    train_path: Path = DEFAULT_TRAIN_FILE,
    val_path: Path = DEFAULT_VAL_FILE,
    model_output_path: Path | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    n_splits: int = CV_SPLITS,
    calibrated: bool = False,
) -> tuple[Pipeline, dict[str, float], dict[str, bool]]:
    """Trenuje i waliduje krzyżowo (5-fold StratifiedKFold) model hybrydowy (cnb lub svm).

    Args:
        model_key: 'cnb' (Hybrid ComplementNB) lub 'svm' (Hybrid LinearSVC).
        train_path: Ścieżka do zbioru treningowego CSV.
        val_path: Ścieżka do zbioru walidacyjnego CSV.
        model_output_path: Ścieżka zapisu wytrenowanego modelu (.joblib).
        results_dir: Katalog zapisu wyników ewaluacji i `cv_comparison.json`.
        max_features: Maksymalna liczba cech dla znakowego TF-IDF (domyślnie 30000).
        ngram_range: Zakres n-gramów znakowych (domyślnie (3, 5)).
        n_splits: Liczba foldów w StratifiedKFold (domyślnie 5).
        calibrated: Czy opakować LinearSVC w CalibratedClassifierCV.

    Returns:
        Krotka: (wytrenowany_pipeline, slownik_metryk_val, status_dod).
    """
    normalized_key = model_key.strip().lower()
    if normalized_key == "cnb":
        display_name = "Hybrid ComplementNB (char TF-IDF + 12 lexical)"
        default_out = CNB_MODEL_FILE
        result_name = CNB_RESULT_NAME
        pipeline = build_cnb_pipeline(max_features=max_features, ngram_range=ngram_range)
    elif normalized_key == "svm":
        display_name = "Hybrid LinearSVC (char TF-IDF + 12 lexical)"
        default_out = SVM_MODEL_FILE
        result_name = SVM_RESULT_NAME
        pipeline = build_svm_pipeline(
            max_features=max_features,
            ngram_range=ngram_range,
            calibrated=calibrated,
        )
    else:
        raise ValueError(f"Nieobsługiwany model hybrydowy: {model_key}")

    out_path = model_output_path if model_output_path is not None else default_out

    print(f"=== Trening modelu hybrydowego: {display_name} ===")
    print(f"1. Wczytywanie danych treningowych ({train_path}) i walidacyjnych ({val_path})...")
    train_df = _load_dataset(train_path)
    val_df = _load_dataset(val_path) if val_path.exists() else None
    print(f"   Liczba próbek train: {len(train_df):,}")

    X_train = train_df["URL"].astype(str)
    y_train = train_df["Label"].astype(int)

    # Upewniamy się, że w tabeli porównawczej znajduje się również Baseline
    _ensure_baseline_in_comparison(
        train_df=train_df,
        val_df=val_df,
        results_dir=results_dir,
        n_splits=n_splits,
        max_features=min(BASELINE_MAX_FEATURES, max_features),
    )

    print(
        f"2. Walidacja krzyżowa StratifiedKFold(n_splits={n_splits}, shuffle=True, "
        f"random_state={RANDOM_STATE}) wewnątrz Pipeline..."
    )
    cv_summary = run_stratified_cv(
        urls=X_train,
        labels=y_train,
        model_key=normalized_key,
        n_splits=n_splits,
        max_features=max_features,
        ngram_range=ngram_range,
        calibrated=calibrated,
    )

    print(
        f"   Wyniki CV ({n_splits}-fold):\n"
        f"   - F1:      {cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}\n"
        f"   - FPR:     {cv_summary['fpr_mean']:.4f} ± {cv_summary['fpr_std']:.4f}\n"
        f"   - ROC-AUC: {cv_summary['roc_auc_mean']:.4f} ± {cv_summary['roc_auc_std']:.4f}\n"
        f"   - Czas treningu foldu: {cv_summary['fold_fit_time_mean_sec']:.2f}s ± "
        f"{cv_summary['fold_fit_time_std_sec']:.2f}s (wall time CV: {cv_summary['cv_wall_time_sec']:.2f}s)"
    )

    print("3. Trening finalnego modelu hybrydowego na pełnym zbiorze train.csv...")
    t_fit0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    full_train_time = time.perf_counter() - t_fit0
    print(f"   Czas treningu na pełnym zbiorze train: {full_train_time:.2f}s (limit DoD: <= 90.0s)")

    val_metrics: dict[str, float] = {}
    if val_df is not None:
        print("4. Ewaluacja modelu na zbiorze walidacyjnym (val.csv)...")
        X_val = val_df["URL"].astype(str)
        y_val = val_df["Label"].astype(int)
        scores, threshold = _predict_continuous_scores(pipeline, X_val)
        val_metrics = compute_metrics(y_val, scores, threshold=threshold)
        for m_name, m_val in val_metrics.items():
            print(f"   - {m_name}: {m_val:.4f}")

        val_json_path = save_metrics(
            metrics=val_metrics,
            name=result_name,
            output_dir=results_dir,
            export_csv=True,
        )
        print(f"   Zapisano metryki walidacyjne do: {val_json_path}")

    saved_model_path = save_model(pipeline, out_path)
    print(f"5. Zapisano wytrenowany model do: {saved_model_path}")

    cv_comparison_path = update_cv_comparison(
        model_key=normalized_key,
        display_name=display_name,
        cv_summary=cv_summary,
        full_train_fit_time_sec=full_train_time,
        val_metrics=val_metrics if val_metrics else None,
        results_dir=results_dir,
    )
    print(f"6. Zaktualizowano zestawienie CV w: {cv_comparison_path}")

    print("7. Weryfikacja kryteriów akceptacji (DoD)...")
    dod_status = verify_dod_hybrid(
        train_time_sec=full_train_time,
        cv_wall_time_sec=cv_summary["cv_wall_time_sec"],
        cv_comparison_path=cv_comparison_path,
        model_key=normalized_key,
    )
    for criterion, status in dod_status.items():
        print(f"   - {criterion}: {'[OK]' if status else '[BŁĄD]'}")

    return pipeline, val_metrics, dod_status


def train_model(
    model_type: str = "baseline",
    train_path: Path = DEFAULT_TRAIN_FILE,
    val_path: Path = DEFAULT_VAL_FILE,
    model_output_path: Path | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> tuple[Any, dict[str, float], dict[str, bool]]:
    """Uruchamia trening wybranego wariantu modelu ('baseline', 'cnb', 'svm')."""
    normalized_type = model_type.strip().lower()
    if normalized_type == "baseline":
        out_path = model_output_path if model_output_path is not None else BASELINE_MODEL_FILE
        return train_baseline(
            train_path=train_path,
            val_path=val_path,
            model_output_path=out_path,
            results_dir=results_dir,
        )
    if normalized_type in {"cnb", "svm"}:
        return train_hybrid(
            model_key=normalized_type,
            train_path=train_path,
            val_path=val_path,
            model_output_path=model_output_path,
            results_dir=results_dir,
        )
    raise ValueError(
        f"Nieobsługiwany typ modelu: '{model_type}'. Dostępne opcje: ['baseline', 'cnb', 'svm']."
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
        choices=["baseline", "cnb", "svm"],
        help="Wybór architektury modelu do wytrenowania: baseline, cnb lub svm.",
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
