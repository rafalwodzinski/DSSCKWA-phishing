"""Finalna ewaluacja modeli na zamrożonym zbiorze testowym."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.evaluate import compute_metrics
from src.train import BASELINE_MODEL_FILE, DEFAULT_TRAIN_FILE, load_model

DEFAULT_TEST_FILE = Path("data/processed/test.csv")
DEFAULT_BEST_HYBRID_MODEL_FILE = Path("models/best_hybrid.joblib")
DEFAULT_THRESHOLD_FILE = Path("models/threshold.json")
DEFAULT_RESULTS_FILE = Path("reports/results/final_test_metrics.json")
DEFAULT_FIGURES_DIR = Path("reports/figures")

BENCHMARK_SAMPLES = 1000
MAX_INFERENCE_TIME_MS = 10.0
MAX_TRAINING_TIME_SECONDS = 90.0
MIN_F1 = 0.92
MAX_FPR = 0.02
MIN_ROC_AUC = 0.95


def _load_dataset(path: Path) -> pd.DataFrame:
    """Wczytuje i waliduje CSV zawierający URL oraz binarną etykietę Label."""
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono zbioru danych: {path}")

    frame = pd.read_csv(path)
    missing_columns = {"URL", "Label"} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} nie zawiera wymaganych kolumn: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError(f"Zbiór danych jest pusty: {path}")

    frame = frame[["URL", "Label"]].dropna().copy()
    frame["URL"] = frame["URL"].astype(str)
    labels = pd.to_numeric(frame["Label"], errors="raise")
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"Etykiety w {path} muszą być binarne (0 lub 1).")
    if set(labels.unique()) != {0, 1}:
        raise ValueError(f"Zbiór {path} musi zawierać obie klasy (0 i 1).")
    frame["Label"] = labels.astype(int)
    if frame.empty:
        raise ValueError(f"Zbiór danych nie zawiera kompletnych wierszy: {path}")
    return frame.reset_index(drop=True)


def _load_threshold(path: Path) -> float:
    """Wczytuje i waliduje próg wyznaczony wcześniej na zbiorze walidacyjnym."""
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono konfiguracji progu: {path}")
    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    threshold = config.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"Plik {path} musi zawierać numeryczny klucz 'threshold'.")
    if not np.isfinite(threshold):
        raise ValueError(f"Próg w pliku {path} musi być wartością skończoną.")
    return float(threshold)


def _continuous_scores(model: Any, urls: list[str]) -> np.ndarray:
    """Zwraca wynik dla klasy phishing z predict_proba lub decision_function."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(urls))
        classes = np.asarray(model.classes_)
        positive_class = np.flatnonzero(classes == 1)
        if probabilities.ndim != 2 or len(positive_class) != 1:
            raise ValueError("Model musi zwracać prawdopodobieństwa dla klasy 1.")
        return probabilities[:, int(positive_class[0])]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(urls)).ravel()
    raise TypeError("Model musi udostępniać predict_proba lub decision_function.")


def _benchmark_inference(model: Any, urls: list[str]) -> dict[str, float | int]:
    """Mierzy czas pojedynczej predykcji, wliczając ekstrakcję cech przez pipeline."""
    durations_ms: list[float] = []
    for url in urls:
        started = time.perf_counter()
        _continuous_scores(model, [url])
        durations_ms.append((time.perf_counter() - started) * 1000.0)

    values = np.asarray(durations_ms)
    mean_ms = float(np.mean(values))
    p95_ms = float(np.percentile(values, 95))
    mean_passed = mean_ms <= MAX_INFERENCE_TIME_MS
    p95_passed = p95_ms <= MAX_INFERENCE_TIME_MS
    return {
        "sample_count": len(durations_ms),
        "mean_ms_per_url": mean_ms,
        "p95_ms_per_url": p95_ms,
        "limit_ms_per_url": MAX_INFERENCE_TIME_MS,
        "mean_passed": bool(mean_passed),
        "p95_passed": bool(p95_passed),
        "passed": bool(mean_passed and p95_passed),
    }


def _feature_importances(model: Any) -> tuple[list[str], np.ndarray]:
    """Zwraca nazwy cech hybrydowych i ważności klasyfikatora."""
    if not hasattr(model, "named_steps") or "features" not in model.named_steps:
        raise TypeError("Best Hybrid musi zawierać krok 'features'.")
    if "classifier" not in model.named_steps:
        raise TypeError("Best Hybrid musi zawierać krok 'classifier'.")

    names = np.asarray(model.named_steps["features"].get_feature_names_out(), dtype=str)
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        importances = np.abs(np.asarray(classifier.coef_)).ravel()
    elif hasattr(classifier, "feature_log_prob_"):
        log_probabilities = np.asarray(classifier.feature_log_prob_)
        positive_class = np.flatnonzero(np.asarray(classifier.classes_) == 1)
        negative_class = np.flatnonzero(np.asarray(classifier.classes_) == 0)
        if len(positive_class) != 1 or len(negative_class) != 1:
            raise ValueError("Klasyfikator hybrydowy musi obsługiwać klasy 0 i 1.")
        importances = np.abs(
            log_probabilities[int(positive_class[0])]
            - log_probabilities[int(negative_class[0])]
        )
    else:
        raise TypeError("Klasyfikator Best Hybrid nie udostępnia współczynników ważności cech.")

    if len(names) != len(importances):
        raise ValueError("Liczba nazw cech nie zgadza się z liczbą współczynników modelu.")
    return names.tolist(), importances


def _plot_confusion_matrices(
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(predictions), figsize=(10, 4))
    for axis, (name, predicted) in zip(np.atleast_1d(axes), predictions.items()):
        matrix = confusion_matrix(targets, predicted, labels=[0, 1])
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.figure.colorbar(image, ax=axis)
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Legalny (0)", "Phishing (1)"],
            yticklabels=["Legalny (0)", "Phishing (1)"],
            title=name,
            xlabel="Predykcja",
            ylabel="Rzeczywistość",
        )
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curves(
    targets: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for name, values in scores.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(targets, values)
        auc = roc_auc_score(targets, values)
        axis.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC = {auc:.4f})")

        threshold = thresholds[name]
        predicted = values >= threshold
        tn, fp, fn, tp = confusion_matrix(targets, predicted, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if fp + tn else 0.0
        tpr = tp / (tp + fn) if tp + fn else 0.0
        axis.scatter([fpr], [tpr], s=40, label=f"{name}, próg = {threshold:.4g}")

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.05),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Krzywe ROC — zbiór testowy",
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(model: Any, path: Path) -> list[dict[str, float | str]]:
    names, importances = _feature_importances(model)
    top_indices = np.argsort(importances)[-15:][::-1]
    top_features = [
        {"feature": names[int(index)], "importance": float(importances[int(index)])}
        for index in top_indices
    ]

    fig, axis = plt.subplots(figsize=(10, 7))
    labels = [str(item["feature"]) for item in reversed(top_features)]
    values = [float(item["importance"]) for item in reversed(top_features)]
    axis.barh(labels, values, color="#1f77b4")
    axis.set(title="Top 15 ważnych cech — Best Hybrid", xlabel="Ważność (wartość bezwzględna)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return top_features


def run_final_evaluation(
    test_path: Path = DEFAULT_TEST_FILE,
    train_path: Path = DEFAULT_TRAIN_FILE,
    baseline_model_path: Path = BASELINE_MODEL_FILE,
    best_hybrid_model_path: Path = DEFAULT_BEST_HYBRID_MODEL_FILE,
    threshold_path: Path = DEFAULT_THRESHOLD_FILE,
    results_path: Path = DEFAULT_RESULTS_FILE,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    benchmark_samples: int = BENCHMARK_SAMPLES,
) -> dict[str, Any]:
    """Ocenia zapisane modele na teście, tworzy wykresy i mierzy wydajność CPU."""
    if benchmark_samples <= 0:
        raise ValueError("Liczba próbek benchmarku musi być większa od zera.")

    test_data = _load_dataset(test_path)
    if len(test_data) < benchmark_samples:
        raise ValueError(
            f"Benchmark wymaga {benchmark_samples} URL-i w zbiorze testowym; "
            f"znaleziono {len(test_data)}."
        )
    train_data = _load_dataset(train_path)

    threshold = _load_threshold(threshold_path)
    baseline = load_model(baseline_model_path)
    best_hybrid = load_model(best_hybrid_model_path)
    targets = test_data["Label"].to_numpy(dtype=int)
    test_urls = test_data["URL"].tolist()
    thresholds = {"Baseline": 0.5, "Best Hybrid": threshold}
    models = {"Baseline": baseline, "Best Hybrid": best_hybrid}

    scores = {
        name: _continuous_scores(model, test_urls)
        for name, model in models.items()
    }
    predictions = {
        name: (model_scores >= thresholds[name]).astype(int)
        for name, model_scores in scores.items()
    }
    metrics = {
        name: compute_metrics(targets, model_scores, threshold=thresholds[name])
        for name, model_scores in scores.items()
    }

    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrices(
        targets,
        predictions,
        figures_dir / "confusion_matrices.png",
    )
    _plot_roc_curves(
        targets,
        scores,
        thresholds,
        figures_dir / "roc_curves.png",
    )
    feature_importance = _plot_feature_importance(
        best_hybrid,
        figures_dir / "feature_importance.png",
    )

    benchmark_urls = test_urls[:benchmark_samples]
    inference_benchmarks = {
        name: _benchmark_inference(model, benchmark_urls)
        for name, model in models.items()
    }

    retrained_hybrid = clone(best_hybrid)
    started = time.perf_counter()
    retrained_hybrid.fit(
        train_data["URL"].tolist(),
        train_data["Label"].to_numpy(dtype=int),
    )
    training_time = time.perf_counter() - started

    acceptance = {
        name: {
            "f1": bool(values["f1"] >= MIN_F1),
            "fpr": bool(values["fpr"] <= MAX_FPR),
            "roc_auc": bool(values["roc_auc"] >= MIN_ROC_AUC),
            "passed": bool(
                values["f1"] >= MIN_F1
                and values["fpr"] <= MAX_FPR
                and values["roc_auc"] >= MIN_ROC_AUC
            ),
        }
        for name, values in metrics.items()
    }
    training_benchmark = {
        "model": "Best Hybrid",
        "dataset": str(train_path),
        "sample_count": len(train_data),
        "time_seconds": float(training_time),
        "limit_seconds": MAX_TRAINING_TIME_SECONDS,
        "passed": bool(training_time <= MAX_TRAINING_TIME_SECONDS),
    }
    best_hybrid_performance_passed = bool(
        inference_benchmarks["Best Hybrid"]["passed"] and training_benchmark["passed"]
    )
    result: dict[str, Any] = {
        "evaluation": "final_test",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(test_path),
        "sample_count": len(test_data),
        "thresholds": thresholds,
        "models": metrics,
        "acceptance_criteria": {
            "minimum_f1": MIN_F1,
            "maximum_fpr": MAX_FPR,
            "minimum_roc_auc": MIN_ROC_AUC,
            "by_model": acceptance,
            "best_hybrid_passed": acceptance["Best Hybrid"]["passed"],
            "performance": {
                "inference_by_model": {
                    name: benchmark["passed"]
                    for name, benchmark in inference_benchmarks.items()
                },
                "training_best_hybrid": training_benchmark["passed"],
                "best_hybrid_passed": best_hybrid_performance_passed,
            },
        },
        "benchmarks": {
            "inference": inference_benchmarks,
            "training": training_benchmark,
        },
        "feature_importance_top_15": feature_importance,
        "figures": {
            "confusion_matrices": str(figures_dir / "confusion_matrices.png"),
            "roc_curves": str(figures_dir / "roc_curves.png"),
            "feature_importance": str(figures_dir / "feature_importance.png"),
        },
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False, allow_nan=False)
    return result


def main() -> None:
    """Uruchamia finalną ewaluację z wiersza poleceń."""
    parser = argparse.ArgumentParser(
        description="Finalna ewaluacja Baseline i Best Hybrid na zamrożonym teście."
    )
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--baseline-model", type=Path, default=BASELINE_MODEL_FILE)
    parser.add_argument("--best-hybrid-model", type=Path, default=DEFAULT_BEST_HYBRID_MODEL_FILE)
    parser.add_argument("--threshold", type=Path, default=DEFAULT_THRESHOLD_FILE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()

    result = run_final_evaluation(
        test_path=args.test,
        train_path=args.train,
        baseline_model_path=args.baseline_model,
        best_hybrid_model_path=args.best_hybrid_model,
        threshold_path=args.threshold,
        results_path=args.results,
        figures_dir=args.figures_dir,
    )
    for name, values in result["models"].items():
        print(
            f"{name}: F1={values['f1']:.4f}, FPR={values['fpr']:.4f}, "
            f"ROC-AUC={values['roc_auc']:.4f}"
        )
    print(f"Czas treningu Best Hybrid: {result['benchmarks']['training']['time_seconds']:.2f}s")
    for name, benchmark in result["benchmarks"]["inference"].items():
        print(
            f"Inferencja {name}: średnia={benchmark['mean_ms_per_url']:.3f} ms/URL, "
            f"p95={benchmark['p95_ms_per_url']:.3f} ms/URL"
        )
    print(f"Zapisano wynik: {args.results}")


if __name__ == "__main__":
    main()
