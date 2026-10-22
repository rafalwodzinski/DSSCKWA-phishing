"""Moduł ujednoliconej ewaluacji modeli uczenia maszynowego.

Zgodnie z wymaganiami projektu DSSCKWA-phishing moduł dostarcza:
- Obliczanie kluczowych metryk klasyfikacji: F1, Precision, Recall, FPR, FNR, ROC-AUC.
- Obsługę predykcji zarówno z predict_proba(), jak i decision_function().
- Generowanie i zapis wykresów (macierz pomyłek, krzywa ROC) w 300 DPI.
- Eksport wyników do plików JSON i CSV w katalogu reports/results/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _extract_1d_scores(
    y_score: np.ndarray | pd.Series | list[float] | list[int],
) -> np.ndarray:
    """Konwertuje i normalizuje wektor ocen/prawdopodobieństw do formatu 1D numpy.ndarray.

    Obsługuje m.in. wyjścia 2D z `predict_proba` (wybiera kolumnę dla klasy pozytywnej 1).
    """
    scores = np.asarray(y_score)
    if scores.ndim == 2:
        if scores.shape[1] == 2:
            scores = scores[:, 1]
        elif scores.shape[1] == 1:
            scores = scores.ravel()
        else:
            raise ValueError(
                f"Oczekiwano wektora 1D lub 2D o maksymalnie 2 kolumnach, otrzymano shape={scores.shape}"
            )
    elif scores.ndim != 1:
        raise ValueError(f"Nieprawidłowy wymiar wektora ocen: ndim={scores.ndim}")
    return scores


def compute_metrics(
    y_true: np.ndarray | pd.Series | list[int],
    y_score: np.ndarray | pd.Series | list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Oblicza ujednolicony zestaw metryk ewaluacyjnych.

    Obsługuje zarówno prawdopodobieństwa z `predict_proba()`, jak i wartości
    z `decision_function()`.

    Metryki:
    - F1: średnia harmoniczna Precision i Recall
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - FPR: False Positive Rate = FP / (FP + TN)
    - FNR: False Negative Rate = FN / (TP + FN)
    - ROC-AUC: pole pod krzywą ROC wyliczone na podstawie wartości ciągłych y_score

    Args:
        y_true: Rzeczywiste etykiety binarne (0 = legalny, 1 = phishing).
        y_score: Wyniki predykcji (prawdopodobieństwa lub wartości decision_function).
        threshold: Próg odcięcia dla klasyfikacji binarnej (domyślnie 0.5).

    Returns:
        Słownik zawierający wyliczone wartości metryk jako float:
        {"f1", "precision", "recall", "fpr", "fnr", "roc_auc"}.
    """
    targets = np.asarray(y_true).astype(int).ravel()
    scores = _extract_1d_scores(y_score)

    if len(targets) != len(scores):
        raise ValueError(
            f"Długość y_true ({len(targets)}) nie zgadza się z długością y_score ({len(scores)})."
        )

    # Predykcja binarna na podstawie progu
    y_pred = (scores >= threshold).astype(int)

    # Macierz pomyłek dla etykiet [0, 1]
    cm = confusion_matrix(targets, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # FPR = FP / (FP + TN)
    actual_negatives = fp + tn
    fpr = float(fp / actual_negatives) if actual_negatives > 0 else 0.0

    # FNR = FN / (TP + FN)
    actual_positives = tp + fn
    fnr = float(fn / actual_positives) if actual_positives > 0 else 0.0

    precision = float(precision_score(targets, y_pred, zero_division=0.0))
    recall = float(recall_score(targets, y_pred, zero_division=0.0))
    f1 = float(f1_score(targets, y_pred, zero_division=0.0))

    try:
        roc_auc = float(roc_auc_score(targets, scores))
    except ValueError:
        # Może wystąpić, gdy w zbiorze testowym obecna jest tylko jedna klasa
        roc_auc = float("nan")

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
    }


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series | list[int],
    y_pred_or_score: np.ndarray | pd.Series | list[float],
    model_name: str = "model",
    threshold: float = 0.5,
    save_path: str | Path | None = None,
    normalize: str | None = None,
) -> Path:
    """Rysuje macierz pomyłek i zapisuje ją w rozdzielczości 300 DPI.

    Args:
        y_true: Prawdziwe etykiety (0 = legalny, 1 = phishing).
        y_pred_or_score: Etykiety binarne lub ciągłe wyniki decyzyjne/prawdopodobieństwa.
        model_name: Nazwa modelu używana w tytule i domyślnej nazwie pliku.
        threshold: Próg decyzyjny (używany jeśli podano oceny ciągłe).
        save_path: Opcjonalna ścieżka zapisu. Domyślnie `reports/figures/confusion_matrix_<model_name>.png`.
        normalize: Normalizacja macierzy ('true', 'pred', 'all' lub None).

    Returns:
        Ścieżka do zapisanego pliku wykresu.
    """
    targets = np.asarray(y_true).astype(int).ravel()
    raw_scores = _extract_1d_scores(y_pred_or_score)

    # Jeśli wartości nie są czysto binarne {0, 1}, zastosuj progowanie
    unique_vals = set(np.unique(raw_scores))
    if not unique_vals.issubset({0, 1}):
        y_pred = (raw_scores >= threshold).astype(int)
    else:
        y_pred = raw_scores.astype(int)

    cm = confusion_matrix(targets, y_pred, labels=[0, 1], normalize=normalize)

    fig, ax = plt.subplots(figsize=(6, 5))
    fmt = ".2%" if normalize is not None else "d"

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        xticklabels=["Legalny (0)", "Phishing (1)"],
        yticklabels=["Legalny (0)", "Phishing (1)"],
        ax=ax,
    )
    ax.set_title(f"Macierz pomyłek – {model_name}", fontsize=12, pad=12)
    ax.set_xlabel("Predykcja", fontsize=10)
    ax.set_ylabel("Rzeczywistość", fontsize=10)

    if save_path is None:
        save_path = Path("reports/figures") / f"confusion_matrix_{model_name}.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_roc_curve(
    y_true: np.ndarray | pd.Series | list[int],
    y_score: np.ndarray | pd.Series | list[float],
    model_name: str = "model",
    threshold: float | None = 0.5,
    save_path: str | Path | None = None,
) -> Path:
    """Rysuje krzywą ROC z zaznaczonym progiem odcięcia i zapisuje w 300 DPI.

    Args:
        y_true: Rzeczywiste etykiety (0 = legalny, 1 = phishing).
        y_score: Wartości ciągłe prawdopodobieństw lub decision_function.
        model_name: Nazwa modelu do tytułu i nazwy pliku.
        threshold: Opcjonalny próg odcięcia do zaznaczenia na krzywej ROC.
        save_path: Opcjonalna ścieżka zapisu (domyślnie `reports/figures/roc_curve_<model_name>.png`).

    Returns:
        Ścieżka do zapisanego pliku wykresu.
    """
    targets = np.asarray(y_true).astype(int).ravel()
    scores = _extract_1d_scores(y_score)

    fpr_arr, tpr_arr, thresholds = roc_curve(targets, scores)
    try:
        auc_score = roc_auc_score(targets, scores)
        label_text = f"{model_name} (ROC-AUC = {auc_score:.4f})"
    except ValueError:
        auc_score = float("nan")
        label_text = f"{model_name} (ROC-AUC = N/A)"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, color="#1f77b4", lw=2, label=label_text)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Losowy (AUC = 0.5000)")

    # Oznaczenie punktu roboczego dla zadanego progu
    if threshold is not None and len(thresholds) > 0:
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        thresh_fpr = fpr_arr[idx]
        thresh_tpr = tpr_arr[idx]
        ax.scatter(
            [thresh_fpr],
            [thresh_tpr],
            color="#d62728",
            s=60,
            zorder=5,
            label=f"Próg {threshold:.2f} (FPR={thresh_fpr:.3f}, Recall={thresh_tpr:.3f})",
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall / TPR)", fontsize=10)
    ax.set_title(f"Krzywa ROC – {model_name}", fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    if save_path is None:
        save_path = Path("reports/figures") / f"roc_curve_{model_name}.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def save_metrics(
    metrics: Mapping[str, Any],
    name: str,
    output_dir: str | Path = "reports/results",
    export_csv: bool = True,
) -> Path:
    """Zapisuje słownik metryk do pliku JSON oraz opcjonalnie do zbiorczego CSV.

    Zgodnie z wymaganiami zadania zapisuje do:
    - `reports/results/<nazwa>.json`
    Oraz zgodnie z kontraktem technicznym w README.md:
    - `reports/results/summary_metrics.csv` (dopisywanie/aktualizacja wiersza).

    Args:
        metrics: Słownik zawierający metryki (np. z compute_metrics).
        name: Nazwa modelu/eksperymentu identyfikująca plik.
        output_dir: Ścieżka katalogu wyników (domyślnie `reports/results`).
        export_csv: Czy dołączyć/zaktualizować wiersz w `summary_metrics.csv`.

    Returns:
        Ścieżka do utworzonego pliku JSON.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{name}.json"
    serializable_metrics = {
        k: (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v)
        for k, v in metrics.items()
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

    if export_csv:
        csv_path = out_dir / "summary_metrics.csv"
        row_data = {"model": name, **serializable_metrics}
        new_row_df = pd.DataFrame([row_data])

        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                # Aktualizacja jeśli model już istnieje lub dodanie nowego
                if "model" in existing_df.columns and name in existing_df["model"].values:
                    existing_df = existing_df[existing_df["model"] != name]
                combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            except Exception:
                combined_df = new_row_df
        else:
            combined_df = new_row_df

        combined_df.to_csv(csv_path, index=False)

    return json_path
