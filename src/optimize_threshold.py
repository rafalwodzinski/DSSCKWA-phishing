"""Moduł lekkiego strojenia hiperparametrów i optymalizacji progu decyzyjnego (Punkt S3).

Zgodnie z wymaganiami projektu DSSCKWA-phishing (Issue #11) moduł odpowiada za:
- Lekkie strojenie hiperparametrów na zbiorach `train.csv` / `val.csv` (`C` dla LinearSVC,
  `alpha` dla ComplementNB, `max_features` dla TF-IDF) oraz wybór najlepszej konfiguracji
  modelu hybrydowego i zapis do `models/best_hybrid.joblib`.
- Wyznaczenie na zbiorze walidacyjnym (`val.csv`) optymalnego progu odcięcia tau*,
  który utrzymuje `FPR <= 0.018` (bufor bezpieczeństwa pod zbiór testowy gwarantujący
  `FPR <= 0.02`) i maksymalizuje `F1-Score`.
- Wygenerowanie wykresu kompromisu `F1 / Recall / FPR vs Threshold` w rozdzielczości 300 DPI
  (`reports/figures/threshold_tuning.png`).
- Zapis wyznaczonego progu oraz metadanych walidacyjnych do `models/threshold.json`
  oraz `reports/results/threshold_tuning.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import _extract_1d_scores, compute_metrics, save_metrics
from src.extract_features import (
    DEFAULT_TRAIN_FILE,
    DEFAULT_VAL_FILE,
    TFIDF_NGRAM_RANGE,
    build_feature_pipeline,
)
from src.train import RANDOM_STATE

DEFAULT_BEST_MODEL_FILE = Path("models/best_hybrid.joblib")
DEFAULT_THRESHOLD_FILE = Path("models/threshold.json")
DEFAULT_PLOT_FILE = Path("reports/figures/threshold_tuning.png")
DEFAULT_RESULTS_DIR = Path("reports/results")

DEFAULT_MAX_FPR_BUFFER = 0.018
DOD_MAX_FPR = 0.02
DOD_MIN_F1 = 0.92
DOD_MIN_ROC_AUC = 0.95

DEFAULT_MAX_FEATURES_GRID: tuple[int, ...] = (20000, 30000)
DEFAULT_CNB_ALPHA_GRID: tuple[float, ...] = (0.1, 0.5, 1.0)
DEFAULT_SVM_C_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)


def _get_continuous_scores(
    model: Pipeline | Any,
    X: Any,
) -> tuple[np.ndarray, str, float]:
    """Zwraca ciągłe wyniki modelu (decision_function lub predict_proba) oraz typ i domyślny próg."""
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return _extract_1d_scores(raw), "decision_function", 0.0
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        return _extract_1d_scores(raw), "predict_proba", 0.5
    preds = model.predict(X)
    return _extract_1d_scores(preds), "predict", 0.5


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series | list[int],
    y_score: np.ndarray | pd.Series | list[float],
    max_fpr: float = DEFAULT_MAX_FPR_BUFFER,
    num_grid_points: int = 500,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    """Wyznacza próg decyzyjny tau* maksymalizujący F1 przy zachowaniu warunku FPR <= max_fpr.

    Args:
        y_true: Rzeczywiste etykiety binarne (0 = legalny, 1 = phishing).
        y_score: Ciągłe wyniki predykcji na zbiorze walidacyjnym (`val.csv`).
        max_fpr: Maksymalny dopuszczalny FPR z buforem bezpieczeństwa (domyślnie 0.018).
        num_grid_points: Liczba punktów siatki progów do analizy i wykresu.

    Returns:
        Krotka: (optymalny_prog_tau, metryki_dla_tau, df_krzywej_progow).
    """
    targets = np.asarray(y_true).astype(int).ravel()
    scores = _extract_1d_scores(y_score).astype(float)

    if len(targets) != len(scores):
        raise ValueError(
            f"Długość y_true ({len(targets)}) nie zgadza się z długością y_score ({len(scores)})."
        )

    # Wyznaczamy zakres progów na podstawie percentyli (od 1% do 99%) oraz unikalnych kwantyli
    low_bound = float(np.percentile(scores, 1.0))
    high_bound = float(np.percentile(scores, 99.0))
    if np.isclose(low_bound, high_bound):
        low_bound = float(np.min(scores)) - 1e-3
        high_bound = float(np.max(scores)) + 1e-3

    grid_thresholds = np.linspace(low_bound, high_bound, num_grid_points)
    # Dodajemy również domyślne progi (0.0 lub 0.5), jeśli mieszczą się w zakresie
    for ref_t in (0.0, 0.5):
        if low_bound <= ref_t <= high_bound:
            grid_thresholds = np.unique(np.append(grid_thresholds, ref_t))

    # Wektoryzowane wyliczenie TP, FP, TN, FN dla każdego progu z siatki
    pos_mask = targets == 1
    neg_mask = targets == 0
    total_pos = int(np.sum(pos_mask))
    total_neg = int(np.sum(neg_mask))

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    records: list[dict[str, float]] = []
    for thresh in grid_thresholds:
        tp = int(np.sum(pos_scores >= thresh))
        fp = int(np.sum(neg_scores >= thresh))
        fn = total_pos - tp
        tn = total_neg - fp

        fpr = float(fp / total_neg) if total_neg > 0 else 0.0
        fnr = float(fn / total_pos) if total_pos > 0 else 0.0
        recall = float(tp / total_pos) if total_pos > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        records.append({
            "threshold": float(thresh),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "fnr": fnr,
        })

    curve_df = pd.DataFrame(records)

    # Filtrujemy progi spełniające warunek bezpieczeństwa FPR <= max_fpr
    feasible_df = curve_df[curve_df["fpr"] <= max_fpr]
    if not feasible_df.empty:
        # Maksymalizujemy F1; w razie remisu wybieramy wyższy Recall (i niższy FPR)
        best_row = feasible_df.sort_values(
            by=["f1", "recall", "precision"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        # Jeśli żaden punkt w siatce nie ma FPR <= max_fpr, wybieramy punkt o najniższym FPR i najwyższym F1
        best_row = curve_df.sort_values(
            by=["fpr", "f1"],
            ascending=[True, False],
        ).iloc[0]

    optimal_tau = float(best_row["threshold"])
    final_metrics = compute_metrics(targets, scores, threshold=optimal_tau)
    return optimal_tau, final_metrics, curve_df


def plot_threshold_tuning(
    curve_df: pd.DataFrame,
    optimal_tau: float,
    optimal_metrics: dict[str, float],
    max_fpr_buffer: float = DEFAULT_MAX_FPR_BUFFER,
    model_name: str = "Hybrid LinearSVC",
    save_path: Path | str = DEFAULT_PLOT_FILE,
) -> Path:
    """Generuje wykres kompromisu F1 / Recall / FPR względem progu decyzyjnego (300 DPI).

    Args:
        curve_df: DataFrame z kolumnami `threshold`, `f1`, `recall`, `fpr`, `precision`.
        optimal_tau: Wyznaczony optymalny próg decyzyjny tau*.
        optimal_metrics: Słownik metryk dla progu tau*.
        max_fpr_buffer: Docelowy limit bufora FPR (domyślnie 0.018).
        model_name: Nazwa modelu prezentowana w tytule wykresu.
        save_path: Ścieżka docelowa zapisu pliku PNG.

    Returns:
        Ścieżka do zapisanego pliku wykresu.
    """
    dest = Path(save_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]}
    )

    thresholds = curve_df["threshold"].to_numpy()
    f1_vals = curve_df["f1"].to_numpy()
    recall_vals = curve_df["recall"].to_numpy()
    precision_vals = curve_df["precision"].to_numpy()
    fpr_vals = curve_df["fpr"].to_numpy()

    # Górny panel: F1, Recall, Precision vs Threshold
    ax_top.plot(thresholds, f1_vals, label="F1-Score", color="#1f77b4", lw=2.2)
    ax_top.plot(thresholds, recall_vals, label="Recall (TPR)", color="#2ca02c", lw=1.8, linestyle="--")
    ax_top.plot(
        thresholds,
        precision_vals,
        label="Precision",
        color="#9467bd",
        lw=1.6,
        linestyle=":",
    )
    ax_top.axvline(
        optimal_tau,
        color="#d62728",
        linestyle="-.",
        lw=1.8,
        label=(
            f"Optymalny próg τ* = {optimal_tau:.4f}\n"
            f"(F1={optimal_metrics['f1']:.4f}, Recall={optimal_metrics['recall']:.4f})"
        ),
    )
    ax_top.scatter([optimal_tau], [optimal_metrics["f1"]], color="#d62728", s=55, zorder=5)
    ax_top.set_ylabel("Wartość metryki (F1 / Recall / Precision)", fontsize=10)
    ax_top.set_title(
        f"Strojenie progu decyzyjnego (τ*) – {model_name}",
        fontsize=12,
        pad=10,
    )
    ax_top.legend(loc="lower left", fontsize=8.5)
    ax_top.grid(alpha=0.3)

    # Dolny panel: FPR vs Threshold z zaznaczonym buforem FPR <= 0.018 i limitem DoD <= 0.020
    ax_bottom.plot(thresholds, fpr_vals, label="False Positive Rate (FPR)", color="#ff7f0e", lw=2.0)
    ax_bottom.axhline(
        max_fpr_buffer,
        color="#d62728",
        linestyle="--",
        lw=1.5,
        label=f"Bufor walidacyjny FPR ≤ {max_fpr_buffer:.3f}",
    )
    ax_bottom.axhline(
        DOD_MAX_FPR,
        color="#7f7f7f",
        linestyle=":",
        lw=1.5,
        label=f"Limit kontraktowy FPR ≤ {DOD_MAX_FPR:.3f}",
    )
    ax_bottom.axvline(optimal_tau, color="#d62728", linestyle="-.", lw=1.8)
    ax_bottom.scatter(
        [optimal_tau],
        [optimal_metrics["fpr"]],
        color="#d62728",
        s=55,
        zorder=5,
        label=f"FPR(τ*) = {optimal_metrics['fpr']:.4f}",
    )
    ax_bottom.set_ylim(0.0, min(0.12, max(float(np.max(fpr_vals)) * 1.05, 0.03)))
    ax_bottom.set_xlabel("Próg decyzyjny (τ)", fontsize=10)
    ax_bottom.set_ylabel("False Positive Rate (FPR)", fontsize=10)
    ax_bottom.legend(loc="upper right", fontsize=8.5)
    ax_bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dest


def tune_hyperparameters_and_threshold(
    train_path: Path = DEFAULT_TRAIN_FILE,
    val_path: Path = DEFAULT_VAL_FILE,
    best_model_path: Path = DEFAULT_BEST_MODEL_FILE,
    threshold_json_path: Path = DEFAULT_THRESHOLD_FILE,
    plot_path: Path = DEFAULT_PLOT_FILE,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    max_features_grid: tuple[int, ...] = DEFAULT_MAX_FEATURES_GRID,
    cnb_alpha_grid: tuple[float, ...] = DEFAULT_CNB_ALPHA_GRID,
    svm_c_grid: tuple[float, ...] = DEFAULT_SVM_C_GRID,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    max_fpr_buffer: float = DEFAULT_MAX_FPR_BUFFER,
) -> tuple[Pipeline, dict[str, Any], dict[str, bool]]:
    """Przeprowadza lekkie strojenie hiperparametrów na train/val i dobiera optymalny próg tau*.

    Zbiór testowy (`test.csv`) NIE jest w żadnym kroku wczytywany ani używany.

    Args:
        train_path: Ścieżka do zbioru treningowego (`train.csv`).
        val_path: Ścieżka do zbioru walidacyjnego (`val.csv`).
        best_model_path: Ścieżka zapisu najlepszego modelu (`models/best_hybrid.joblib`).
        threshold_json_path: Ścieżka zapisu konfiguracji progu (`models/threshold.json`).
        plot_path: Ścieżka zapisu wykresu strojenia progu (`reports/figures/threshold_tuning.png`).
        results_dir: Katalog zapisu wyników JSON/CSV (`reports/results`).
        max_features_grid: Siatka wartości `max_features` dla znakowego TF-IDF.
        cnb_alpha_grid: Siatka wartości `alpha` dla `ComplementNB`.
        svm_c_grid: Siatka wartości `C` dla `LinearSVC`.
        ngram_range: Zakres n-gramów znakowych (domyślnie (3, 5)).
        max_fpr_buffer: Maksymalny dopuszczalny FPR na zbiorze walidacyjnym (domyślnie 0.018).

    Returns:
        Krotka: (najlepszy_pipeline, slownik_wynikow_strojenia, status_dod).
    """
    # Ochrona przed przypadkowym przekazaniem test.csv
    if "test" in train_path.stem.lower() or "test" in val_path.stem.lower():
        raise ValueError(
            "DoD 1: Zbiór testowy (test.csv) nie może być używany podczas strojenia ani doboru progu!"
        )

    if not train_path.exists():
        raise FileNotFoundError(f"Brak pliku treningowego: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Brak pliku walidacyjnego: {val_path}")

    print("=== Lekkie strojenie hiperparametrów i optymalizacja progu (Punkt S3) ===")
    print(f"1. Wczytywanie zbiorów train ({train_path}) oraz val ({val_path})...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train_urls = train_df["URL"].astype(str)
    y_train = train_df["Label"].astype(int).to_numpy()
    X_val_urls = val_df["URL"].astype(str)
    y_val = val_df["Label"].astype(int).to_numpy()
    print(f"   Liczba próbek: train={len(train_df):,}, val={len(val_df):,}")

    candidates_log: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    best_features: FeatureUnion | None = None
    best_clf: Any = None
    best_val_scores: np.ndarray | None = None

    print("2. Przeszukiwanie siatki hiperparametrów (max_features, ComplementNB alpha, LinearSVC C)...")
    for max_feat in max_features_grid:
        t_feat0 = time.perf_counter()
        feature_union = build_feature_pipeline(max_features=max_feat, ngram_range=ngram_range)
        X_tr_mat = feature_union.fit_transform(X_train_urls)
        X_va_mat = feature_union.transform(X_val_urls)
        feat_time = time.perf_counter() - t_feat0
        print(f"   [max_features={max_feat}] Ekstrakcja cech: {feat_time:.2f}s")

        # 2a. Strojenie ComplementNB(alpha)
        for alpha in cnb_alpha_grid:
            t_cnb0 = time.perf_counter()
            cnb = ComplementNB(alpha=alpha)
            cnb.fit(X_tr_mat, y_train)
            clf_time = time.perf_counter() - t_cnb0

            scores_cnb = cnb.predict_proba(X_va_mat)[:, 1]
            default_metrics = compute_metrics(y_val, scores_cnb, threshold=0.5)
            tau_cnb, tuned_metrics, _ = find_optimal_threshold(
                y_val, scores_cnb, max_fpr=max_fpr_buffer, num_grid_points=250
            )

            cand_entry = {
                "model_family": "ComplementNB",
                "model_key": "cnb",
                "max_features": int(max_feat),
                "hyperparameters": {"alpha": float(alpha), "max_features": int(max_feat)},
                "score_method": "predict_proba",
                "default_threshold": 0.5,
                "default_val_metrics": default_metrics,
                "optimal_threshold": float(tau_cnb),
                "tuned_val_metrics": tuned_metrics,
                "fit_time_sec": round(float(feat_time + clf_time), 3),
            }
            candidates_log.append(cand_entry)
            print(
                f"      - CNB(alpha={alpha}): default F1={default_metrics['f1']:.4f} "
                f"(FPR={default_metrics['fpr']:.4f}) | tuned tau*={tau_cnb:.4f} -> "
                f"F1={tuned_metrics['f1']:.4f} (FPR={tuned_metrics['fpr']:.4f}, AUC={tuned_metrics['roc_auc']:.4f})"
            )

            if _is_better_candidate(cand_entry, best_candidate, max_fpr_buffer):
                best_candidate = cand_entry
                best_features = feature_union
                best_clf = cnb
                best_val_scores = scores_cnb

        # 2b. Strojenie LinearSVC(C)
        for c_val in svm_c_grid:
            t_svm0 = time.perf_counter()
            svm = LinearSVC(C=c_val, random_state=RANDOM_STATE)
            svm.fit(X_tr_mat, y_train)
            clf_time = time.perf_counter() - t_svm0

            scores_svm = svm.decision_function(X_va_mat)
            default_metrics = compute_metrics(y_val, scores_svm, threshold=0.0)
            tau_svm, tuned_metrics, _ = find_optimal_threshold(
                y_val, scores_svm, max_fpr=max_fpr_buffer, num_grid_points=250
            )

            cand_entry = {
                "model_family": "LinearSVC",
                "model_key": "svm",
                "max_features": int(max_feat),
                "hyperparameters": {"C": float(c_val), "max_features": int(max_feat)},
                "score_method": "decision_function",
                "default_threshold": 0.0,
                "default_val_metrics": default_metrics,
                "optimal_threshold": float(tau_svm),
                "tuned_val_metrics": tuned_metrics,
                "fit_time_sec": round(float(feat_time + clf_time), 3),
            }
            candidates_log.append(cand_entry)
            print(
                f"      - SVM(C={c_val}):     default F1={default_metrics['f1']:.4f} "
                f"(FPR={default_metrics['fpr']:.4f}) | tuned tau*={tau_svm:.4f} -> "
                f"F1={tuned_metrics['f1']:.4f} (FPR={tuned_metrics['fpr']:.4f}, AUC={tuned_metrics['roc_auc']:.4f})"
            )

            if _is_better_candidate(cand_entry, best_candidate, max_fpr_buffer):
                best_candidate = cand_entry
                best_features = feature_union
                best_clf = svm
                best_val_scores = scores_svm

    assert best_candidate is not None and best_features is not None and best_clf is not None
    assert best_val_scores is not None

    # 3. Precyzyjne wyznaczenie progu tau* (gęsta siatka 600 punktów) dla najlepszego modelu
    print("\n3. Wyznaczanie optymalnego progu tau* na gęstej siatce dla najlepszego modelu...")
    optimal_tau, optimal_val_metrics, curve_df = find_optimal_threshold(
        y_val,
        best_val_scores,
        max_fpr=max_fpr_buffer,
        num_grid_points=600,
    )
    best_candidate["optimal_threshold"] = float(optimal_tau)
    best_candidate["tuned_val_metrics"] = optimal_val_metrics

    best_pipeline = Pipeline([
        ("features", best_features),
        ("classifier", best_clf),
    ])

    best_display_name = (
        f"Hybrid {best_candidate['model_family']} "
        f"({best_candidate['hyperparameters']})"
    )
    print(f"   Wybrano najlepszy model: {best_display_name}")
    print(f"   Optymalny próg decyzyjny tau* = {optimal_tau:.6f} (bufor FPR <= {max_fpr_buffer})")
    for m_name, m_val in optimal_val_metrics.items():
        print(f"   - {m_name}: {m_val:.4f}")

    # 4. Zapis najlepszego modelu do models/best_hybrid.joblib
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, best_model_path)
    print(f"4. Zapisano najlepszy model hybrydowy do: {best_model_path}")

    # 5. Wygenerowanie wykresu kompromisu F1 / Recall / FPR vs Threshold (300 DPI)
    saved_plot = plot_threshold_tuning(
        curve_df=curve_df,
        optimal_tau=optimal_tau,
        optimal_metrics=optimal_val_metrics,
        max_fpr_buffer=max_fpr_buffer,
        model_name=f"Hybrid {best_candidate['model_family']}",
        save_path=plot_path,
    )
    print(f"5. Wygenerowano wykres strojenia progu (300 DPI): {saved_plot}")

    # 6. Zapis progu i metadanych do models/threshold.json oraz reports/results/
    threshold_payload: dict[str, Any] = {
        "threshold": round(float(optimal_tau), 6),
        "optimal_threshold": round(float(optimal_tau), 6),
        "default_threshold": float(best_candidate["default_threshold"]),
        "score_method": str(best_candidate["score_method"]),
        "model_family": str(best_candidate["model_family"]),
        "model_key": str(best_candidate["model_key"]),
        "hyperparameters": best_candidate["hyperparameters"],
        "max_fpr_buffer": float(max_fpr_buffer),
        "dod_max_fpr_limit": float(DOD_MAX_FPR),
        "val_metrics_at_optimal_threshold": {
            k: round(float(v), 6) for k, v in optimal_val_metrics.items()
        },
        "val_metrics_at_default_threshold": {
            k: round(float(v), 6) for k, v in best_candidate["default_val_metrics"].items()
        },
        "test_set_used": False,
    }

    threshold_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_payload, f, indent=2, ensure_ascii=False)
    print(f"6. Zapisano próg decyzyjny do: {threshold_json_path}")

    results_dir.mkdir(parents=True, exist_ok=True)
    tuning_report_path = results_dir / "threshold_tuning.json"
    tuning_report = {
        "best_model": threshold_payload,
        "candidates": candidates_log,
    }
    with open(tuning_report_path, "w", encoding="utf-8") as f:
        json.dump(tuning_report, f, indent=2, ensure_ascii=False)

    save_metrics(
        metrics=optimal_val_metrics,
        name="val_best_hybrid",
        output_dir=results_dir,
        export_csv=True,
    )

    # 7. Weryfikacja kryteriów akceptacji (DoD S3)
    print("7. Weryfikacja kryteriów akceptacji (DoD – Punkt S3)...")
    dod_status = verify_dod_s3(
        train_path=train_path,
        val_path=val_path,
        best_model_path=best_model_path,
        threshold_json_path=threshold_json_path,
        plot_path=saved_plot,
        val_metrics=optimal_val_metrics,
    )
    for criterion, status in dod_status.items():
        print(f"   - {criterion}: {'[OK]' if status else '[BŁĄD]'}")

    print("\nPunkt synchronizacji S3 gotowy!")
    return best_pipeline, tuning_report, dod_status


def _is_better_candidate(
    candidate: dict[str, Any],
    current_best: dict[str, Any] | None,
    max_fpr_buffer: float,
) -> bool:
    """Porównuje dwa modele kandydujące: priorytet ma spełnienie FPR <= max_fpr_buffer, a następnie F1."""
    if current_best is None:
        return True

    cand_metrics = candidate["tuned_val_metrics"]
    best_metrics = current_best["tuned_val_metrics"]

    cand_feasible = cand_metrics["fpr"] <= max_fpr_buffer
    best_feasible = best_metrics["fpr"] <= max_fpr_buffer

    if cand_feasible and not best_feasible:
        return True
    if not cand_feasible and best_feasible:
        return False

    return bool(cand_metrics["f1"] > best_metrics["f1"])


def verify_dod_s3(
    train_path: Path,
    val_path: Path,
    best_model_path: Path,
    threshold_json_path: Path,
    plot_path: Path,
    val_metrics: dict[str, float],
) -> dict[str, bool]:
    """Weryfikuje kryteria Definition of Done (DoD) dla Issue #11 i punktu synchronizacji S3.

    Kryteria akceptacji:
    1. Zbiór testowy (`test.csv`) nie został użyty podczas strojenia ani doboru progu.
    2. Na zbiorze walidacyjnym model osiąga: `F1 >= 0.92`, `FPR <= 0.02`, `ROC-AUC >= 0.95`.
    3. Pliki `models/best_hybrid.joblib` oraz `models/threshold.json` są gotowe dla A, C i D (S3).
    """
    # DoD 1: Brak użycia test.csv
    no_test_used = (
        "test" not in train_path.stem.lower()
        and "test" not in val_path.stem.lower()
    )
    assert no_test_used, "DoD 1 niespełnione: wykryto użycie zbioru testowego!"

    # DoD 2: Wymogi jakościowe na zbiorze walidacyjnym
    f1_ok = bool(val_metrics["f1"] >= DOD_MIN_F1)
    fpr_ok = bool(val_metrics["fpr"] <= DOD_MAX_FPR)
    auc_ok = bool(val_metrics["roc_auc"] >= DOD_MIN_ROC_AUC)
    assert f1_ok, f"DoD 2 niespełnione: F1 ({val_metrics['f1']:.4f}) < {DOD_MIN_F1}"
    assert fpr_ok, f"DoD 2 niespełnione: FPR ({val_metrics['fpr']:.4f}) > {DOD_MAX_FPR}"
    assert auc_ok, f"DoD 2 niespełnione: ROC-AUC ({val_metrics['roc_auc']:.4f}) < {DOD_MIN_ROC_AUC}"

    # DoD 3: Gotowość artefaktów S3 (models/best_hybrid.joblib, models/threshold.json oraz wykres)
    assert best_model_path.exists(), f"DoD 3 niespełnione: brak pliku {best_model_path}"
    assert threshold_json_path.exists(), f"DoD 3 niespełnione: brak pliku {threshold_json_path}"
    assert plot_path.exists(), f"DoD 3 niespełnione: brak wykresu {plot_path}"

    loaded_model = joblib.load(best_model_path)
    assert isinstance(loaded_model, Pipeline), "DoD 3: Wczytany model nie jest obiektem Pipeline"

    with open(threshold_json_path, encoding="utf-8") as f:
        thresh_data = json.load(f)
    assert "threshold" in thresh_data and isinstance(thresh_data["threshold"], (float, int)), (
        "DoD 3: Plik threshold.json nie zawiera klucza 'threshold'"
    )

    # Szybki test inferencji z użyciem wczytanego modelu i progu
    sample_urls = ["example.com/home", "paypal-security-verify-account.com/login"]
    scores, _, _ = _get_continuous_scores(loaded_model, sample_urls)
    preds = (scores >= float(thresh_data["threshold"])).astype(int)
    assert preds.shape == (2,), "DoD 3: Błąd kształtu predykcji z progiem tau*"

    return {
        "test_set_not_used": no_test_used,
        "val_metrics_meet_targets": bool(f1_ok and fpr_ok and auc_ok),
        "s3_artifacts_ready": True,
    }


def main() -> None:
    """Punkt wejścia CLI dla skryptu strojenia hiperparametrów i progu decyzyjnego."""
    parser = argparse.ArgumentParser(
        description="Strojenie hiperparametrów i optymalizacja progu decyzyjnego (Punkt S3)."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help="Ścieżka do pliku train.csv (domyślnie: data/processed/train.csv).",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_VAL_FILE,
        help="Ścieżka do pliku val.csv (domyślnie: data/processed/val.csv).",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_BEST_MODEL_FILE,
        help="Ścieżka zapisu najlepszego modelu (domyślnie: models/best_hybrid.joblib).",
    )
    parser.add_argument(
        "--output-threshold",
        type=Path,
        default=DEFAULT_THRESHOLD_FILE,
        help="Ścieżka zapisu progu decyzyjnego (domyślnie: models/threshold.json).",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=DEFAULT_PLOT_FILE,
        help="Ścieżka zapisu wykresu (domyślnie: reports/figures/threshold_tuning.png).",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=DEFAULT_MAX_FPR_BUFFER,
        help=f"Maksymalny dopuszczalny FPR na zbiorze walidacyjnym (domyślnie: {DEFAULT_MAX_FPR_BUFFER}).",
    )
    args = parser.parse_args()

    try:
        tune_hyperparameters_and_threshold(
            train_path=args.train,
            val_path=args.val,
            best_model_path=args.output_model,
            threshold_json_path=args.output_threshold,
            plot_path=args.output_plot,
            max_fpr_buffer=args.max_fpr,
        )
    except Exception as exc:
        print(f"\nBłąd podczas optymalizacji progu (S3): {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
