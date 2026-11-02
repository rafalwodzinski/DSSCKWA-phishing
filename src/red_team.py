"""Moduł Red Teamingu generujący złośliwe linki testowe (adversarial evaluation).

Zgodnie z wymaganiami zadania (Issue #12) implementuje 4 techniki zaciemniania (obfuskacji)
składni adresów URL:
1. Subdomain stuffing - doklejanie zaufanych marek jako subdomen
2. Kodowanie znaków (Hex / URL encoding) - zamiana wybranych znaków w ścieżce na %xx
3. Typosquatting - podmiana znaków wizualnie podobnych (m -> rn, l -> 1, o -> 0)
4. Fałszywe parametry query - dopisanie 'bezpiecznych' parametrów zapytań
Oraz rundę kombinowaną (połączenie powyższych technik).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

# Upewnij się, że katalog główny projektu jest w sys.path (dla wywołania poprzez python src/red_team.py)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

DEFAULT_INPUT_FILE = Path("data/processed/test.csv")
DEFAULT_OUTPUT_FILE = Path("data/adversarial/adversarial.csv")
DEFAULT_EVAL_RESULTS_FILE = Path("reports/results/red_team_eval.csv")
DEFAULT_EVAL_FIGURE_FILE = Path("reports/figures/red_team_recall_drop.png")
DEFAULT_SEED = 42
DEFAULT_SAMPLES_PER_TECHNIQUE = 125
DEFAULT_BASELINE_MODEL_FILE = Path("models/baseline_mnb.joblib")
DEFAULT_HYBRID_MODEL_FILE = Path("models/best_hybrid.joblib")
DEFAULT_THRESHOLD_FILE = Path("models/threshold.json")

ROUND_MAP = {
    "round_0": "Runda 0",
    "subdomain_stuffing": "Runda 1",
    "url_encoding": "Runda 2",
    "typosquatting": "Runda 3",
    "fake_query_params": "Runda 4",
    "combined": "Runda 5",
}

ROUND_ORDER = [
    "Runda 0",
    "Runda 1",
    "Runda 2",
    "Runda 3",
    "Runda 4",
    "Runda 5",
]

TECHNIQUE_ROUND_MAP = {
    "round_0": "Runda 0",
    "subdomain_stuffing": "Runda 1",
    "url_encoding": "Runda 2",
    "typosquatting": "Runda 3",
    "fake_query_params": "Runda 4",
    "combined": "Runda 5",
}

TRUSTED_BRANDS = [
    "paypal.com.secure-verify",
    "apple.com.id-security",
    "login.microsoftonline.com",
    "accounts.google.com.auth",
    "allegro.pl.weryfikacja",
    "mbank.pl.logowanie-online",
    "dhl.com.tracking-package",
    "inpost.pl.status-paczki",
    "secure.netflix.com.login",
    "meta.com.account-recovery",
]

SAFE_QUERY_PARAMS = [
    "safe=true&redirect=google.com&lang=en",
    "utm_source=trusted_partner&verified=1&security=high",
    "status=verified&auth=success&ref=official_support",
    "ssl=1&mode=secure_view&session=valid",
    "verification=completed&client=trusted&view=desktop",
]

TYPO_RULES = [
    ("m", "rn"),
    ("l", "1"),
    ("o", "0"),
    ("vv", "w"),
    ("w", "vv"),
    ("i", "1"),
]

HEX_ENCODING_MAP = {
    "/": "%2f",
    "a": "%61",
    "e": "%65",
    "i": "%69",
    "o": "%6f",
    "l": "%6c",
    "p": "%70",
    "s": "%73",
    ".": "%2e",
}


def _split_url_components(url: str) -> tuple[str, str, str, str, str, bool]:
    """Rozdziela URL na komponenty, zachowując informację o obecności prefiksu schematu."""
    has_scheme = "://" in url or url.startswith("//")
    norm_url = url if has_scheme else f"//{url}"
    parsed = urlsplit(norm_url)
    return (
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.query,
        parsed.fragment,
        has_scheme,
    )


def _rebuild_url(
    scheme: str,
    netloc: str,
    path: str,
    query: str,
    fragment: str,
    had_scheme: bool,
) -> str:
    """Odtwarza URL z komponentów, szanując pierwotny format schematu."""
    if not had_scheme:
        # Usuwamy schemat jeśli go nie było
        rebuilt = urlunsplit(("", netloc, path, query, fragment))
        if rebuilt.startswith("//"):
            return rebuilt[2:]
        return rebuilt
    return urlunsplit((scheme, netloc, path, query, fragment))


def apply_subdomain_stuffing(url: str, rng: random.Random) -> str:
    """Technika 1: Doklejanie zaufanych marek jako subdomen."""
    scheme, netloc, path, query, fragment, had_scheme = _split_url_components(url)
    brand = rng.choice(TRUSTED_BRANDS)

    if netloc:
        new_netloc = f"{brand}.{netloc}"
    else:
        # Jeśli brak wyraźnego netloc, doklejamy brand na początku
        parts = path.lstrip("/").split("/", 1)
        host_part = parts[0] if parts else "verify-portal.net"
        rest_path = "/" + parts[1] if len(parts) > 1 else ""
        new_netloc = f"{brand}.{host_part}"
        path = rest_path

    return _rebuild_url(scheme, new_netloc, path, query, fragment, had_scheme)


def apply_url_encoding(url: str, rng: random.Random) -> str:
    """Technika 2: Zamiana wybranych znaków w ścieżce na sekwencje %xx."""
    scheme, netloc, path, query, fragment, had_scheme = _split_url_components(url)

    target_chars = list(HEX_ENCODING_MAP.keys())
    # Wybieramy 2-4 znaki do podmiany
    sample_size = min(len(target_chars), rng.randint(2, 4))
    chosen_chars = rng.sample(target_chars, sample_size)

    new_path = path
    for char in chosen_chars:
        if char in new_path:
            new_path = new_path.replace(char, HEX_ENCODING_MAP[char])

    # Jeśli w ścieżce nie było żadnego z wybranych znaków, podmieńmy w query lub doklejmy zakodowany segment
    if new_path == path:
        if query:
            new_query = query
            for char in chosen_chars:
                if char in new_query:
                    new_query = new_query.replace(char, HEX_ENCODING_MAP[char])
            query = new_query
        else:
            new_path = (path.rstrip("/") + "/%2f%61%65%69%6f") if path else "/%2flogin"

    return _rebuild_url(scheme, netloc, new_path, query, fragment, had_scheme)


def apply_typosquatting(url: str, rng: random.Random) -> str:
    """Technika 3: Podmiana znaków wizualnie podobnych w słowach kluczowych (m->rn, l->1, o->0)."""
    scheme, netloc, path, query, fragment, had_scheme = _split_url_components(url)

    # Najpierw próbujemy zmodyfikować netloc
    modified_netloc = netloc
    applied = False

    # Losowa kolejność reguł
    rules = list(TYPO_RULES)
    rng.shuffle(rules)

    for src_char, dst_char in rules:
        if src_char in modified_netloc:
            # Podmieniamy pierwsze lub wszystkie wystąpienia
            modified_netloc = modified_netloc.replace(src_char, dst_char, 1)
            applied = True
            break

    # Jeśli w netloc nie znaleziono znaków do typosquattingu, modyfikujemy ścieżkę
    modified_path = path
    if not applied:
        for src_char, dst_char in rules:
            if src_char in modified_path:
                modified_path = modified_path.replace(src_char, dst_char, 1)
                applied = True
                break

    # Jeśli nadal nic nie znaleziono (np. same cyfry lub bardzo krótki string), dodajemy literówkowy segment
    if not applied:
        modified_netloc = f"rn{modified_netloc}" if modified_netloc else "1ogin-0n1ine.net"

    return _rebuild_url(
        scheme, modified_netloc, modified_path, query, fragment, had_scheme
    )


def apply_fake_query_params(url: str, rng: random.Random) -> str:
    """Technika 4: Dopisanie 'bezpiecznych' parametrów query (np. ?safe=true&redirect=google.com)."""
    scheme, netloc, path, query, fragment, had_scheme = _split_url_components(url)
    fake_params = rng.choice(SAFE_QUERY_PARAMS)

    if query:
        new_query = f"{query}&{fake_params}"
    else:
        new_query = fake_params

    return _rebuild_url(scheme, netloc, path, new_query, fragment, had_scheme)


def apply_combined_attack(url: str, rng: random.Random) -> str:
    """Runda kombinowana: sekwencyjne nałożenie 3 różnych technik na ten sam URL."""
    # Losujemy kolejność 3 technik
    funcs = [
        apply_subdomain_stuffing,
        apply_url_encoding,
        apply_typosquatting,
        apply_fake_query_params,
    ]
    chosen_funcs = rng.sample(funcs, 3)

    adversarial_url = url
    for func in chosen_funcs:
        adversarial_url = func(adversarial_url, rng)

    return adversarial_url


TECHNIQUE_MAPPING = {
    "subdomain_stuffing": apply_subdomain_stuffing,
    "url_encoding": apply_url_encoding,
    "typosquatting": apply_typosquatting,
    "fake_query_params": apply_fake_query_params,
    "combined": apply_combined_attack,
}


def generate_adversarial_dataset(
    input_path: Path = DEFAULT_INPUT_FILE,
    output_path: Path = DEFAULT_OUTPUT_FILE,
    samples_per_technique: int = DEFAULT_SAMPLES_PER_TECHNIQUE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generuje zbiór linków adversarial ze zbioru testowego.

    Args:
        input_path: Ścieżka do test.csv zawierającego kolumny URL i Label.
        output_path: Ścieżka docelowa do data/adversarial/adversarial.csv.
        samples_per_technique: Liczba próbek per technika (min. 125).
        seed: Ziarno losowości dla pełnej deterministyczności (domyślnie 42).

    Returns:
        pd.DataFrame z wygenerowanymi próbkami adversarial.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku wejściowego: {input_path}. "
            "Uruchom najpierw 'python src/prepare_data.py', aby wygenerować zbiór testowy."
        )

    df_test = pd.read_csv(input_path)
    if "URL" not in df_test.columns or "Label" not in df_test.columns:
        raise ValueError(
            f"Plik {input_path} musi zawierać kolumny 'URL' oraz 'Label'."
        )

    # Filtrujemy wyłącznie linki phishingowe (Label == 1)
    df_phishing = df_test[df_test["Label"] == 1].copy()
    if len(df_phishing) == 0:
        raise ValueError("W podanym zbiorze brak próbek phishingowych (Label == 1).")

    rng = random.Random(seed)
    total_needed = samples_per_technique * len(TECHNIQUE_MAPPING)

    # Pobieramy próbkę linków bazowych (z powtórzeniami jeśli zbiór byłby mniejszy niż total_needed)
    phishing_urls = df_phishing["URL"].tolist()
    if len(phishing_urls) >= total_needed:
        base_urls = rng.sample(phishing_urls, total_needed)
    else:
        base_urls = [rng.choice(phishing_urls) for _ in range(total_needed)]

    records: list[dict[str, str | int]] = []
    idx = 0

    for technique_name, technique_func in TECHNIQUE_MAPPING.items():
        for _ in range(samples_per_technique):
            original_url = base_urls[idx]
            idx += 1
            adv_url = technique_func(original_url, rng)
            records.append({
                "URL": adv_url,
                "Label": 1,
                "Technique": technique_name,
                "Original_URL": original_url,
            })

    adv_df = pd.DataFrame(records)

    # Zapis do pliku
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adv_df.to_csv(output_path, index=False)
    print(f"Zapisano {len(adv_df):,} linków adversarial w: {output_path}")

    return adv_df


def verify_dod(df: pd.DataFrame, min_samples: int = 500) -> dict[str, bool]:
    """Weryfikuje kryteria Definition of Done (DoD) dla Issue #12."""
    # 1. Liczba rekordów >= min_samples
    count_ok = len(df) >= min_samples

    # 2. Minimum 125 na każdą z 4 technik + runda kombinowana
    tech_counts = df["Technique"].value_counts().to_dict()
    required_techniques = [
        "subdomain_stuffing",
        "url_encoding",
        "typosquatting",
        "fake_query_params",
        "combined",
    ]
    tech_ok = all(tech_counts.get(t, 0) >= 125 for t in required_techniques)

    # 3. Poprawność składniowa URL (każdy da się sparsować przez urlsplit)
    valid_syntax = True
    for url in df["URL"]:
        if not isinstance(url, str) or len(url.strip()) == 0:
            valid_syntax = False
            break
        try:
            urlsplit(url if "://" in url else f"//{url}")
        except Exception:
            valid_syntax = False
            break

    # 4. Brak braków danych (NaN/None)
    no_nan = df["URL"].isna().sum() == 0 and df["Label"].isna().sum() == 0

    return {
        "min_500_records": count_ok,
        "min_125_per_technique_and_combined": tech_ok,
        "valid_url_syntax": valid_syntax,
        "no_nan_values": no_nan,
    }


def _as_binary_predictions(
    model_or_callable: Any,
    urls: list[str],
    threshold: float,
) -> np.ndarray:
    """Zamienia ciągłą ocenę modelu na predykcje binarne {0,1} na podstawie progu."""
    if callable(model_or_callable):
        scores = model_or_callable(urls)
    elif hasattr(model_or_callable, "decision_function"):
        scores = model_or_callable.decision_function(urls)
    elif hasattr(model_or_callable, "predict_proba"):
        probabilities = model_or_callable.predict_proba(urls)
        scores = np.asarray(probabilities)[:, 1] if np.asarray(probabilities).ndim == 2 else np.asarray(probabilities)
    elif hasattr(model_or_callable, "predict"):
        scores = model_or_callable.predict(urls)
    else:
        raise TypeError(f"Nieobsługiwany typ modelu do ewaluacji: {type(model_or_callable)}")

    scores_array = np.asarray(scores)
    if scores_array.ndim == 2 and scores_array.shape[1] == 2:
        scores_array = scores_array[:, 1]
    elif scores_array.ndim == 2 and scores_array.shape[1] == 1:
        scores_array = scores_array.ravel()

    if scores_array.ndim == 1 and scores_array.size == 0:
        return np.asarray([], dtype=int)

    if scores_array.dtype.kind in {"i", "u", "b"}:
        return scores_array.astype(int)
    return (scores_array >= threshold).astype(int)


def evaluate_model_recall_by_round(
    dataset: pd.DataFrame,
    predictions: dict[str, Any],
    round_order: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Ewaluuje Recall dla modelu na rundach 0–5 obfuskacji URL.

    Args:
        dataset: DataFrame z kolumnami URL, Label, Technique, Original_URL.
        predictions: Słownik model_name -> obiekt modelu lub callable zwracający oceny.
        round_order: Kolejność rund (np. ['round_0', 'url_encoding'] lub ['Runda 0', 'Runda 2']).
        threshold: Próg decyzyjny dla predykcji binarnej (0/1).

    Returns:
        DataFrame z kolumnami: Model, Round, Technique, Recall, Delta_Recall_pp.
    """
    if "URL" not in dataset.columns or "Label" not in dataset.columns:
        raise ValueError("Dataset musi zawierać kolumny 'URL' i 'Label'.")

    if round_order is None:
        ordered_rounds = list(ROUND_ORDER)
    else:
        ordered_rounds = list(round_order)

    round_records: list[dict[str, Any]] = []

    def resolve_round_reference(round_label: str) -> tuple[str, str | None]:
        label_lower = str(round_label).lower()
        if label_lower in {"runda 0", "round_0"}:
            return "round_0", "round_0"
        if label_lower in {"runda 1", "subdomain_stuffing"}:
            return "Runda 1", "subdomain_stuffing"
        if label_lower in {"runda 2", "url_encoding"}:
            return "Runda 2", "url_encoding"
        if label_lower in {"runda 3", "typosquatting"}:
            return "Runda 3", "typosquatting"
        if label_lower in {"runda 4", "fake_query_params"}:
            return "Runda 4", "fake_query_params"
        if label_lower in {"runda 5", "combined"}:
            return "Runda 5", "combined"
        if round_label in TECHNIQUE_ROUND_MAP.values():
            return round_label, next(
                (tech for tech, mapped in TECHNIQUE_ROUND_MAP.items() if mapped == round_label),
                None,
            )
        return str(round_label), None

    for model_name, model in predictions.items():
        baseline_recall = None
        for round_label in ordered_rounds:
            round_name, technique_name = resolve_round_reference(round_label)
            if round_name == "round_0":
                if "Original_URL" in dataset.columns:
                    subset = dataset[dataset["Original_URL"].notna()].copy()
                    urls = subset["Original_URL"].astype(str).tolist()
                    labels = subset["Label"].astype(int).tolist()
                else:
                    subset = dataset.copy()
                    urls = subset["URL"].astype(str).tolist()
                    labels = subset["Label"].astype(int).tolist()
                technique_name = "round_0"
            else:
                if technique_name is None:
                    variant = str(round_label).lower().replace(" ", "_")
                    technique_name = variant
                subset = dataset[dataset.get("Technique", "") == technique_name].copy()
                if subset.empty and technique_name == "subdomain_stuffing" and "Original_URL" in dataset.columns:
                    subset = dataset[dataset["Original_URL"].notna()].copy()
                urls = subset["URL"].astype(str).tolist()
                labels = subset["Label"].astype(int).tolist()

            if not urls:
                recall_value = 0.0
            else:
                predictions_array = _as_binary_predictions(model, urls, threshold)
                recall_value = float(recall_score(labels, predictions_array, zero_division=0.0))

            if baseline_recall is None:
                baseline_recall = recall_value
                delta_pp = 0.0
            else:
                delta_pp = (baseline_recall - recall_value) * 100.0

            round_records.append(
                {
                    "Model": model_name,
                    "Round": round_label,
                    "Technique": technique_name,
                    "Recall": round(recall_value, 6),
                    "Delta_Recall_pp": round(delta_pp, 4),
                    "Baseline_Recall": round(baseline_recall, 6),
                }
            )

    result = pd.DataFrame(round_records)
    if result.empty:
        return result

    result = result.sort_values(
        ["Model", "Round"],
        key=lambda col: col.map({value: idx for idx, value in enumerate(ordered_rounds)}),
    )
    return result.reset_index(drop=True)


def export_red_team_results(
    dataset: pd.DataFrame,
    model_predictions: dict[str, Any],
    csv_path: str | Path = DEFAULT_EVAL_RESULTS_FILE,
    figure_path: str | Path = DEFAULT_EVAL_FIGURE_FILE,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Eksportuje wynik ewaluacji red-team do CSV i rysuje wykres spadku Recall."""
    result = evaluate_model_recall_by_round(
        dataset=dataset,
        predictions=model_predictions,
        round_order=ROUND_ORDER,
        threshold=threshold,
    )

    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_target, index=False)

    fig_target = Path(figure_path)
    fig_target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(result["Model"].unique())
    x_positions = np.arange(len(ROUND_ORDER))
    bar_width = 0.8 / max(len(models), 1)

    for index, model_name in enumerate(models):
        model_rows = result[result["Model"] == model_name].set_index("Round").loc[ROUND_ORDER]
        ax.bar(
            x_positions + (index - (len(models) - 1) / 2) * bar_width,
            model_rows["Delta_Recall_pp"].to_numpy(),
            width=bar_width,
            label=model_name,
        )

    ax.set_title("Spadek Recall w rundach obfuskacji URL")
    ax.set_xlabel("Runda")
    ax.set_ylabel("Δ Recall (p.p.)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ROUND_ORDER, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_target, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return result


def evaluate_red_team_models(
    adversarial_path: str | Path = DEFAULT_OUTPUT_FILE,
    baseline_model_path: str | Path = DEFAULT_BASELINE_MODEL_FILE,
    hybrid_model_path: str | Path = DEFAULT_HYBRID_MODEL_FILE,
    threshold_path: str | Path = DEFAULT_THRESHOLD_FILE,
    results_path: str | Path = DEFAULT_EVAL_RESULTS_FILE,
    figure_path: str | Path = DEFAULT_EVAL_FIGURE_FILE,
) -> pd.DataFrame:
    """Wczytuje modele i danych adversarial, a następnie produkuje tabelę oraz wykres red-team."""
    dataset = pd.read_csv(adversarial_path)
    if "Technique" not in dataset.columns:
        raise ValueError(f"Plik {adversarial_path} nie zawiera kolumny 'Technique'.")

    if not Path(baseline_model_path).exists():
        raise FileNotFoundError(f"Nie znaleziono modelu baseline: {baseline_model_path}")
    if not Path(hybrid_model_path).exists():
        raise FileNotFoundError(f"Nie znaleziono modelu hybrydowego: {hybrid_model_path}")

    with open(threshold_path, encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])

    baseline_model = joblib.load(Path(baseline_model_path))
    hybrid_model = joblib.load(Path(hybrid_model_path))
    return export_red_team_results(
        dataset=dataset,
        model_predictions={
            "Baseline": baseline_model,
            "Hybrid": hybrid_model,
        },
        csv_path=results_path,
        figure_path=figure_path,
        threshold=threshold,
    )


def main() -> None:
    """Punkt wejścia CLI dla skryptu Red Teamingu."""
    parser = argparse.ArgumentParser(
        description="Generator linków adversarial dla Red Teamingu (Issue #12) oraz ewaluacja hipotezy issue #13"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Ścieżka do zbioru testowego (domyślnie: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Ścieżka zapisu linków adversarial (domyślnie: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--samples-per-technique",
        type=int,
        default=DEFAULT_SAMPLES_PER_TECHNIQUE,
        help=f"Liczba próbek per technika (domyślnie: {DEFAULT_SAMPLES_PER_TECHNIQUE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Ziarno losowości dla deterministyczności (domyślnie: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Po wygenerowaniu/odczycie zbioru adversarial uruchamia ewaluację modeli na rundach 0–5.",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=DEFAULT_BASELINE_MODEL_FILE,
        help=f"Ścieżka do modelu Baseline (domyślnie: {DEFAULT_BASELINE_MODEL_FILE})",
    )
    parser.add_argument(
        "--hybrid-model",
        type=Path,
        default=DEFAULT_HYBRID_MODEL_FILE,
        help=f"Ścieżka do modelu hybrydowego (domyślnie: {DEFAULT_HYBRID_MODEL_FILE})",
    )
    parser.add_argument(
        "--threshold",
        type=Path,
        default=DEFAULT_THRESHOLD_FILE,
        help=f"Ścieżka do pliku threshold.json (domyślnie: {DEFAULT_THRESHOLD_FILE})",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_EVAL_RESULTS_FILE,
        help=f"Ścieżka zapisu tabeli Recall i ΔRecall (domyślnie: {DEFAULT_EVAL_RESULTS_FILE})",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_EVAL_FIGURE_FILE,
        help=f"Ścieżka zapisu wykresu spadku Recall (domyślnie: {DEFAULT_EVAL_FIGURE_FILE})",
    )
    args = parser.parse_args()

    print("=== Generator Red Teamingu (Issue #12) ===")
    try:
        if args.evaluate:
            if not args.output.exists():
                generate_adversarial_dataset(
                    input_path=args.input,
                    output_path=args.output,
                    samples_per_technique=args.samples_per_technique,
                    seed=args.seed,
                )
            df = pd.read_csv(args.output)
            result = evaluate_red_team_models(
                adversarial_path=args.output,
                baseline_model_path=args.baseline_model,
                hybrid_model_path=args.hybrid_model,
                threshold_path=args.threshold,
                results_path=args.results_csv,
                figure_path=args.figure,
            )
            print("\nWyniki ewaluacji red-team:")
            print(result.to_string(index=False))
            print(f"\nCSV zapisano do: {args.results_csv}")
            print(f"PNG zapisano do: {args.figure}")
            return

        df = generate_adversarial_dataset(
            input_path=args.input,
            output_path=args.output,
            samples_per_technique=args.samples_per_technique,
            seed=args.seed,
        )
        print("\nWeryfikacja kryteriów akceptacji (DoD):")
        dod_res = verify_dod(df)
        for name, ok in dod_res.items():
            print(f"  - {name}: {'[OK]' if ok else '[BŁĄD]'}")

        if not all(dod_res.values()):
            print("\nOstrzeżenie: Niektóre kryteria DoD nie zostały spełnione!", file=sys.stderr)
            sys.exit(1)

        print("\nSukces: Wygenerowano deterministyczny zbiór adversarial.")
    except Exception as err:
        print(f"\nBłąd wykonania skryptu: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
