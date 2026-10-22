"""
src/prepare_data.py

Moduł przygotowania danych (punkt synchronizacji S1):
- Wczytanie surowego zbioru URL (Kaggle: taruntiwarihp/phishing-site-urls)
- Ujednolicenie kolumn i mapowanie etykiet (good -> 0, bad -> 1)
- Czyszczenie (NaN, puste stringi, ujednolicenie prefiksów, usunięcie konfliktów i duplikatów)
- Stratyfikowany podział 70/15/15 (train/val/test) z random_state=42
- Zapis zbiorów do data/processed/ oraz statystyk do reports/results/dataset_stats.json
- Weryfikacja kryteriów DoD (assert)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Stałe konfiguracyjne ──────────────────────────────────────────────────────
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RAW_DIR = Path("data/raw")
DEFAULT_RAW_FILE = RAW_DIR / "phishing_site_urls.csv"
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports/results")

# Jawne mapowanie etykiet ze zbioru Kaggle
LABEL_MAP = {
    "good": 0,
    "bad": 1,
}


def load_raw_data(path: Path) -> tuple[pd.DataFrame, int]:
    """Wczytuje surowy plik CSV i ujednolica nazwy kolumn do 'URL' i 'Label'."""
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {path}\n"
            f"Wgraj zbiór danych (np. z Kaggle taruntiwarihp/phishing-site-urls) "
            f"do katalogu: {path.parent.resolve()}"
        )

    df = pd.read_csv(path, dtype=str)
    initial_count = len(df)

    # Ujednolicenie nazw kolumn (obsługa wielkości liter i spacji)
    rename_dict: dict[str, str] = {}
    for col in df.columns:
        norm_col = col.strip().lower()
        if norm_col in {"url", "urls", "address"}:
            rename_dict[col] = "URL"
        elif norm_col in {"label", "class", "target", "type"}:
            rename_dict[col] = "Label"

    df = df.rename(columns=rename_dict)

    if "URL" not in df.columns or "Label" not in df.columns:
        raise ValueError(
            f"Plik {path} musi zawierać kolumny 'URL' i 'Label'. Dostępne: {list(df.columns)}"
        )

    return df[["URL", "Label"]], initial_count


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Czyści dane w określonym porządku:
    1. Usunięcie NaN i pustych stringów
    2. Przycięcie białych znaków
    3. Walidacja i mapowanie etykiet (good -> 0, bad -> 1)
    4. Ujednolicenie prefiksów (http://, https://, www.)
    5. Całkowite usunięcie URL-i z konfliktami etykiet
    6. Usunięcie dokładnych duplikatów URL
    """
    stats: dict[str, int] = {}
    initial_len = len(df)

    # 1. Usunięcie NaN
    df_clean = df.dropna(subset=["URL", "Label"]).copy()

    # 2. Przycięcie białych znaków i usunięcie pustych stringów
    df_clean["URL"] = df_clean["URL"].astype(str).str.strip()
    df_clean["Label"] = df_clean["Label"].astype(str).str.strip().str.lower()

    df_clean = df_clean[(df_clean["URL"] != "") & (df_clean["Label"] != "")]
    stats["removed_nan_or_empty"] = initial_len - len(df_clean)

    # 3. Walidacja etykiet i jawne mapowanie
    present_labels = set(df_clean["Label"].unique())
    invalid_labels = present_labels - set(LABEL_MAP.keys())
    if invalid_labels:
        raise ValueError(
            f"Wykryto nieznane wartości etykiet: {invalid_labels}. "
            f"Dopuszczalne wartości to wyłącznie: {list(LABEL_MAP.keys())}."
        )

    df_clean["Label"] = df_clean["Label"].map(LABEL_MAP).astype(int)

    # 4. Ujednolicenie prefiksów (http://, https://, www.)
    # Usuwamy protokół oraz 'www.', aby uniknąć przecieku danych i duplikatów
    prefix_pattern = r"^(https?://)?(www\.)?"
    df_clean["URL"] = df_clean["URL"].str.replace(prefix_pattern, "", regex=True, flags=re.IGNORECASE)

    # Weryfikacja czy po usunięciu prefiksu URL nie stał się pusty
    df_clean = df_clean[df_clean["URL"].str.strip() != ""]

    # 5. Wykrycie i całkowite usunięcie konfliktów etykiet (ten sam URL z różnymi klasami)
    label_counts_per_url = df_clean.groupby("URL")["Label"].nunique()
    conflicting_urls = label_counts_per_url[label_counts_per_url > 1].index
    stats["removed_conflicting_urls"] = len(conflicting_urls)
    count_before_conflict_removal = len(df_clean)
    df_clean = df_clean[~df_clean["URL"].isin(conflicting_urls)]
    stats["removed_conflict_rows"] = count_before_conflict_removal - len(df_clean)

    # 6. Usunięcie dokładnych duplikatów URL
    count_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    stats["removed_duplicates"] = count_before_dedup - len(df_clean)

    return df_clean, stats


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratyfikowany podział na train (70%), val (15%) i test (15%) z random_state=42.
    Wykonywany przez dwukrotne wywołanie train_test_split.
    """
    # Krok 1: Wydzielenie zbioru testowego (15%)
    train_val, test = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df["Label"],
        random_state=RANDOM_STATE,
    )

    # Krok 2: Z pozostałych 85% wydzielenie walidacyjnego (15% całości -> 0.15 / 0.85)
    relative_val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val["Label"],
        random_state=RANDOM_STATE,
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def verify_dod(
    cleaned_df: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, bool]:
    """Weryfikuje kryteria Definition of Done (DoD) za pomocą asercji."""
    total_cleaned = len(cleaned_df)

    # Kryterium 1: Łączna liczba oczyszczonych rekordów wynosi >= 50 000
    cond_size = total_cleaned >= 50_000
    assert cond_size, f"DoD 1 niespełnione: {total_cleaned} < 50 000"

    # Kryterium 2: Proporcje 70/15/15 i zgodność rozkładu klas
    p_train = len(train) / total_cleaned
    p_val = len(val) / total_cleaned
    p_test = len(test) / total_cleaned

    cond_proportions = (
        abs(p_train - TRAIN_RATIO) <= 0.01
        and abs(p_val - VAL_RATIO) <= 0.01
        and abs(p_test - TEST_RATIO) <= 0.01
    )
    assert cond_proportions, (
        f"DoD 2 (proporcje) niespełnione: train={p_train:.4f}, val={p_val:.4f}, test={p_test:.4f}"
    )

    # Zgodność udziału phishingu (klasa 1) pomiędzy zbiorami (odchylenie < 0.005)
    phish_ratio_train = train["Label"].mean()
    phish_ratio_val = val["Label"].mean()
    phish_ratio_test = test["Label"].mean()
    cond_class_balance = (
        abs(phish_ratio_train - phish_ratio_val) < 0.005
        and abs(phish_ratio_train - phish_ratio_test) < 0.005
    )
    assert cond_class_balance, (
        f"DoD 2 (rozkład klas) niespełnione: train={phish_ratio_train:.4f}, "
        f"val={phish_ratio_val:.4f}, test={phish_ratio_test:.4f}"
    )

    # Kryterium 3: Rozłączne zbiory URL (przecięcie równe dokładnie 0)
    overlap_train_val = set(train["URL"]) & set(val["URL"])
    overlap_train_test = set(train["URL"]) & set(test["URL"])
    overlap_val_test = set(val["URL"]) & set(test["URL"])

    cond_disjoint = (
        len(overlap_train_val) == 0
        and len(overlap_train_test) == 0
        and len(overlap_val_test) == 0
    )
    assert cond_disjoint, (
        f"DoD 3 (przecięcie URL) niespełnione: "
        f"train∩val={len(overlap_train_val)}, "
        f"train∩test={len(overlap_train_test)}, "
        f"val∩test={len(overlap_val_test)}"
    )

    return {
        "min_50k_records": bool(cond_size),
        "proportions_and_class_distribution": bool(cond_proportions and cond_class_balance),
        "zero_url_overlap": bool(cond_disjoint),
    }


def save_results(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    stats: dict,
) -> None:
    """Zapisuje podzielone zbiory do data/processed oraz metadane do reports/results/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    stats_file = REPORTS_DIR / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def main(raw_path: Path | None = None) -> None:
    if raw_path is None:
        raw_path = DEFAULT_RAW_FILE

    print(f"=== Przygotowanie danych (Punkt synchronizacji S1) ===")
    print(f"1. Wczytywanie surowych danych: {raw_path}")
    df_raw, raw_count = load_raw_data(raw_path)
    print(f"   Liczba surowych rekordów: {raw_count:,}")

    print("2. Czyszczenie danych...")
    df_clean, clean_stats = clean_data(df_raw)
    cleaned_count = len(df_clean)
    print(f"   Rekordy po oczyszczeniu: {cleaned_count:,}")
    print(f"   - Usunięte NaN / puste: {clean_stats['removed_nan_or_empty']:,}")
    print(f"   - Usunięte konflikty etykiet (unikalne URL): {clean_stats['removed_conflicting_urls']:,}")
    print(f"   - Usunięte duplikaty: {clean_stats['removed_duplicates']:,}")

    print("3. Stratyfikowany podział 70/15/15 (seed=42)...")
    train, val, test = split_data(df_clean)

    print("4. Weryfikacja kryteriów akceptacji (DoD)...")
    dod_results = verify_dod(df_clean, train, val, test)
    for criterion, status in dod_results.items():
        print(f"   - {criterion}: {'[SPEŁNIONE]' if status else '[NIESPEŁNIONE]'}")

    def get_split_stats(subset: pd.DataFrame, name: str) -> dict:
        total = len(subset)
        counts = subset["Label"].value_counts().to_dict()
        b_count = int(counts.get(0, 0))
        p_count = int(counts.get(1, 0))
        return {
            "name": name,
            "total": total,
            "benign": b_count,
            "phishing": p_count,
            "benign_ratio": round(b_count / total, 4) if total else 0.0,
            "phishing_ratio": round(p_count / total, 4) if total else 0.0,
            "split_ratio": round(total / cleaned_count, 4) if cleaned_count else 0.0,
        }

    dataset_stats = {
        "raw_records_count": raw_count,
        "cleaned_records_count": cleaned_count,
        "clean_details": clean_stats,
        "splits": [
            get_split_stats(train, "train"),
            get_split_stats(val, "val"),
            get_split_stats(test, "test"),
        ],
        "dod_verified": dod_results,
    }

    print("5. Zapisywanie przetworzonych zbiorów do data/processed/...")
    save_results(train, val, test, dataset_stats)
    print(f"   Zapisano train.csv, val.csv, test.csv w {PROCESSED_DIR}/")
    print(f"   Zapisano statystyki w {REPORTS_DIR / 'dataset_stats.json'}")
    print("Zakończono sukcesem.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przygotowanie i podział zbioru URL (S1)")
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Ścieżka do surowego pliku CSV (domyślnie data/raw/phishing_site_urls.csv)",
    )
    args = parser.parse_args()

    # Jeśli nie podano argumentu, a domyślny plik nie istnieje, sprawdź czy jest jakikolwiek CSV w data/raw/
    chosen_path = args.raw
    if chosen_path is None:
        if DEFAULT_RAW_FILE.exists():
            chosen_path = DEFAULT_RAW_FILE
        else:
            csv_candidates = list(RAW_DIR.glob("*.csv"))
            if csv_candidates:
                chosen_path = csv_candidates[0]
            else:
                chosen_path = DEFAULT_RAW_FILE

    try:
        main(chosen_path)
    except Exception as e:
        print(f"\nBłąd wykonania: {e}", file=sys.stderr)
        sys.exit(1)
