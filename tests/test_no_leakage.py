"""
tests/test_no_leakage.py

Niezależna weryfikacja braku wycieku danych (Data Leakage) pomiędzy zbiorami
train / val / test oraz poprawności zakresów fit() w pipeline'ach.

Testy obejmują:
  1. Rozłączność URL-i pomiędzy wszystkimi parami zbiorów (unit + integracyjny).
  2. Gwarancję, że TfidfVectorizer.vocabulary_ oraz MaxAbsScaler zostały
     dopasowane (fit) wyłącznie na zbiorze train.
  3. Stabilność podziału przy zadanym random_state — deterministyczny podział.
  4. Brak „information leak" przez statystyki skalera (MaxAbsScaler dopasowany
     na train NIE widzi val/test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

from src.extract_features import build_feature_pipeline
from src.prepare_data import clean_data, split_data


# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------

def _make_clean_df(n: int = 1_000, seed: int = 42) -> pd.DataFrame:
    """Zwraca oczyszczony DataFrame z n URL-ami o unikalnych adresach."""
    rng = np.random.default_rng(seed)
    suffixes = rng.choice(["com", "org", "net", "pl", "io"], size=n)
    urls = [f"site{i}.example{i % 50}.{suffixes[i]}/path?q={i}" for i in range(n)]
    labels = ([0] * (n // 2)) + ([1] * (n - n // 2))
    return pd.DataFrame({"URL": urls, "Label": labels})


@pytest.fixture(scope="module")
def split_dfs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Podzielony (train, val, test) moduł-level fixture."""
    df = _make_clean_df(1_000)
    return split_data(df)


# ---------------------------------------------------------------------------
# 1. Rozłączność URL-i
# ---------------------------------------------------------------------------

class TestUrlDisjointness:
    """Testy sprawdzające brak współdzielenia URL-i między podzbiorami."""

    def test_train_val_disjoint(self, split_dfs):
        train, val, _ = split_dfs
        overlap = set(train["URL"]) & set(val["URL"])
        assert len(overlap) == 0, (
            f"Wyciek danych: {len(overlap)} URL-i wspólnych dla train i val: "
            f"{list(overlap)[:5]}..."
        )

    def test_train_test_disjoint(self, split_dfs):
        train, _, test = split_dfs
        overlap = set(train["URL"]) & set(test["URL"])
        assert len(overlap) == 0, (
            f"Wyciek danych: {len(overlap)} URL-i wspólnych dla train i test: "
            f"{list(overlap)[:5]}..."
        )

    def test_val_test_disjoint(self, split_dfs):
        _, val, test = split_dfs
        overlap = set(val["URL"]) & set(test["URL"])
        assert len(overlap) == 0, (
            f"Wyciek danych: {len(overlap)} URL-i wspólnych dla val i test: "
            f"{list(overlap)[:5]}..."
        )

    def test_all_urls_preserved(self, split_dfs):
        """Suma zbiorów == pełny zbiór wejściowy (żadne URL nie zaginęło)."""
        train, val, test = split_dfs
        all_urls = set(train["URL"]) | set(val["URL"]) | set(test["URL"])
        total = len(train) + len(val) + len(test)
        assert len(all_urls) == total, (
            "Suma URL-i z train/val/test jest różna od ich sumy zbiorów — "
            "możliwe wewnętrzne duplikaty w podziale."
        )

    def test_disjointness_is_total(self, split_dfs):
        """Żaden URL nie może pojawić się w więcej niż jednym podzbiorze."""
        train, val, test = split_dfs
        url_to_splits: dict[str, list[str]] = {}
        for url in train["URL"]:
            url_to_splits.setdefault(url, []).append("train")
        for url in val["URL"]:
            url_to_splits.setdefault(url, []).append("val")
        for url in test["URL"]:
            url_to_splits.setdefault(url, []).append("test")

        leaking = {u: splits for u, splits in url_to_splits.items() if len(splits) > 1}
        assert len(leaking) == 0, (
            f"{len(leaking)} URL-i pojawia się w więcej niż jednym zbiorze: "
            f"{dict(list(leaking.items())[:3])}"
        )


# ---------------------------------------------------------------------------
# 2. vocabulary_ i MaxAbsScaler dopasowane wyłącznie na train
# ---------------------------------------------------------------------------

class TestFitOnlyOnTrain:
    """Testy gwarantujące, że transformatory zostały dopasowane wyłącznie na train."""

    @pytest.fixture
    def fitted_pipeline(self, split_dfs):
        train, _, _ = split_dfs
        pipeline = build_feature_pipeline(max_features=500)
        pipeline.fit(train["URL"])
        return pipeline, split_dfs

    def test_tfidf_vocabulary_fitted_only_on_train(self, fitted_pipeline):
        """vocabulary_ TF-IDF nie zmienia się po przetworzeniu val/test."""
        pipeline, (train, val, test) = fitted_pipeline
        tfidf: TfidfVectorizer = pipeline.transformer_list[0][1]

        vocab_before = dict(tfidf.vocabulary_)

        # Transformacja val i test nie powinna modyfikować vocabulary_
        pipeline.transform(val["URL"])
        pipeline.transform(test["URL"])

        vocab_after = dict(tfidf.vocabulary_)
        assert vocab_before == vocab_after, (
            "vocabulary_ TF-IDF uległo zmianie po transform() na val/test — "
            "potencjalny wyciek danych (refitting)."
        )

    def test_tfidf_vocabulary_does_not_expand_on_val_transform(self, fitted_pipeline):
        """Liczba tokenów w vocabulary_ nie rośnie po transform() na val/test."""
        pipeline, (train, val, test) = fitted_pipeline
        tfidf: TfidfVectorizer = pipeline.transformer_list[0][1]

        n_vocab_train = len(tfidf.vocabulary_)
        pipeline.transform(val["URL"])
        pipeline.transform(test["URL"])
        n_vocab_after = len(tfidf.vocabulary_)

        assert n_vocab_after == n_vocab_train, (
            f"Rozmiar vocabulary_ wzrósł: {n_vocab_train} -> {n_vocab_after}. "
            "TF-IDF prawdopodobnie był re-fitowany na val/test."
        )

    def test_maxabs_scaler_fitted_only_on_train(self, fitted_pipeline):
        """MaxAbsScaler.scale_ (max wartości) nie zmienia się po transform() na val/test."""
        pipeline, (train, val, test) = fitted_pipeline
        # pipeline transformer_list[1] to Pipeline([lexical, scaler])
        lexical_pipeline = pipeline.transformer_list[1][1]
        scaler: MaxAbsScaler = lexical_pipeline.named_steps["scaler"]

        scale_before = scaler.scale_.copy()

        pipeline.transform(val["URL"])
        pipeline.transform(test["URL"])

        np.testing.assert_array_equal(
            scaler.scale_,
            scale_before,
            err_msg=(
                "MaxAbsScaler.scale_ uległo zmianie po transform() na val/test — "
                "skaler mógł być re-fitowany (data leakage)."
            ),
        )

    def test_maxabs_scaler_scale_derived_from_train_only(self, fitted_pipeline):
        """scale_ skalera odpowiada max wartościom bezwzględnym z train (nie val/test).

        Uwaga: sklearn MaxAbsScaler zastępuje scale_=0 wartością 1.0 dla kolumn
        z samymi zerami (unikanie dzielenia przez zero) – test uwzględnia tę konwencję.
        """
        pipeline, (train, val, test) = fitted_pipeline
        lexical_pipeline = pipeline.transformer_list[1][1]
        lexical = lexical_pipeline.named_steps["lexical"]
        scaler: MaxAbsScaler = lexical_pipeline.named_steps["scaler"]

        # Oblicz max abs bezpośrednio ze zbioru train
        X_train_raw = lexical.transform(train["URL"])
        expected_scale = np.abs(X_train_raw).max(axis=0)
        # Replikuj zachowanie sklearn: kolumny z max=0 dostają scale_=1.0
        expected_scale[expected_scale == 0.0] = 1.0

        np.testing.assert_allclose(
            scaler.scale_,
            expected_scale,
            rtol=1e-6,
            err_msg=(
                "MaxAbsScaler.scale_ nie odpowiada max|X_train| (z korektą dla zer). "
                "Skaler mógł być dopasowany na danych innych niż train."
            ),
        )

    def test_tfidf_idf_stable_after_val_transform(self, fitted_pipeline):
        """idf_ TF-IDF nie zmienia się po transform() na val/test."""
        pipeline, (_, val, test) = fitted_pipeline
        tfidf: TfidfVectorizer = pipeline.transformer_list[0][1]

        idf_before = tfidf.idf_.copy()
        pipeline.transform(val["URL"])
        pipeline.transform(test["URL"])

        np.testing.assert_array_equal(
            tfidf.idf_,
            idf_before,
            err_msg="idf_ TF-IDF uległo zmianie po transform() na val/test (data leakage).",
        )

    def test_fit_transform_vs_fit_then_transform_equivalence(self, split_dfs):
        """fit_transform(train) == fit(train).transform(train) — brak skutków ubocznych."""
        train, _, _ = split_dfs
        urls = train["URL"]

        pipe_a = build_feature_pipeline(max_features=300)
        X_a = pipe_a.fit_transform(urls)

        pipe_b = build_feature_pipeline(max_features=300)
        pipe_b.fit(urls)
        X_b = pipe_b.transform(urls)

        np.testing.assert_allclose(
            X_a.toarray(),
            X_b.toarray(),
            rtol=1e-6,
            err_msg="fit_transform(train) daje inny wynik niż fit(train).transform(train).",
        )


# ---------------------------------------------------------------------------
# 3. Stabilność podziału — determinizm random_state
# ---------------------------------------------------------------------------

class TestSplitDeterminism:
    """Weryfikacja, że split_data z tym samym wejściem daje identyczny wynik."""

    def test_split_is_deterministic(self):
        df = _make_clean_df(500, seed=7)
        train_a, val_a, test_a = split_data(df)
        train_b, val_b, test_b = split_data(df)

        pd.testing.assert_frame_equal(train_a.reset_index(drop=True), train_b.reset_index(drop=True))
        pd.testing.assert_frame_equal(val_a.reset_index(drop=True), val_b.reset_index(drop=True))
        pd.testing.assert_frame_equal(test_a.reset_index(drop=True), test_b.reset_index(drop=True))

    def test_different_seeds_give_different_splits(self):
        """Dwa wywołania z innym random_state dają różne podziały (sanity-check)."""
        from src.prepare_data import RANDOM_STATE
        import src.prepare_data as prep_module

        df = _make_clean_df(500, seed=0)

        # split z domyślnym seed
        train_default, _, _ = split_data(df)

        # Tymczasowo podmieniamy RANDOM_STATE
        original_rs = prep_module.RANDOM_STATE
        prep_module.RANDOM_STATE = original_rs + 1
        try:
            train_other, _, _ = split_data(df)
        finally:
            prep_module.RANDOM_STATE = original_rs

        # Podziały nie mogą być identyczne przy różnych seed-ach
        assert not train_default["URL"].reset_index(drop=True).equals(
            train_other["URL"].reset_index(drop=True)
        ), "Różne wartości random_state dają identyczny podział — seed jest ignorowany."


# ---------------------------------------------------------------------------
# 4. Brak leakage informacji przez statystyki skalera
# ---------------------------------------------------------------------------

class TestScalerLeakage:
    """Sprawdzenie, że MaxAbsScaler nie „widzi" statystyk val/test podczas fit."""

    def test_scaler_max_does_not_change_after_val_transform(self, split_dfs):
        """max_ skalera (maksima kolumn) pozostaje niezmienne po transform(val/test)."""
        train, val, test = split_dfs
        pipeline = build_feature_pipeline(max_features=400)
        pipeline.fit(train["URL"])

        lexical_pipeline = pipeline.transformer_list[1][1]
        scaler: MaxAbsScaler = lexical_pipeline.named_steps["scaler"]
        max_before = scaler.max_abs_.copy()

        pipeline.transform(val["URL"])
        pipeline.transform(test["URL"])

        np.testing.assert_array_equal(
            scaler.max_abs_,
            max_before,
            err_msg=(
                "MaxAbsScaler.max_abs_ uległo zmianie po transform(val/test). "
                "Skaler był ponownie dopasowywany — to data leakage."
            ),
        )

    def test_val_url_outside_train_does_not_expand_vocabulary(self, split_dfs):
        """URL-e z val nieobecne w train nie rozszerzają vocabulary_ TF-IDF."""
        train, val, _ = split_dfs
        pipeline = build_feature_pipeline(max_features=500)
        pipeline.fit(train["URL"])

        tfidf: TfidfVectorizer = pipeline.transformer_list[0][1]
        vocab_size_before = len(tfidf.vocabulary_)

        # val może zawierać tokeny nieobecne w train — transform() nie powinien ich dodawać
        pipeline.transform(val["URL"])

        vocab_size_after = len(tfidf.vocabulary_)
        assert vocab_size_after == vocab_size_before, (
            f"vocabulary_ rozrosło się po transform(val): "
            f"{vocab_size_before} -> {vocab_size_after} tokenów."
        )

    def test_clean_data_deduplication_prevents_leakage(self):
        """clean_data usuwa duplikaty, co zapobiega potencjalnemu wyciekowi przez powtórzone URL-e."""
        # Ten sam URL z tą samą etykietą — zostaje tylko jeden rekord
        df = pd.DataFrame({
            "URL": ["http://phish.example.com/login"] * 5
                   + ["http://safe.example.com/home"] * 3,
            "Label": ["bad"] * 5 + ["good"] * 3,
        })
        cleaned, stats = clean_data(df)
        assert stats["removed_duplicates"] == 6  # 4 phish + 2 safe duplikatów usunięto
        assert len(cleaned) == 2
        # Po wyczyszczeniu split jest możliwy bez duplikatów (nie testujemy split tu,
        # bo 2 rekordy to za mało, ale duplikaty zostały wyeliminowane)

    def test_no_url_overlap_after_clean_and_split(self):
        """Pełen pipeline: clean_data -> split_data -> brak przecięć URL."""
        raw = pd.DataFrame({
            "URL": [f"http://test-site-{i}.com/page" for i in range(1_000)],
            "Label": (["good"] * 500) + (["bad"] * 500),
        })
        cleaned, _ = clean_data(raw)
        train, val, test = split_data(cleaned)

        assert len(set(train["URL"]) & set(val["URL"])) == 0, "train ∩ val ≠ ∅"
        assert len(set(train["URL"]) & set(test["URL"])) == 0, "train ∩ test ≠ ∅"
        assert len(set(val["URL"]) & set(test["URL"])) == 0, "val ∩ test ≠ ∅"

