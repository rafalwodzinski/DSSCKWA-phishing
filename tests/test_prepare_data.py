"""
tests/test_prepare_data.py

Testy jednostkowe modułu src/prepare_data.py.
"""

import pandas as pd
import pytest

from src.prepare_data import clean_data, load_raw_data, split_data, verify_dod


def test_clean_data_removes_nan_and_empty():
    df = pd.DataFrame({
        "URL": ["google.com", None, "   ", "onet.pl", "wp.pl"],
        "Label": ["good", "bad", "good", None, "   "],
    })
    cleaned, stats = clean_data(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["URL"] == "google.com"
    assert cleaned.iloc[0]["Label"] == 0
    assert stats["removed_nan_or_empty"] == 4


def test_clean_data_unifies_prefixes():
    df = pd.DataFrame({
        "URL": [
            "http://example.com/login",
            "https://example.com/login",
            "http://www.example.com/login",
            "https://www.example.com/login",
            "www.example.com/login",
            "example.com/login",
        ],
        "Label": ["bad"] * 6,
    })
    cleaned, stats = clean_data(df)
    # Po ujednoliceniu prefiksów wszystkie stają się 'example.com/login',
    # więc powinien zostać dokładnie 1 rekord
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["URL"] == "example.com/login"
    assert stats["removed_duplicates"] == 5


def test_clean_data_removes_label_conflicts():
    df = pd.DataFrame({
        "URL": [
            "http://conflict-site.com",
            "https://conflict-site.com",  # ta sama domena po unifikacji, ale inna klasa!
            "safe-site.com",
        ],
        "Label": ["good", "bad", "good"],
    })
    cleaned, stats = clean_data(df)
    # conflict-site.com ma konflikt etykiet -> powinien zostać całkowicie usunięty
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["URL"] == "safe-site.com"
    assert stats["removed_conflicting_urls"] == 1


def test_clean_data_raises_on_invalid_labels():
    df = pd.DataFrame({
        "URL": ["example.com"],
        "Label": ["unknown_class"],
    })
    with pytest.raises(ValueError, match="Wykryto nieznane wartości etykiet"):
        clean_data(df)


def test_split_data_proportions_and_disjoint():
    # Tworzymy syntetyczny zbiór z 1000 wierszy
    urls = [f"site{i}.com" for i in range(1000)]
    labels = [0] * 500 + [1] * 500
    df = pd.DataFrame({"URL": urls, "Label": labels})

    train, val, test = split_data(df)

    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150

    # Przecięcia URL równe 0
    assert len(set(train["URL"]) & set(val["URL"])) == 0
    assert len(set(train["URL"]) & set(test["URL"])) == 0
    assert len(set(val["URL"]) & set(test["URL"])) == 0

    # Identyczny rozkład klas w każdym zbiorze (dokładnie 50% klasy 1)
    assert train["Label"].mean() == 0.5
    assert val["Label"].mean() == 0.5
    assert test["Label"].mean() == 0.5


def test_verify_dod_assertions():
    # Sprawdzenie zachowania verify_dod dla małego zbioru (< 50 000)
    urls = [f"site{i}.com" for i in range(100)]
    labels = [0] * 50 + [1] * 50
    df = pd.DataFrame({"URL": urls, "Label": labels})
    train, val, test = split_data(df)

    with pytest.raises(AssertionError, match="DoD 1 niespełnione"):
        verify_dod(df, train, val, test)


def test_load_raw_data_existing_file(tmp_path):
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text("url,label\nhttp://test.com,good\nhttp://bad.com,bad\n", encoding="utf-8")
    df, count = load_raw_data(csv_path)
    assert count == 2
    assert list(df.columns) == ["URL", "Label"]
    assert len(df) == 2


def test_load_raw_data_auto_download(tmp_path, monkeypatch):
    target_path = tmp_path / "subdir" / "data.csv"
    fake_download_dir = tmp_path / "kaggle_download"
    fake_download_dir.mkdir()
    (fake_download_dir / "phishing.csv").write_text(
        "URL,Label\nhttp://example.com,bad\n", encoding="utf-8"
    )

    monkeypatch.setattr("kagglehub.dataset_download", lambda handle: str(fake_download_dir))

    df, count = load_raw_data(target_path)
    assert target_path.exists()
    assert count == 1
    assert len(df) == 1
