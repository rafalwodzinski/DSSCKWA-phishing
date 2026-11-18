"""Testy jednostkowe modułu Red Teamingu (src/red_team.py)."""

from __future__ import annotations

import random
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import pytest

from src.red_team import (
    TECHNIQUE_MAPPING,
    apply_combined_attack,
    apply_fake_query_params,
    apply_subdomain_stuffing,
    apply_typosquatting,
    apply_url_encoding,
    generate_adversarial_dataset,
    verify_dod,
)


@pytest.fixture
def sample_phishing_csv(tmp_path: Path) -> Path:
    """Tworzy mały fixture test.csv z linkami phishingowymi."""
    csv_path = tmp_path / "test.csv"
    data = {
        "URL": [
            "anene.info/design/login.php",
            "secure-bank.com/account/auth.html",
            "verify-paypal.org/update?user=123",
            "allegro-promocja.net/claim/index",
            "mbank-serwis.top/panel/login",
            "apple-id-cloud.com/auth/verify",
            "dhl-przesylka.biz/track/parcel",
            "inpost-paczka.online/odbierz/kod",
            "microsoft-office365.site/session",
            "netflix-subskrypcja.club/billing",
        ]
        * 15,  # 150 linków phishingowych
        "Label": [1] * 150,
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


def test_apply_subdomain_stuffing():
    """Weryfikacja techniki 1: Subdomain stuffing dokleja zaufane marki."""
    rng = random.Random(42)
    original_url = "malicious-host.ru/login/form"
    adv_url = apply_subdomain_stuffing(original_url, rng)

    assert adv_url != original_url
    # Sprawdzamy, czy w hoście pojawiła się znana marka lub domenowa subdomena
    assert any(
        brand in adv_url
        for brand in ["paypal.com", "apple.com", "microsoft", "allegro", "mbank"]
    )
    # Sprawdzamy poprawność składniową
    parsed = urlsplit(adv_url if "://" in adv_url else f"//{adv_url}")
    assert parsed.netloc != ""


def test_apply_url_encoding():
    """Weryfikacja techniki 2: Zamiana znaków w ścieżce na %xx."""
    rng = random.Random(42)
    original_url = "phishing-site.com/account/login/verify.html"
    adv_url = apply_url_encoding(original_url, rng)

    assert adv_url != original_url
    # Sprawdzamy obecność sekwencji kodowania procentowego (%xx)
    assert "%" in adv_url
    assert any(hex_code in adv_url for hex_code in ["%2f", "%61", "%65", "%69", "%6f"])


def test_apply_typosquatting():
    """Weryfikacja techniki 3: Podmiana znaków wizualnie podobnych (m->rn, l->1, o->0)."""
    rng = random.Random(42)
    original_url = "online-login-mail.com/portal/home"
    adv_url = apply_typosquatting(original_url, rng)

    assert adv_url != original_url
    # Sprawdzamy obecność typowych podstawień
    has_typo = any(
        t in adv_url for t in ["rn", "1", "0", "vv"]
    ) or adv_url.startswith("rn")
    assert has_typo


def test_apply_fake_query_params():
    """Weryfikacja techniki 4: Dopisanie 'bezpiecznych' parametrów query."""
    rng = random.Random(42)
    original_url = "bank-auth.com/login"
    adv_url = apply_fake_query_params(original_url, rng)

    assert adv_url != original_url
    assert "?" in adv_url
    assert any(
        param in adv_url
        for param in ["safe=true", "redirect=google.com", "verified=1", "ssl=1"]
    )


def test_apply_combined_attack():
    """Weryfikacja rundy kombinowanej: łączenie min. 3 technik."""
    rng = random.Random(42)
    original_url = "bad-portal.org/login/index.php"
    adv_url = apply_combined_attack(original_url, rng)

    assert adv_url != original_url
    # Powinny wystąpić co najmniej 2 z cech (np. query, subdomena, encoding lub typosquatting)
    indicators = [
        "?" in adv_url,
        "%" in adv_url,
        any(b in adv_url for b in ["paypal", "apple", "google", "allegro", "mbank"]),
        any(t in adv_url for t in ["rn", "1", "0"]),
    ]
    assert sum(indicators) >= 2


def test_determinism_seed(sample_phishing_csv: Path, tmp_path: Path):
    """Weryfikacja deterministyczności: ten sam seed daje identyczny wynik."""
    out_1 = tmp_path / "adv1.csv"
    out_2 = tmp_path / "adv2.csv"

    df_1 = generate_adversarial_dataset(
        sample_phishing_csv, out_1, samples_per_technique=10, seed=42
    )
    df_2 = generate_adversarial_dataset(
        sample_phishing_csv, out_2, samples_per_technique=10, seed=42
    )

    pd.testing.assert_frame_equal(df_1, df_2)

    # Inny seed daje inny wynik
    out_3 = tmp_path / "adv3.csv"
    df_3 = generate_adversarial_dataset(
        sample_phishing_csv, out_3, samples_per_technique=10, seed=123
    )
    assert not df_1["URL"].equals(df_3["URL"])


def test_generate_adversarial_dataset_dod(sample_phishing_csv: Path, tmp_path: Path):
    """Weryfikacja kryteriów Definition of Done (DoD)."""
    out_csv = tmp_path / "adversarial.csv"
    samples_per_tech = 125
    df = generate_adversarial_dataset(
        input_path=sample_phishing_csv,
        output_path=out_csv,
        samples_per_technique=samples_per_tech,
        seed=42,
    )

    # 1. Minimum 500 rekordów (125 * 5 = 625)
    assert len(df) == 625
    assert len(df) >= 500

    # 2. Każda z 4 technik + kombinowana ma dokładnie po 125 próbek
    for tech in TECHNIQUE_MAPPING.keys():
        assert len(df[df["Technique"] == tech]) == 125

    # 3. Wszystkie etykiety to 1 (phishing)
    assert (df["Label"] == 1).all()

    # 4. Brak brakujących wartości
    assert df["URL"].isna().sum() == 0
    assert df["Original_URL"].isna().sum() == 0

    # 5. Funkcja verify_dod potwierdza spełnienie wszystkich kryteriów
    dod_results = verify_dod(df)
    assert all(dod_results.values()), f"Kryteria DoD niespełnione: {dod_results}"


def test_real_dataset_generation_if_exists():
    """Test weryfikujący generowanie na rzeczywistym data/processed/test.csv jeśli istnieje."""
    real_test_csv = Path("data/processed/test.csv")
    if not real_test_csv.exists():
        pytest.skip("data/processed/test.csv nie istnieje lokalnie, pomijam")

    out_csv = Path("data/adversarial/adversarial.csv")
    df = generate_adversarial_dataset(
        input_path=real_test_csv,
        output_path=out_csv,
        samples_per_technique=125,
        seed=42,
    )

    assert len(df) >= 500
    assert out_csv.exists()
    dod_results = verify_dod(df)
    assert all(dod_results.values())
