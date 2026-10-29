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
import random
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

DEFAULT_INPUT_FILE = Path("data/processed/test.csv")
DEFAULT_OUTPUT_FILE = Path("data/adversarial/adversarial.csv")
DEFAULT_SEED = 42
DEFAULT_SAMPLES_PER_TECHNIQUE = 125

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


def main() -> None:
    """Punkt wejścia CLI dla skryptu Red Teamingu."""
    parser = argparse.ArgumentParser(
        description="Generator linków adversarial dla Red Teamingu (Issue #12)"
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
    args = parser.parse_args()

    print("=== Generator Red Teamingu (Issue #12) ===")
    try:
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
