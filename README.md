# DSSCKWA-phishing

Wykrywanie phishingowych adresów URL metodami klasycznego uczenia maszynowego (TF-IDF + NB / SVM, cechy leksykalne URL).

📄 [Sprawozdanie (Prism, tylko do odczytu)](https://prism.openai.com/?u=693493ed-d0bc-4893-a8f4-f57962460494&pg=1&m=main.tex&d=7)

## Szybki start

Wymagania: **Python ≥ 3.10**, `git`. Wszystkie polecenia uruchamiaj w katalogu głównym repozytorium.

### 1. Pobranie repozytorium

```bash
git clone https://github.com/rafalwodzinski/DSSCKWA-phishing.git
cd DSSCKWA-phishing
git switch develop
```

### 2. Środowisko wirtualne

Sprawdź wersję Pythona (`python3 --version` na macOS/Linux, `python --version` na Windows), a następnie utwórz i aktywuj środowisko:

| System | Utworzenie | Aktywacja |
|---|---|---|
| macOS / Linux | `python3 -m venv .venv` | `source .venv/bin/activate` |
| Windows (PowerShell) | `python -m venv .venv` | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `python -m venv .venv` | `.venv\Scripts\activate.bat` |

Po aktywacji w terminalu pojawi się prefiks `(.venv)`. Wyjście ze środowiska: `deactivate`.

> Katalog `.venv/` jest w `.gitignore` – nie jest commitowany.

### 3. Instalacja bibliotek

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Weryfikacja: `pip list` powinno pokazać m.in. `scikit-learn`, `pandas`, `numpy`, `scipy`, `tldextract`.

### 4. Uruchomienie testów

```bash
pytest            # wszystkie testy
pytest -v         # z listą testów
pytest tests/test_<nazwa>.py   # pojedynczy plik
```

Testy korzystają z małego zbioru `tests/fixtures/sample_urls.csv`, więc nie wymagają pobierania danych.

### 5. Uruchomienie programu

> 🚧 Do uzupełnienia, gdy program będzie gotowy. Tu pojawią się: pobranie danych do `data/`, trenowanie modeli (zapis do `models/`), predykcja dla pojedynczego URL i generowanie wyników do `reports/`.

### Google Colab

```python
!git clone https://github.com/rafalwodzinski/DSSCKWA-phishing.git
%cd DSSCKWA-phishing
!pip install -r requirements.txt
!pytest
```

W Colabie nie jest potrzebne środowisko wirtualne. Jeśli repozytorium jest prywatne, użyj tokena dostępu (PAT) w adresie klonowania.

## Struktura repozytorium

| Katalog | Przeznaczenie |
|---|---|
| `src/` | Kod źródłowy (moduły projektu) |
| `data/` | Zbiory danych (**w `.gitignore`**) |
| `models/` | Zapisane modele `.joblib` (**w `.gitignore`**) |
| `notebooks/` | Notebooki eksploracyjne |
| `reports/figures/` | Wykresy do raportu |
| `reports/results/` | Tabele i metryki do raportu |
| `tests/fixtures/` | Małe dane testowe (`sample_urls.csv`, 200 wierszy) |

## Słownik skrótów

| Skrót | Znaczenie |
|---|---|
| TF-IDF | Term Frequency – Inverse Document Frequency; wagowanie n-gramów tekstu |
| NB | Naive Bayes (naiwny klasyfikator bayesowski) |
| SVM | Support Vector Machine (maszyna wektorów nośnych) |
| FPR | False Positive Rate – odsetek legalnych URL błędnie oznaczonych jako phishing |
| FNR | False Negative Rate – odsetek phishingu niewykrytego |
| DGA | Domain Generation Algorithm – algorytmicznie generowane domeny |
| ESG | Environmental, Social, Governance – kontekst raportowania zrównoważonego rozwoju |

## Kontrakty techniczne

> Uzupełnić/zatwierdzić na pierwszym spotkaniu zespołu. Zmiana kontraktu = osobne Issue + powiadomienie zespołu.

| Element | Kontrakt |
|---|---|
| Format danych wejściowych | CSV z kolumnami `URL`, `Label` |
| Etykiety | `0` = legalny, `1` = phishing |
| Kodowanie / separator | UTF-8, przecinek |
| Wejście modeli | `pandas.Series` z URL-ami (`str`) lub macierz cech `scipy.sparse` |
| Wyjście klasyfikatora | `numpy.ndarray` etykiet 0/1 + opcjonalnie prawdopodobieństwa klasy 1 |
| Serializacja modeli | `joblib.dump` do `models/<nazwa>.joblib` |
| Ziarno losowości | `RANDOM_STATE = 42` |
| Podział danych | stratyfikowany, train/test 80/20 |
| Metryki | Precision, Recall, F1, FPR, FNR, ROC-AUC |
| Wykresy / wyniki | `reports/figures/*.png` (300 dpi), `reports/results/*.csv` |
| Styl kodu | PEP 8, type hints, docstringi |

## Współpraca

Konwencja commitów (Conventional Commits), gałęzie, Pull Requesty i review: [CONTRIBUTING.md](CONTRIBUTING.md).
