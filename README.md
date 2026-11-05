# DSSCKWA-phishing

Wykrywanie phishingowych adresów URL metodami klasycznego uczenia maszynowego (hybrydowy znakowy TF-IDF 3–5 + 12 cech leksykalno-entropijnych, klasyfikatory Naive Bayes i LinearSVC) w kontekście symulacji bramki bezpieczeństwa poczty i ruchu webowego (ESG/SOC proxy).

📄 [Sprawozdanie (Prism, tylko do odczytu)](https://prism.openai.com/?u=693493ed-d0bc-4893-a8f4-f57962460494&pg=1&m=main.tex&d=7)

---

## Checklista zlecenia (weryfikacja w 3 minuty)

### Wymóg ze zlecenia vs Osiągnięty wynik na zbiorze testowym (`test.csv`, $N = 75\,964$)

| Kryterium / Metryka | Wymóg ze zlecenia | Baseline (`MultinomialNB`, word tokens) | Best Hybrid (`LinearSVC`, char TF-IDF + 12 cech, $\tau^* = -0.088141$) | Status |
|---|---|---|---|---|
| **F1-Score** | $\ge 0.92$ | `0.8916` | **`0.9466`** (Precision: `0.9535`, Recall: `0.9399`) | ✅ Spełniony |
| **False Positive Rate (FPR)** | $\le 0.02$ ($\le 2.0\%$) | `0.0181` (`1.81%`) | **`0.0133`** (**`1.33%`**, FNR: `6.01%`) | ✅ Spełniony |
| **ROC-AUC** | $\ge 0.95$ | `0.9840` | **`0.9944`** | ✅ Spełniony |
| **$\Delta\text{Recall}$ Red Team (5 rund Evasion)** | Analiza odporności na ataki omijające (Rundy 1–5 vs Runda 0) | R0: `83.04%`<br>Śr. $\Delta\text{Recall}$: `-9.44 p.p.`<br>Max spadek (R3): `+1.44 p.p.` | R0: **`93.76%`**<br>Śr. $\Delta\text{Recall}$: **`-1.07 p.p.`**<br>Max spadek (R2 *URL encoding*): `+6.56 p.p.` (R1, R4, R5: wzrost do `94.4%–100%`) | ✅ Spełniony |
| **Czas inferencji (`Latency ms`)** | $\le 10.0\text{ ms / URL}$ | średnia: `0.154 ms/URL`<br>p95: `0.170 ms/URL` | średnia: **`0.472 ms/URL`**<br>p95: **`0.537 ms/URL`** | ✅ Spełniony ($\sim 20\times$ szybciej) |
| **Czas treningu (`Train time s`)** | $\le 90.0\text{ s}$ na CPU | `1.14 s` (5-fold CV: `1.59 s`) | **`29.57 s`** na zbiorze `train.csv` ($N = 354\,494$) | ✅ Spełniony ($\sim 3\times$ szybciej) |

### Porównanie walidacji krzyżowej (5-fold StratifiedKFold na `train.csv`, $N = 354\,494$)

| Model | Reprezentacja cech | F1 (5-fold CV) | FPR (5-fold CV) | ROC-AUC (5-fold CV) | Czas 1 foldu |
|---|---|---|---|---|---|
| **Baseline (`MultinomialNB`)** | Word-level Bag-of-Words (`max_features=20000`) | `0.8876 ± 0.0013` | `0.0185 ± 0.0005` | `0.9836 ± 0.0004` | `1.32 s` |
| **Hybrid `ComplementNB`** | Char TF-IDF (3–5) + 12 cech leksykalnych + `MaxAbsScaler` | `0.8629 ± 0.0028` | `0.0475 ± 0.0017` | `0.9800 ± 0.0005` | `26.69 s` |
| **Hybrid `LinearSVC`** | Char TF-IDF (3–5) + 12 cech leksykalnych + `MaxAbsScaler` | **`0.9408 ± 0.0004`** | **`0.0107 ± 0.0004`** | **`0.9935 ± 0.0003`** | `30.65 s` |

---

## Instrukcja uruchomienia krok po kroku

Wymagania: **Python ≥ 3.10**, `git`. Wszystkie polecenia uruchamiaj w katalogu głównym repozytorium.

### 1. Pobranie repozytorium i środowisko wirtualne

```bash
git clone https://github.com/rafalwodzinski/DSSCKWA-phishing.git
cd DSSCKWA-phishing
git switch develop
```

Utwórz i aktywuj środowisko wirtualne oraz zainstaluj zależności:

| System | Utworzenie | Aktywacja |
|---|---|---|
| macOS / Linux | `python3 -m venv .venv` | `source .venv/bin/activate` |
| Windows (PowerShell) | `python -m venv .venv` | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `python -m venv .venv` | `.venv\Scripts\activate.bat` |

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pełny potok przetwarzania i eksperymentów (Krok po kroku)

Poniższe komendy można skopiować bezpośrednio do terminala (realizują kompletny przepływ od surowych danych do symulacji ESG/SOC proxy):

```bash
# Krok 1 (S1): Pobranie/wczytanie surowego zbioru, czyszczenie, deduplikacja i stratyfikowany podział 70/15/15
python -m src.prepare_data

# Krok 2 (S2): Ekstrakcja 12 cech leksykalno-entropijnych + char TF-IDF (3-5) i zapis models/feature_pipeline.joblib
python -m src.extract_features

# Krok 3: Trening modelu bazowego (MultinomialNB) oraz modeli hybrydowych (ComplementNB i LinearSVC) z 5-fold CV
python -m src.train --model baseline
python -m src.train --model cnb
python -m src.train --model svm

# Krok 4 (S3): Strojenie hiperparametrów na zbiorze walidacyjnym i dobór progu decyzyjnego (FPR <= 0.018)
python -m src.optimize_threshold

# Krok 5: Generowanie złośliwych mutacji URL (5 rund Evasion) i ewaluacja odporności Red Team (Baseline vs Hybrid)
python -m src.red_team --evaluate

# Krok 6: Jednokrotna ewaluacja końcowa na zbiorze testowym (test.csv) oraz benchmark czasu inferencji i treningu
python -m src.final_eval

# Krok 7: Inspekcja pojedynczego lub wielu adresów URL w CLI (symulacja bramki ESG/SOC proxy: ALLOW / QUARANTINE / PHISHING)
python -m src.predict --url "http://secure-login.paypa1.com/verify" --pretty
```

### 3. Uruchomienie testów automatycznych

```bash
pytest -v
```

Zestaw 103 testów jednostkowych i integracyjnych (w tym weryfikacja braku przecieku danych `test_no_leakage.py`) korzysta z fikstur w `tests/fixtures/sample_urls.csv` i wykonuje się w kilka sekund.

### Google Colab

```python
!git clone https://github.com/rafalwodzinski/DSSCKWA-phishing.git
%cd DSSCKWA-phishing
!pip install -r requirements.txt
!pytest
```

---

## Contribution Matrix (Wkład zespołu)

Zespół pracował zgodnie z zasadami opisanymi w [CONTRIBUTING.md](CONTRIBUTING.md) (feature branches od `develop`, Conventional Commits, obowiązkowe Code Review i Pull Requesty powiązane z Issues). Wszystkie zadania zostały zrealizowane przy równym wkładzie pracy (**25% każdy członek zespołu**).

| Członek zespołu | Konto GitHub / Git | Udział | Główne obszary odpowiedzialności i zrealizowane zadania | Zmergowane Pull Requesty (PR) |
|---|---|---|---|---|
| **Rafał Wodziński** | `@rafalwodzinski` (`Rafał Wodziński`) | **25%** | Inicjalizacja repozytorium i CI/CD, budowa hybrydowego potoku cech `FeatureUnion` (**S2**), trening modelu `Baseline` (`MultinomialNB`) oraz modeli hybrydowych (`ComplementNB`, `LinearSVC`) z 5-fold CV, strojenie hiperparametrów i optymalizacja progu decyzyjnego $\tau^*$ (**S3**), analiza anomalii Red Teaming i finalizacja dokumentacji. | `#22`, `#24`, `#26`, `#28`, `#33`, `#34` |
| **Julita Skorżycka** | `@JulitaSkorzycka` (`JulitaSkorzycka`) | **25%** | Zunifikowany moduł ewaluacji metryk i wizualizacji (`src/evaluate.py`, **S0**), implementacja ekstraktora 12 cech leksykalno-entropijnych `LexicalExtractor`, eksploracyjna analiza danych (`notebooks/01_eda.ipynb`), generator 5 rund ataków omijających i ewaluacja Red Team (`src/red_team.py`). | `#19`, `#21`, `#23`, `#25`, `#31` |
| **Anna Tokarz** | `@nisee` (`nisee`, `Ania`) | **25%** | Potok pobierania, czyszczenia, deduplikacji i stratyfikowanego podziału danych `train/val/test` (`src/prepare_data.py`, **S1**), testy jednostkowe i weryfikacja braku przecieku danych (`tests/test_no_leakage.py`), interfejs CLI symulujący bramkę bezpieczeństwa ESG/SOC (`src/predict.py`: `ALLOW` / `QUARANTINE` / `PHISHING`). | `#20`, `#27`, `#29` |
| **Bartłomiej Piwowar** | `@BartlomiejPiwowar` (`BartlomiejPiwowar`, `Bartłomiej Piwowar`) | **25%** | Współtworzenie analizy eksploracyjnej (EDA), moduł jednokrotnej ewaluacji końcowej na zbiorze testowym (`src/final_eval.py`), benchmarki wydajnościowe (czas treningu $\le 90\text{ s}$ oraz latencja inferencji $\le 10\text{ ms/URL}$), generowanie finalnych wykresów macierzy pomyłek, krzywych ROC i ważności cech. | `#23` *(współautor)*, `#30` |

### Wykaz zmergowanych Pull Requestów

- **PR #19** – `feat(evaluate): add unified evaluation metrics and visualization module` (`@JulitaSkorzycka`)
- **PR #20** – `Feat/3 prepare data` (`@nisee` – Anna Tokarz)
- **PR #21** – `Feat/5 extract features` (`@JulitaSkorzycka`)
- **PR #22** – `feat(features): add hybrid char TF-IDF and lexical pipeline (S2)` (`@rafalwodzinski`)
- **PR #23** – `feat(eda): initial eda notebook setup` (`@JulitaSkorzycka`, `@BartlomiejPiwowar`)
- **PR #24** – `feat(ml): add word-token MultinomialNB baseline model` (`@rafalwodzinski`)
- **PR #25** – `Feat/12 red team` (`@JulitaSkorzycka`)
- **PR #26** – `feat(ml): add hybrid ComplementNB and LinearSVC training with 5-fold CV` (`@rafalwodzinski`)
- **PR #27** – `test(leakage): add data leakage verification tests (#6)` (`@nisee` – Anna Tokarz)
- **PR #28** – `feat(ml): add hyperparameter tuning and threshold optimization for S3` (`@rafalwodzinski`)
- **PR #29** – `feat(predict): add URL inspection CLI for ESG/SOC proxy simulation` (`@nisee` – Anna Tokarz)
- **PR #30** – `feat(eval): jednokrotna ewaluacja na zbiorze testowym i benchmark (≤ 10 ms/URL, ≤ 90 s)` (`@BartlomiejPiwowar`)
- **PR #31** – `Fix red-team evaluation and document issue mismatch` (`@JulitaSkorzycka`, `@BartlomiejPiwowar`)
- **PR #33** – `fix(red-team): apply per-model decision thresholds and analyze evasion anomalies` (`@rafalwodzinski`)

### Podsumowanie `git shortlog -sn` (gałąź `develop` po squash-merge PR-ów)

```text
    10  Rafał Wodziński
     5  JulitaSkorzycka
     3  nisee
     1  BartlomiejPiwowar
```

*(Uwaga: Z uwagi na scalanie gałęzi zadaniowych metodą **Squash and merge** na GitHubie, każdy zmergowany PR odpowiada pojedynczemu commitowi w historii głównej gałęzi `develop`).*

---

## Struktura repozytorium

| Katalog / Plik | Przeznaczenie |
|---|---|
| `src/prepare_data.py` | Pobieranie, czyszczenie, deduplikacja i stratyfikowany podział 70/15/15 (**S1**) |
| `src/extract_features.py` | Transformator `LexicalExtractor` (12 cech) i potok `FeatureUnion` z char TF-IDF 3–5 (**S2**) |
| `src/evaluate.py` | Zunifikowane metryki (`F1`, `Precision`, `Recall`, `FPR`, `FNR`, `ROC-AUC`) i wykresy (**S0**) |
| `src/train.py` | Trening `Baseline` (`MultinomialNB`) oraz modeli hybrydowych (`ComplementNB`, `LinearSVC`) z 5-fold CV |
| `src/optimize_threshold.py` | Strojenie hiperparametrów na `val.csv` i dobór progu decyzyjnego $\tau^*$ dla $\text{FPR} \le 0.018$ (**S3**) |
| `src/red_team.py` | Generator 5 rund ataków omijających (Evasion) i ewaluacja spadku czułości ($\Delta\text{Recall}$) |
| `src/final_eval.py` | Jednokrotna ewaluacja końcowa na `test.csv` oraz benchmarki czasu treningu i inferencji |
| `src/predict.py` | CLI do inspekcji adresów URL w czasie rzeczywistym (`ALLOW` / `QUARANTINE` / `PHISHING`) |
| `notebooks/01_eda.ipynb` | Eksploracyjna analiza danych (EDA) i rozkłady cech leksykalnych |
| `reports/figures/` | Wygenerowane wykresy 300 DPI do raportu IEEE |
| `reports/results/` | Zapisane tabele wyników, statystyki zbioru i metryki (`.json`, `.csv`) |
| `tests/` | Testy jednostkowe, integracyjne oraz testy braku przecieku danych (`tests/fixtures/sample_urls.csv`) |

## Słownik skrótów

| Skrót | Znaczenie |
|---|---|
| TF-IDF | Term Frequency – Inverse Document Frequency; wagowanie n-gramów tekstu |
| NB | Naive Bayes (naiwny klasyfikator bayesowski: `MultinomialNB`, `ComplementNB`) |
| SVM | Support Vector Machine (maszyna wektorów nośnych: `LinearSVC`) |
| FPR | False Positive Rate – odsetek legalnych URL błędnie oznaczonych jako phishing |
| FNR | False Negative Rate – odsetek phishingu niewykrytego |
| DGA | Domain Generation Algorithm – algorytmicznie generowane domeny |
| ESG / SOC | Email Security Gateway / Security Operations Center – kontekst wdrożenia filtra URL |

## Kontrakty techniczne

| Element | Kontrakt |
|---|---|
| Format danych wejściowych | CSV z kolumnami `URL`, `Label` |
| Etykiety | `0` = legalny (`good`), `1` = phishing (`bad`) |
| Kodowanie / separator | UTF-8, przecinek |
| Wejście modeli | `pandas.Series` / sekwencja z URL-ami (`str`) lub macierz cech `scipy.sparse` |
| Wyjście klasyfikatora | `numpy.ndarray` etykiet `0`/`1` + ciągłe wyniki decyzyjne (`predict_proba` / `decision_function`) |
| Serializacja modeli | `joblib.dump` do `models/<nazwa>.joblib` (`feature_pipeline.joblib`, `baseline_mnb.joblib`, `best_hybrid.joblib`) |
| Ziarno losowości | `RANDOM_STATE = 42` |
| Podział danych | Stratyfikowany `train` / `val` / `test` w proporcji **70% / 15% / 15%** (0% części wspólnej URL) |
| Metryki | Precision, Recall, F1, FPR, FNR, ROC-AUC |
| Wykresy / wyniki | `reports/figures/*.png` (300 dpi), `reports/results/*.json` oraz `*.csv` |
| Styl kodu | PEP 8, type hints, docstringi |

## Współpraca

Konwencja commitów (Conventional Commits), gałęzie, Pull Requesty i review: [CONTRIBUTING.md](CONTRIBUTING.md).
