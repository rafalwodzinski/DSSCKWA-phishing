# Konwencja pracy zespołu

## Gałęzie

- `main` – stabilna wersja, tylko przez Pull Request (po pierwszym commicie startowym).
- `develop` – gałąź integracyjna całego zespołu; PR-y z gałęzi roboczych trafiają tutaj.
- Gałąź robocza **per Issue**: `<typ>/<nr-issue>-<krótki-opis>`, np. `feat/12-url-lexical-features`.

Każde Issue ma przypisaną własną gałąź roboczą (tworzoną z `develop`).

## Przepływ pracy

1. Weź Issue z GitHub Project Board, przypisz się, przenieś do *In progress*.
2. `git switch develop && git pull && git switch -c feat/<nr>-<opis>`
3. Commituj małymi krokami wg [konwencji commitów](#konwencja-commitów).
4. Przed PR: `pytest` przechodzi, nowy kod ma testy.
5. Otwórz PR do `develop`, powiąż z Issue (`Closes #<nr>`), poproś o review.
6. Merge po min. 1 akceptacji; usuń gałąź po merge'u (squash merge).

## Schemat Pull Requesta

Tytuł PR zapisujemy jak commit (Conventional Commits), np. `feat(features): add URL lexical features`. Opis PR kopiujemy z poniższego schematu:

```markdown
## Opis
Krótko: co zmienia ten PR i dlaczego.

## Powiązane Issue
Closes #<nr>

## Zakres zmian
- [ ] zmiana 1
- [ ] zmiana 2

## Jak przetestować
1. `pytest`
2. (opcjonalnie) kroki do ręcznego sprawdzenia

## Checklista
- [ ] Gałąź utworzona z `develop`, PR kieruje do `develop`
- [ ] `pytest` przechodzi
- [ ] Nowy kod ma testy i docstringi
- [ ] Brak danych, modeli (`*.joblib`) i dużych CSV w commitach
- [ ] Zmiany kontraktów technicznych są opisane w README
- [ ] Przypisano reviewera
```

## Konwencja commitów

Format: `<typ>(<zakres>): <opis w trybie rozkazującym>`

| Typ | Użycie |
|---|---|
| `feat` | nowa funkcjonalność |
| `fix` | poprawka błędu |
| `docs` | dokumentacja, raport |
| `test` | testy |
| `refactor` | refaktoryzacja bez zmiany zachowania |
| `chore` | konfiguracja, zależności, struktura repo |

Przykład: `feat(features): add URL length and digit-ratio features`

## Zasady

- Zespół nie ma stałych ról: każdy samodzielnie wybiera Issue, którym chce się zająć w danym tygodniu (przypisuje się w Project Board).
- Autorstwo zadań jest rejestrowane przez przypisanie w Issue i historię commitów; na tej podstawie powstaje Macierz wkładu w raporcie.

- Nie commitujemy danych (`data/`), modeli (`*.joblib`) ani dużych plików CSV.
- Sekcje raportu IEEE wpisujemy bezpośrednio we wspólnym dokumencie (bez plików pośrednich).
- Zmiany kontraktów technicznych w README wymagają osobnego Issue i zgody zespołu.
- Synchroniczne spotkania zespołu: czwartki 16:15-17:45 (podczas zajęć projektowych z Zarządzania projektem informatycznym)
