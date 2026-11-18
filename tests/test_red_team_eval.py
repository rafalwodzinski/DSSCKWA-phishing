from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.red_team import evaluate_model_recall_by_round, export_red_team_results


def test_evaluate_model_recall_by_round_returns_rounds_and_delta_pp():
    dataset = pd.DataFrame(
        {
            "URL": [
                "original-a.com/login",
                "original-b.com/login",
                "original-c.com/login",
                "obfuscated-a.com/login",
                "obfuscated-b.com/login",
                "obfuscated-c.com/login",
            ],
            "Label": [1, 1, 1, 1, 0, 1],
            "Technique": ["round_0", "round_0", "round_0", "url_encoding", "url_encoding", "url_encoding"],
            "Original_URL": [
                "original-a.com/login",
                "original-b.com/login",
                "original-c.com/login",
                "original-a.com/login",
                "original-b.com/login",
                "original-c.com/login",
            ],
        }
    )

    def mock_predict(urls):
        scores = []
        for url in urls:
            if url.startswith("original"):
                scores.append(0.9)
            else:
                scores.append(0.2)
        return scores

    result = evaluate_model_recall_by_round(
        dataset,
        predictions={"model": mock_predict},
        round_order=["round_0", "url_encoding"],
        threshold=0.5,
    )

    assert set(result["Round"]) == {"round_0", "url_encoding"}
    assert "Recall" in result.columns
    assert "Delta_Recall_pp" in result.columns
    assert result.loc[result["Round"] == "url_encoding", "Delta_Recall_pp"].iloc[0] == 100.0


def test_export_red_team_results_writes_csv_and_png(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "URL": [
                "baseline-a.com/login",
                "baseline-b.com/login",
                "evaded-a.com/login",
                "evaded-b.com/login",
            ],
            "Label": [1, 1, 1, 1],
            "Technique": ["round_0", "round_0", "url_encoding", "url_encoding"],
            "Original_URL": [
                "baseline-a.com/login",
                "baseline-b.com/login",
                "baseline-a.com/login",
                "baseline-b.com/login",
            ],
        }
    )

    csv_path = tmp_path / "red_team_eval.csv"
    image_path = tmp_path / "red_team_recall_drop.png"

    def mock_predict(urls):
        return [0.9 if url.startswith("baseline") else 0.2 for url in urls]

    result = export_red_team_results(
        dataset=dataset,
        model_predictions={"hybrid": mock_predict, "baseline": mock_predict},
        csv_path=csv_path,
        figure_path=image_path,
        threshold=0.5,
    )

    assert csv_path.exists()
    assert image_path.exists()
    assert len(result) >= 2


def test_evaluate_model_recall_by_round_supports_per_model_thresholds():
    """Weryfikuje obsługę słownika progów (np. Baseline: 0.5, Hybrid: -0.088) oraz fallback dla predict_proba."""
    dataset = pd.DataFrame(
        {
            "URL": ["orig1.com/login", "orig2.com/login", "adv1.com/login", "adv2.com/login"],
            "Label": [1, 1, 1, 1],
            "Technique": ["round_0", "round_0", "typosquatting", "typosquatting"],
            "Original_URL": ["orig1.com/login", "orig2.com/login", "orig1.com/login", "orig2.com/login"],
        }
    )

    class DummyProbaModel:
        def predict_proba(self, urls):
            # Pierwszy URL ma P(1)=0.8 (>=0.5), drugi ma P(1)=0.2 (<0.5, ale >= -0.088)
            return [[0.2, 0.8] if "1" in u else [0.8, 0.2] for u in urls]

    class DummyDecisionModel:
        def decision_function(self, urls):
            return [0.5 if "1" in u else -0.05 for u in urls]

    res = evaluate_model_recall_by_round(
        dataset,
        predictions={"Baseline": DummyProbaModel(), "Hybrid": DummyDecisionModel()},
        round_order=["round_0", "typosquatting"],
        threshold={"Baseline": 0.5, "Hybrid": -0.088},
    )

    base_r0 = res[(res["Model"] == "Baseline") & (res["Round"] == "round_0")]["Recall"].iloc[0]
    hyb_r0 = res[(res["Model"] == "Hybrid") & (res["Round"] == "round_0")]["Recall"].iloc[0]

    # Przy progu 0.5 dla Baseline wykrywana jest 1 z 2 próbek (Recall = 0.5)
    assert base_r0 == 0.5
    # Przy progu -0.088 dla Hybrid obie próbki (0.5 oraz -0.05) są >= -0.088 (Recall = 1.0)
    assert hyb_r0 == 1.0

