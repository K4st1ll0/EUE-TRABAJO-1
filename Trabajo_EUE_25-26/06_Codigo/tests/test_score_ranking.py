from src.modal_metrics import rank_runs


def test_rank_runs_prefers_lower_weighted_error_score() -> None:
    runs = [
        {"modal_metrics": {"ranking_key": [0.42]}},
        {"modal_metrics": {"ranking_key": [0.18]}},
        {"modal_metrics": {"ranking_key": [0.33]}},
    ]

    ranked = rank_runs(runs)

    assert ranked[0]["modal_metrics"]["ranking_key"] == [0.18]


def test_rank_runs_honors_mac_first_tiebreak_order() -> None:
    runs = [
        {"modal_metrics": {"ranking_key": [-0.91, 0.02, 0.10]}},
        {"modal_metrics": {"ranking_key": [-0.95, 0.05, 0.20]}},
        {"modal_metrics": {"ranking_key": [-0.91, 0.01, 0.05]}},
    ]

    ranked = rank_runs(runs)

    assert ranked[0]["modal_metrics"]["ranking_key"] == [-0.95, 0.05, 0.20]
    assert ranked[1]["modal_metrics"]["ranking_key"] == [-0.91, 0.01, 0.05]

