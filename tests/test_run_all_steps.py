"""Driver contract: the run_all step list and the quick
expansion are frozen (12 steps; quick = verify + aggregate + figures)."""

EXPECTED_STEPS = ["verify", "factorial", "sparkf", "append", "aggregate",
                  "ablation", "falseesc", "regret", "ops", "eeg", "bp",
                  "figures"]


def test_step_list_is_expected_12():
    from scripts import run_all

    assert run_all.STEPS == EXPECTED_STEPS


def test_quick_expands_to_verify_aggregate_figures():
    from scripts import run_all

    assert run_all.QUICK_STEPS == ["verify", "aggregate", "figures"]
    assert set(run_all.QUICK_STEPS) <= set(run_all.STEPS)
