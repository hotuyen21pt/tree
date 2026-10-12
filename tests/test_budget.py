import time

from tree_cot_rt.budget import Budget, Deadline, steps_within


def test_unlimited_budget_never_expires():
    budget = Budget(None)
    assert budget.unlimited
    assert budget.remaining() == float("inf")
    assert not budget.expired()
    deadline = Deadline(budget, None, "stage")
    assert deadline.seconds is None
    assert not deadline.expired()
    assert "no time limit" in deadline.describe()


def test_deadline_cannot_outlive_the_session_budget():
    budget = Budget(10.0)
    # A phase asking for more than the session has left is clamped to it.
    deadline = Deadline(budget, 600.0, "greedy_stage")
    assert deadline.seconds is not None
    assert deadline.seconds <= 10.0


def test_expiring_a_deadline_records_the_phase_once(capsys):
    budget = Budget(100.0)
    deadline = Deadline(budget, 0.0, "stage_one")
    assert deadline.expired()
    assert deadline.expired()  # still expired, but reported only once
    assert budget.truncated == ["stage_one"]
    assert capsys.readouterr().out.count("stage_one") == 1


def test_report_describes_what_was_cut_short():
    budget = Budget(100.0)
    Deadline(budget, 0.0, "S1a_generation").expired()
    report = budget.report()
    assert report["total_seconds"] == 100.0
    assert report["truncated_phases"] == ["S1a_generation"]
    assert report["remaining_seconds"] < 100.0


def test_stage_shares_split_the_remaining_time():
    # Mirrors the training loop: each stage takes a share of what is *left*,
    # so the shares always add up to the original budget and a stage that
    # overruns silently shrinks the ones after it rather than the total.
    total = 1000.0
    fractions = [0.25, 0.25, 0.50]
    remaining = total
    allotted = []
    for index, _ in enumerate(fractions):
        share = fractions[index] / sum(fractions[index:])
        seconds = remaining * share
        allotted.append(seconds)
        remaining -= seconds  # the stage then spends exactly its slice
    assert allotted == [250.0, 250.0, 500.0]
    assert sum(allotted) == total
    assert remaining == 0.0


def test_a_stage_that_overruns_shrinks_only_the_later_stages():
    total = 1000.0
    fractions = [0.25, 0.25, 0.50]
    remaining = total
    allotted = []
    for index, _ in enumerate(fractions):
        share = fractions[index] / sum(fractions[index:])
        seconds = remaining * share
        allotted.append(seconds)
        # First stage overspends by 150s; later stages must absorb it.
        remaining -= seconds + (150.0 if index == 0 else 0.0)
    assert allotted[0] == 250.0
    assert sum(allotted) + 150.0 == total
    assert allotted[2] < 500.0


def test_steps_within_fits_the_loop_to_the_deadline():
    budget = Budget(100.0)
    deadline = Deadline(budget, 10.0, "loop")
    assert steps_within(deadline, seconds_per_step=1.0, requested=100) <= 10
    assert steps_within(deadline, seconds_per_step=1.0, requested=3) == 3
    assert steps_within(Deadline(Budget(None), None, "loop"), 1.0, 42) == 42


def test_elapsed_advances():
    budget = Budget(5.0)
    before = budget.remaining()
    time.sleep(0.01)
    assert budget.remaining() < before
