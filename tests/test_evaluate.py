"""Offline contracts for the evaluation set and scorer. No models, no macOS calls."""

from collections import Counter, defaultdict

import pytest

from laya_voice import actions, evaluate
from laya_voice.brain import ACTIONS, Brain, Plan
from laya_voice.evaluate import Case, Result, executes, matches, outcome, score, suggest_threshold, sweep

CASES = evaluate.load_cases()
CTX = evaluate.load_context()
QUESTIONS = Brain.__new__(Brain)._questions({"c0": "x"}, CTX["apps"])

CHOICE_ARGS = {
    ("media", "op"): "media_op", ("volume", "op"): "volume_op", ("system", "op"): "system_op",
    ("shortcut", "shortcut"): "shortcut", ("open_app", "app"): "app", ("scroll", "direction"): "scroll_dir",
    ("scroll", "amount"): "scroll_amount", ("open_folder", "folder"): "folder", ("web_search", "engine"): "engine",
    ("open_website", "site"): "site", ("new_item", "kind"): "new_kind",
}
FREE_ARGS = {("type_text", "text"), ("type_text", "submit"), ("web_search", "query"), ("new_item", "title")}


def test_ids_are_unique():
    dup = [i for i, n in Counter(c.id for c in CASES).items() if n > 1]
    assert not dup


def test_splits_are_known_and_groups_do_not_cross_them():
    splits = defaultdict(set)
    for c in CASES:
        assert c.split in evaluate.SPLITS
        splits[c.group].add(c.split)
    assert {g: s for g, s in splits.items() if len(s) > 1} == {}


def test_every_category_is_in_both_splits():
    by_cat = defaultdict(set)
    for c in CASES:
        by_cat[c.category].add(c.split)
    assert {cat: s for cat, s in by_cat.items() if s != set(evaluate.SPLITS)} == {}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_expectations_use_offered_options(case):
    if case.reject:
        assert case.expect == {"reject": True}
        return
    action = case.expect["action"]
    assert action in ACTIONS and action not in ("none", "task")
    for arg, value in case.expect.items():
        if arg == "action":
            continue
        key = CHOICE_ARGS.get((action, arg))
        if key:
            assert value in QUESTIONS[key]["criteria"], f"{arg}={value!r} is not an option of {key}"
        else:
            assert (action, arg) in FREE_ARGS, f"{action} has no argument {arg}"


def test_context_apps_cover_expected_apps():
    wanted = {c.expect["app"] for c in CASES if c.expect.get("app")}
    assert wanted <= set(CTX["apps"])


def test_load_cases_filters_split():
    dev = evaluate.load_cases(split="dev")
    assert dev and all(c.split == "dev" for c in dev)
    assert len(evaluate.load_cases(split="all")) == len(CASES)


def case(expect, category="media"):
    return Case("x", "g", "dev", category, "u", expect)


def plan(action="media", conf=0.9, error=None, **args):
    return Plan("u", action, conf, args, error=error)


MEDIA = {"action": "media", "op": "play_pause"}


@pytest.mark.parametrize("p,threshold,runs", [
    (plan(conf=0.5), 0.35, True),
    (plan(conf=0.2), 0.35, False),
    (plan("none", 1.0), 0.0, False),
    (plan("stop", 0.0), 0.9, True),
    (plan(conf=1.0, error="boom"), 0.0, False),
])
def test_executes_mirrors_main(p, threshold, runs):
    assert executes(p, threshold) is runs


def test_matches_compares_only_expected_args_and_normalizes_text():
    assert matches(plan(op="play_pause", compound=False), MEDIA)
    assert not matches(plan(op="next"), MEDIA)
    assert not matches(plan("volume", op="play_pause"), MEDIA)
    assert matches(plan("type_text", text="Hello  World", submit=False), {"action": "type_text", "text": "hello world", "submit": False})
    assert not matches(plan("type_text", text="hello world", submit=True), {"action": "type_text", "text": "hello world", "submit": False})
    assert not matches(plan(), {"reject": True})


@pytest.mark.parametrize("p,expect,want", [
    (plan(op="play_pause"), MEDIA, "correct"),
    (plan(op="next"), MEDIA, "false_action"),
    (plan(op="play_pause", conf=0.1), MEDIA, "held"),
    (plan(op="play_pause"), {"reject": True}, "false_action"),
    (plan(op="play_pause", conf=0.1), {"reject": True}, "held"),
    (plan("none", 1.0), {"reject": True}, "held"),
    (plan(error="laya: down"), MEDIA, "held"),
])
def test_outcome(p, expect, want):
    assert outcome(Result(case(expect), p), 0.35) == want


def results():
    return [
        Result(case(MEDIA), plan(op="play_pause", conf=0.9)),
        Result(case(MEDIA), plan(op="next", conf=0.6)),
        Result(case(MEDIA), plan(op="play_pause", conf=0.2)),
        Result(case({"reject": True}, "reject_unsafe"), plan("system", 0.4, op="lock")),
        Result(case({"reject": True}, "reject_unsafe"), plan("none", 1.0)),
    ]


def test_score_counts():
    s = score(results(), 0.35)
    assert (s["cases"], s["commands"], s["rejects"]) == (5, 3, 2)
    assert s["command_accuracy"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["action_accuracy"] == 1.0
    assert s["arg_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["reject_rate"] == 0.5
    assert s["missed_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["false_action_rate"] == 0.4
    assert s["by_category"]["reject_unsafe"] == {"n": 2, "false_action": 1, "held": 1}


def test_score_handles_empty():
    s = score([], 0.35)
    assert s["cases"] == 0 and s["command_accuracy"] is None


def test_sweep_false_actions_never_rise_with_threshold():
    rates = [row["false_action_rate"] for row in sweep(results(), [0.0, 0.3, 0.5, 0.7, 0.95])]
    assert rates == sorted(rates, reverse=True)


def test_suggest_threshold_picks_lowest_within_budget():
    assert suggest_threshold(results(), 0.0, [0.0, 0.3, 0.5, 0.7])["threshold"] == 0.7
    assert suggest_threshold(results(), 0.2, [0.0, 0.3, 0.5, 0.7])["threshold"] == 0.5


def test_suggest_threshold_none_when_unreachable():
    rs = [Result(case({"reject": True}), plan("stop", 0.0))]
    assert suggest_threshold(rs, 0.0, [0.0, 0.5, 0.95]) is None


def test_run_uses_fixed_context_and_restores_it():
    seen = {}

    class Prov:
        name, model, staged = "fake", "1", False

        def answer(self, state, questions):
            seen["apps"], seen["front"] = state["apps"], state["frontmost_app"]
            return {k: ({"choice": next(iter(q["criteria"])), "confidence": 0.9} if q["type"] == "choice" else {"noul": 0.0})
                    for k, q in questions.items()}

    before = actions.installed_apps, actions.frontmost_app
    rs = evaluate.run(Brain(Prov()), CASES[:2], CTX, warmup=False)
    assert len(rs) == 2
    assert seen == {"apps": CTX["apps"], "front": CTX["frontmost_app"]}
    assert (actions.installed_apps, actions.frontmost_app) == before


def test_report_renders_and_sweeps_only_on_dev():
    dev = evaluate.report(results(), 0.35, "dev", 0.02, "failures")
    test = evaluate.report(results(), 0.35, "test", 0.02, "failures")
    assert "lowest threshold" in dev or "no threshold" in dev
    assert "threshold" in test and "lowest threshold" not in test and "no threshold" not in test
