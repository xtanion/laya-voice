"""Offline contracts for the training-data builder and paraphrase cleaning. No models, no API."""

import random
from collections import Counter

import pytest

from laya_voice import actions, config, paraphrase, training
from laya_voice.brain import SHORTCUT_GROUPS, Brain
from laya_voice.providers import parse_blocked

BLOCKED = parse_blocked(config.LAYA_BLOCKED)
INSTALLED = ["Google Chrome", "Notes", "Slack", "Spotify", "Terminal", "TextEdit", "Visual Studio Code", "System Settings", "Mail", "Zoom"]


class Recorder:
    """Staged provider that answers with a row's labels and records every question it is asked."""

    name, model, staged = "rec", "1", True

    def __init__(self, expected):
        self.expected, self.asked, self.blocked = expected, {}, BLOCKED

    def answer(self, state, questions):
        out = {}
        for k, q in questions.items():
            self.asked[k] = q
            if q["type"] == "choice":
                out[k] = {"choice": self.expected.get(k, next(iter(q["criteria"]))), "confidence": 0.9}
            else:
                out[k] = {"noul": 1.0 if self.expected.get(k) else 0.0}
        return out


@pytest.fixture
def fixed_apps(monkeypatch):
    monkeypatch.setattr(actions, "installed_apps", lambda: list(INSTALLED))
    monkeypatch.setattr(actions, "frontmost_app", lambda: "Notes")


PARITY = [
    ("pause the song", {"action": "media", "op": "play_pause"}),
    ("open slack", {"action": "open_app", "app": "Slack"}),
    ("refresh the page", {"action": "shortcut", "shortcut": "reload"}),
    ("type see you soon and press enter", {"action": "type_text", "text": "see you soon", "submit": True}),
    ("type eggs in notes", {"action": "type_text", "text": "eggs", "submit": False, "in_app": "Notes"}),
    ("search amazon for desk lamp", {"action": "web_search", "engine": "amazon", "query": "desk lamp"}),
    ("scroll down a little", {"action": "scroll", "direction": "down", "amount": "little"}),
    ("create a note called plans", {"action": "new_item", "kind": "item", "title": "plans"}),
    ("open chrome and go to reddit", {"action": "open_app", "app": "Google Chrome", "compound": True}),
    ("stop listening", {"action": "stop"}),
    ("delete everything", {"reject": True}),
    ("how was lunch", {"reject": True, "chatter": True}),
]


@pytest.mark.parametrize("utterance,expect", PARITY, ids=[p[0] for p in PARITY])
def test_row_questions_match_what_brain_asks(utterance, expect, fixed_apps):
    row = training.build_row(utterance, expect, INSTALLED, BLOCKED)
    assert row is not None
    rec = Recorder(row["expected"])
    plan = Brain(rec).evaluate(utterance)
    assert plan.error is None
    for key, q in row["questions"].items():
        assert rec.asked[key] == q, key
    assert row["state"] == {"utterance": utterance}


def test_row_labels_are_offered_options():
    row = training.build_row("refresh the page", {"action": "shortcut", "shortcut": "reload"}, INSTALLED, BLOCKED)
    assert row["expected"]["shortcut_group"] == "browser"
    assert set(row["questions"]["shortcut"]["criteria"]) == set(SHORTCUT_GROUPS["browser"][1])
    for k, v in row["expected"].items():
        if row["questions"][k]["type"] == "choice":
            assert v in row["questions"][k]["criteria"]


def test_text_label_is_the_matching_candidate_key():
    row = training.build_row("type see you soon", {"action": "type_text", "text": "see you soon", "submit": False}, INSTALLED, BLOCKED)
    key = row["expected"]["text"]
    assert row["questions"]["text"]["criteria"][key] == "see you soon"
    assert "submit" not in row["expected"]


def test_text_question_skipped_when_no_candidate_matches():
    misses = Counter()
    row = training.build_row("find desk lamp on amazon", {"action": "web_search", "engine": "amazon", "query": "desk lamp"},
                             INSTALLED, BLOCKED, misses)
    assert "text" not in row["expected"]
    assert row["expected"]["engine"] == "amazon"
    assert misses["text_not_a_candidate"] == 1


def test_stop_without_stop_phrase_is_unrenderable():
    misses = Counter()
    assert training.build_row("halt", {"action": "stop"}, INSTALLED, BLOCKED, misses) is None
    assert misses["action_not_offered:stop"] == 1


def test_blocked_action_is_unrenderable():
    assert training.build_row("do my taxes", {"action": "task"}, INSTALLED, BLOCKED) is None


def test_rejects_are_labelled_none_and_chatter_unaddressed():
    unsafe = training.build_row("delete everything", {"reject": True}, INSTALLED, BLOCKED)
    chat = training.build_row("how was lunch", {"reject": True, "chatter": True}, INSTALLED, BLOCKED)
    assert unsafe["expected"] == {"action": "none"}
    assert chat["expected"] == {"action": "none", "addressed": False}


def test_app_not_installed_is_labelled_none():
    row = training.build_row("open photoshop", {"action": "open_app", "app": "none"}, INSTALLED, BLOCKED)
    assert row["expected"]["app"] == "none"


def test_near_duplicate():
    protected = [training._norm("pause the music")]
    assert training.near_duplicate("Pause the music.", protected)
    assert training.near_duplicate("pause the musics", protected)
    assert not training.near_duplicate("skip this track", protected)


def test_templates_cover_every_offered_action():
    families = {expect.get("action", "none") for _, _, expect in training.template_utterances(random.Random(0))}
    assert families >= {"media", "volume", "system", "screenshot", "stop", "scroll", "shortcut", "open_app", "open_website",
                        "web_search", "type_text", "new_item", "open_folder"}


def test_shortcut_phrases_use_known_shortcuts():
    assert set(training.SHORTCUT_PHRASES) <= {k for _, ks in SHORTCUT_GROUPS.values() for k in ks}


@pytest.fixture(scope="module")
def built():
    return training.build(seed=0, paraphrases=False)


def test_build_keeps_eval_cases_out(built):
    rows, stats = built
    from laya_voice import evaluate
    protected = [training._norm(c.utterance) for c in evaluate.load_cases()]
    assert not [r for r in rows if training.near_duplicate(r["state"]["utterance"], protected)]
    assert stats["dropped"]["dropped_near_eval"] > 0


def test_build_rows_are_unique_and_deterministic(built):
    rows, _ = built
    assert len({r["state"]["utterance"] for r in rows}) == len(rows)
    again, _ = training.build(seed=0, paraphrases=False)
    assert [r["id"] for r in again] == [r["id"] for r in rows]


def seed(family="type_text.plain.x", expect=None, keep=None):
    return paraphrase.Seed(family, "s", expect or {"action": "type_text", "text": "buy milk", "submit": False}, "m", keep=keep or ["buy milk"])


def test_clean_normalizes_and_enforces_keep_words():
    s = seed()
    assert paraphrase.clean(s, ["Type: Buy Milk!", "write groceries", "type buy milk"]) == ["type buy milk"]


def test_clean_enforces_submit_gate():
    plain = seed()
    submit = seed(expect={"action": "type_text", "text": "buy milk", "submit": True})
    phr = ["type buy milk", "type buy milk and press enter"]
    assert paraphrase.clean(plain, phr) == ["type buy milk"]
    assert paraphrase.clean(submit, phr) == ["type buy milk and press enter"]


def test_clean_enforces_stop_gate():
    s = paraphrase.Seed("stop", "stop listening", {"action": "stop"}, "m")
    assert paraphrase.clean(s, ["stop listening now", "shut up", "goodbye for now"]) == ["stop listening now", "goodbye for now"]


def test_seeds_are_unique_and_labels_valid():
    ss = paraphrase.seeds()
    assert len({s.family for s in ss}) == len(ss)
    for s in ss:
        assert s.expect.get("reject") or s.expect["action"] in Brain.__new__(Brain)._questions({"c0": "x"}, [])["action"]["criteria"]
        for k in s.keep:
            assert k in s.seed or k in s.meaning


def test_prompt_mentions_keep_words_and_count():
    s = seed()
    text = paraphrase.prompt(s)
    assert "'buy milk'" in text and "Write 25" in text
