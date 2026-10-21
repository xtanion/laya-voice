"""Offline contracts for the single-command decision path. No paid APIs, no macOS calls."""

import json
import os
from unittest.mock import Mock

import httpx
import pytest

from laya_voice import actions, brain, config, main, providers
from laya_voice.brain import SHORTCUT_CRITERIA, SHORTCUT_GROUPS, STAGE_QUESTIONS, Brain, block, shortlist_apps, stage_keys, text_candidates
from laya_voice.providers import JevProvider, LayaProvider, ProviderError, parse_blocked, validate_answers

APPS = ["Google Chrome", "Notes", "Spotify"]


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self, picks=None, conf=0.9, raw=None, exc=None, mutate=None, staged=False):
        self.staged = staged
        self.picks = picks or {}
        self.mutate = mutate
        self.conf = conf
        self.raw = raw
        self.exc = exc
        self.calls = []

    def answer(self, state, questions):
        self.calls.append((state, questions))
        if self.exc:
            raise self.exc
        if self.raw is not None:
            return self.raw
        out = {}
        for key, q in questions.items():
            if q["type"] == "choice":
                out[key] = {"choice": self.picks.get(key, next(iter(q["criteria"]))), "confidence": self.conf}
            else:
                out[key] = {"noul": self.picks.get(key, 0.0)}
        if self.mutate:
            self.mutate(out)
        return out


@pytest.fixture(autouse=True)
def no_macos(monkeypatch):
    monkeypatch.setattr(actions, "installed_apps", lambda: list(APPS))
    monkeypatch.setattr(actions, "frontmost_app", lambda: "Spotify")


def plan_for(utterance, **kw):
    return Brain(FakeProvider(**kw)).evaluate(utterance)


def test_pause_the_music_selects_media_play_pause():
    p = plan_for("pause the music", picks={"action": "media", "media_op": "play_pause"})
    assert (p.action, p.args["op"], p.error) == ("media", "play_pause", None)
    assert main.execute(p, dry=True).startswith("[dry] media(")


@pytest.mark.parametrize("utterance,picks,arg", [
    ("next song", {"action": "media", "media_op": "next"}, ("op", "next")),
    ("volume up", {"action": "volume", "volume_op": "up"}, ("op", "up")),
    ("go back", {"action": "shortcut", "shortcut": "browser_back"}, ("shortcut", "browser_back")),
    ("open spotify", {"action": "open_app", "app": "Spotify"}, ("app", "Spotify")),
])
def test_actions_map_to_typed_args(utterance, picks, arg):
    p = plan_for(utterance, picks=picks)
    assert p.action == picks["action"]
    assert p.args[arg[0]] == arg[1]


def test_text_payload_comes_from_code_cut_candidates():
    prov = FakeProvider(picks={"action": "type_text", "text": "c0", "submit": 0.9})
    p = Brain(prov).evaluate("type hello world and hit enter")
    state, _ = prov.calls[0]
    assert p.args["text"] == state["candidates"]["c0"] == "hello world"
    assert p.args["submit"] is True


def test_state_carries_utterance_apps_and_front():
    prov = FakeProvider(picks={"action": "none"})
    Brain(prov).evaluate("hmm")
    state, questions = prov.calls[0]
    assert state["utterance"] == "hmm"
    assert state["apps"] == APPS
    assert state["frontmost_app"] == "Spotify"
    assert set(questions["app"]["criteria"]) == {*APPS, "none"}


def test_low_confidence_is_not_executed():
    p = plan_for("pause the music", picks={"action": "media", "media_op": "play_pause"}, conf=0.2)
    assert p.confidence == pytest.approx(0.2)
    assert main.execute(p, dry=True) == "Not sure what you meant."


def test_plan_confidence_is_the_minimum_of_used_answers():
    p = plan_for("pause", picks={"action": "media", "media_op": "play_pause"}, mutate=lambda a: a["media_op"].update(confidence=0.4))
    assert p.confidence == pytest.approx(0.4)


@pytest.mark.parametrize("mutate,reason", [
    (lambda a: a.pop("media_op"), "missing answer: media_op"),
    (lambda a: a["action"].update(choice="rm_rf"), "not offered"),
    (lambda a: a["action"].update(confidence=1.5), "bad confidence"),
    (lambda a: a["action"].update(confidence="high"), "bad confidence"),
    (lambda a: a["action"].update(confidence=float("nan")), "bad confidence"),
    (lambda a: a["compound"].update(noul=True), "bad noul"),
    (lambda a: a.__setitem__("submit", "yes"), "missing answer: submit"),
])
def test_invalid_answers_fail_closed(mutate, reason):
    p = plan_for("pause the music", picks={"action": "media", "media_op": "play_pause"}, mutate=mutate)
    assert p.action == "none"
    assert p.confidence == 0.0
    assert reason in p.error


def test_non_object_answers_fail_closed():
    p = plan_for("pause", raw=["media"])
    assert p.error == "answers is not an object"


def test_provider_error_becomes_rejected_plan():
    p = plan_for("pause", exc=ProviderError("timeout"))
    assert (p.action, p.error) == ("none", "timeout")
    assert str(p).startswith("rejected: timeout")


def test_rejected_plan_never_reaches_executor(monkeypatch):
    executed = Mock()
    monkeypatch.setattr(main, "execute", executed)
    monkeypatch.setattr(main, "OVERLAY", Mock())
    b = Brain(FakeProvider(exc=ProviderError("down")))
    assert main.handle(b, Mock(), "pause the music", dry=True) is True
    executed.assert_not_called()


def test_validate_rejects_unknown_question_type():
    with pytest.raises(ProviderError, match="unknown question type"):
        validate_answers({"x": {"type": "number"}}, {"x": {"value": 1}})


def test_brain_label_names_provider():
    assert Brain(FakeProvider()).label == "fake fake-1"


def jev_with(handler):
    return JevProvider(api_key="k", model="jev-test", url="https://jev.test/v1", http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_jev_provider_posts_fanout_and_returns_answers():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": {"action": {"choice": "none", "confidence": 1.0}}})

    out = jev_with(handler).answer({"utterance": "hi"}, {"action": {"type": "choice", "criteria": {"none": None}}})
    assert out == {"action": {"choice": "none", "confidence": 1.0}}
    assert seen["body"]["model"] == "jev-test"
    assert seen["body"]["state"] == {"utterance": "hi"}
    assert "action" in seen["body"]["questions"]


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="boom"),
    httpx.Response(200, text="not json"),
    httpx.Response(200, json={"no_answers": {}}),
])
def test_jev_provider_errors_raise_provider_error(response):
    with pytest.raises(ProviderError):
        jev_with(lambda request: response).answer({}, {})


def test_jev_provider_network_error_raises_provider_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError):
        jev_with(handler).answer({}, {})


def test_jev_provider_requires_key(monkeypatch):
    monkeypatch.setattr(config, "TYPESAFE_API_KEY", "")
    with pytest.raises(SystemExit):
        JevProvider(http=httpx.Client())


def test_make_provider_rejects_unknown_name():
    with pytest.raises(SystemExit, match="Unknown DECISION_PROVIDER"):
        providers.make_provider("gpt")


def test_make_provider_selects_registered(monkeypatch):
    monkeypatch.setitem(providers.PROVIDERS, "fake", FakeProvider)
    assert isinstance(providers.make_provider("FAKE"), FakeProvider)
    assert isinstance(brain.Brain(providers.make_provider("fake")).provider, FakeProvider)


def staged_plan(utterance, picks, blocked=None):
    prov = FakeProvider(picks=picks, staged=True)
    prov.blocked = blocked or set()
    return Brain(prov).evaluate(utterance), prov


def test_staged_asks_action_first_then_only_its_questions():
    p, prov = staged_plan("pause the music", {"action": "media", "media_op": "play_pause"})
    (s1, q1), (s2, q2) = prov.calls
    assert list(q1) == ["action"]
    assert set(q2) == {"media_op", "addressed"}
    assert s1 == s2 == {"utterance": "pause the music"}
    assert (p.action, p.args["op"], p.error) == ("media", "play_pause", None)


@pytest.mark.parametrize("action", sorted(STAGE_QUESTIONS))
def test_staged_plans_build_for_every_action(action):
    p, prov = staged_plan("do the thing", {"action": action, "app": "Notes", "site": "other", "text": "c0"})
    assert p.error is None
    assert len(prov.calls) == (3 if action == "shortcut" else 2)


@pytest.mark.parametrize("action", ["none", "stop", "screenshot"])
def test_staged_actions_without_args_ask_only_gates(action):
    p, prov = staged_plan("hmm", {"action": action})
    assert set(prov.calls[1][1]) == {"addressed"}
    assert p.action == action


def test_staged_in_app_reads_app():
    p, _ = staged_plan("new note in the notes app", {"action": "new_item", "new_kind": "item", "in_app": 0.9, "app": "Notes"})
    assert p.args["in_app"] == "Notes"


def test_staged_website_fallback_to_search_has_its_answers():
    p, _ = staged_plan("go to", {"action": "open_website", "site": "other", "engine": "google", "text": "c0"})
    assert p.action == "web_search"
    assert p.args["engine"] == "google"


def test_staged_shortlists_apps(monkeypatch):
    monkeypatch.setattr(actions, "installed_apps", lambda: [f"App {i}" for i in range(50)] + ["Spotify"])
    monkeypatch.setattr(config, "LAYA_MAX_APPS", 4)
    _, prov = staged_plan("open spotify", {"action": "open_app", "app": "Spotify"})
    crit = prov.calls[1][1]["app"]["criteria"]
    assert "Spotify" in crit and "none" in crit
    assert len(crit) == 5


def test_staged_second_round_error_fails_closed():
    prov = FakeProvider(picks={"action": "media"}, staged=True)
    real = FakeProvider.answer

    def answer(state, questions):
        if "action" not in questions:
            raise ProviderError("round two")
        return real(prov, state, questions)

    prov.answer = answer
    p = Brain(prov).evaluate("pause")
    assert (p.action, p.error) == ("none", "round two")


@pytest.mark.parametrize("utterance,app", [
    ("open spotify", "Spotify"),
    ("switch to chrome", "Google Chrome"),
    ("open system settings", "System Settings"),
    ("new note in the notes app", "Notes"),
])
def test_shortlist_apps_keeps_the_named_app(utterance, app):
    apps = ["Calculator", "Google Chrome", "Mail", "Notes", "Photos", "Safari", "Spotify", "System Settings", "Terminal", "Xcode"]
    assert app in shortlist_apps(utterance, apps, 3)


def test_shortlist_apps_respects_k_and_empty_utterance():
    apps = ["A", "B", "C"]
    assert shortlist_apps("", apps, 2) == ["A", "B"]


class FakeLaya:
    def __init__(self, answers=None, exc=None):
        self.answers, self.exc, self.calls = answers, exc, []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.exc:
            raise self.exc
        return {"answers": self.answers, "routing": {"model": "english"}}


def test_laya_provider_returns_answers_and_is_staged():
    agent = FakeLaya({"action": {"choice": "none", "confidence": 0.8}})
    prov = LayaProvider(agent=agent)
    assert prov.staged is True
    assert prov.model == config.LAYA_REPO + (f"/{config.LAYA_SUBFOLDER}" if config.LAYA_SUBFOLDER else "")
    assert prov.answer({"utterance": "x"}, {"action": {}}) == {"action": {"choice": "none", "confidence": 0.8}}
    assert agent.calls == [({"utterance": "x"}, {"action": {}})]


def test_laya_provider_names_subfolder():
    assert LayaProvider(subfolder="typed-decisions", agent=FakeLaya()).model.endswith("/typed-decisions")


def test_laya_provider_errors_raise_provider_error():
    agent = FakeLaya(exc=RuntimeError("mps oom"))
    with pytest.raises(ProviderError, match="laya: mps oom"):
        LayaProvider(agent=agent).answer({}, {})


def test_laya_provider_missing_answers_raises_provider_error():
    agent = Mock(predict=Mock(return_value={"routing": {}}))
    with pytest.raises(ProviderError, match="laya"):
        LayaProvider(agent=agent).answer({}, {})


def test_laya_provider_missing_package(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "laya":
            raise ImportError(name)
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit, match="uv sync --extra laya"):
        LayaProvider()


@pytest.mark.skipif(not os.environ.get("LAYA_LIVE"), reason="set LAYA_LIVE=1 to load the real Laya checkpoint")
@pytest.mark.parametrize("utterance,action", [
    ("pause the music", "media"),
    ("turn the volume up", "volume"),
    ("type hello world and hit enter", "type_text"),
    ("what time is dinner tonight", "none"),
])
def test_laya_live(utterance, action, laya_provider):
    p = Brain(laya_provider).evaluate(utterance)
    assert p.error is None
    assert p.action == action


@pytest.fixture(scope="module")
def laya_provider():
    return LayaProvider()


def test_staged_shortcut_picks_group_then_key_within_it():
    p, prov = staged_plan("go back", {"action": "shortcut", "shortcut_group": "browser", "shortcut": "browser_back"})
    assert list(prov.calls[1][1]) == ["shortcut_group", "addressed"]
    assert set(prov.calls[2][1]["shortcut"]["criteria"]) == set(SHORTCUT_GROUPS["browser"][1])
    assert (p.action, p.args["shortcut"]) == ("shortcut", "browser_back")


def test_staged_shortcut_outside_group_fails_closed():
    p, _ = staged_plan("go back", {"action": "shortcut", "shortcut_group": "keys", "shortcut": "browser_back"})
    assert p.action == "none"
    assert "not offered" in p.error


def test_staged_shortcut_confidence_includes_group():
    prov = FakeProvider(picks={"action": "shortcut", "shortcut_group": "browser", "shortcut": "reload"}, staged=True,
                        mutate=lambda a: a.get("shortcut_group", {}).update(confidence=0.3))
    assert Brain(prov).evaluate("reload").confidence == pytest.approx(0.3)


def test_shortcut_groups_partition_all_shortcuts():
    keys = [k for _, ks in SHORTCUT_GROUPS.values() for k in ks]
    assert len(keys) == len(set(keys))
    assert set(keys) == set(SHORTCUT_CRITERIA)


def test_unstaged_never_sees_staged_only_questions():
    prov = FakeProvider(picks={"action": "shortcut", "shortcut": "reload"})
    Brain(prov).evaluate("reload")
    assert len(prov.calls) == 1
    assert "shortcut_group" not in prov.calls[0][1]


@pytest.mark.parametrize("utterance,asked", [
    ("type hello", False),
    ("type hello in notes", True),
    ("new note into the notes app", True),
])
def test_in_app_asked_only_when_utterance_names_a_place(utterance, asked):
    keys = stage_keys("type_text", utterance)
    assert ("in_app" in keys) is asked
    assert ("app" in keys) is asked


def test_in_app_never_asked_for_actions_that_ignore_it():
    assert "in_app" not in stage_keys("media", "pause the music in spotify")


@pytest.mark.parametrize("utterance,asked", [("open chrome", False), ("open chrome and go to youtube", True), ("open notes then type hi", True)])
def test_compound_asked_only_with_a_conjunction(utterance, asked):
    assert ("compound" in stage_keys("open_app", utterance)) is asked


def test_blocked_options_are_withdrawn_from_staged_questions():
    _, prov = staged_plan("delete all my files", {"action": "none"}, blocked={("action", "task")})
    assert "task" not in prov.calls[0][1]["action"]["criteria"]


def test_blocked_option_selected_anyway_fails_closed():
    p, _ = staged_plan("delete all my files", {"action": "task"}, blocked={("action", "task")})
    assert p.action == "none"
    assert "not offered" in p.error


def test_block_ignores_unknown_pairs_and_keeps_input_intact():
    qs = {"system_op": {"type": "choice", "criteria": {"lock": "x", "empty_trash": "y"}}}
    out = block(qs, {("system_op", "empty_trash"), ("nope", "x"), ("system_op", "nope")})
    assert set(out["system_op"]["criteria"]) == {"lock"}
    assert set(qs["system_op"]["criteria"]) == {"lock", "empty_trash"}


def test_parse_blocked():
    assert parse_blocked(" action.task, system_op.empty_trash ,") == {("action", "task"), ("system_op", "empty_trash")}
    assert parse_blocked("") == set()
    with pytest.raises(SystemExit):
        parse_blocked("task")


def test_laya_provider_blocks_task_by_default():
    assert ("action", "task") in LayaProvider(agent=FakeLaya()).blocked


@pytest.mark.parametrize("utterance,cands", [
    ("type hello world and hit enter", ["hello world"]),
    ("search youtube for lofi hip hop", ["lofi hip hop"]),
    ("search on amazon for usb c cable", ["usb c cable"]),
    ("search for amazon prime deals", ["amazon prime deals"]),
    ("google best ramen near me", ["best ramen near me"]),
    ("make a new note called groceries", ["groceries"]),
    ("hello how are you", ["hello how are you"]),
])
def test_text_candidates_drop_command_words(utterance, cands):
    assert list(text_candidates(utterance).values()) == cands
