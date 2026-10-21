"""Offline contracts for the single-command decision path. No paid APIs, no macOS calls."""

import json
from unittest.mock import Mock

import httpx
import pytest

from laya_voice import actions, brain, config, main, providers
from laya_voice.brain import Brain
from laya_voice.providers import JevProvider, ProviderError, validate_answers

APPS = ["Google Chrome", "Notes", "Spotify"]


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self, picks=None, conf=0.9, raw=None, exc=None, mutate=None):
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
