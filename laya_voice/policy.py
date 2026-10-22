"""Jev chooses an operation and its target; a small OpenAI-compatible model writes field values.

Port of jev-ultrafast/model.py to the desktop. One TypeSafe request per decision cycle carries:

* ``operation``      Choice over the operations that are actually possible on this screen.
* ``click_target``   Choice over the observed clickable elements (speculative).
* ``type_text_target`` Choice over the observed editable fields (speculative).
* ``open_app_target`` Choice over installed applications (speculative).
* ``press_key_target`` Choice over a small closed key set (speculative).
* ``text_value``     Only without a text model: Choice over spans cut from the goal by code,
                     so the typed value is *selected*, never generated.

The executor consumes only the head that matches the chosen operation.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any

import httpx

from . import config
from .questions import APP_TARGET, KEY_TARGET, NEXT_ACTION, TARGET, TEXT_SELECT, TEXT_VALUE

CLIENT = httpx.Client(http2=False, timeout=25)

OPERATIONS = {"click": "CLICK", "fill": "TYPE_TEXT", "open_app": "OPEN_APP", "key": "PRESS_KEY"}
LABELS = {
    "CLICK": "Click a visible button, link, checkbox, radio, tab, menu item, row, or field on the current screen.",
    "TYPE_TEXT": "Enter text into a visible editable field. The value is supplied separately from the goal.",
    "OPEN_APP": "Open or switch to a different installed application; the screen then shows that app.",
    "PRESS_KEY": "Press one key: enter, escape, tab, arrows, backspace, or command-enter.",
}


def post_json(url: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(3):
        try:
            response = CLIENT.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def validate_choice(answer: Any, ids: Any) -> dict[str, Any]:
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def action_space(actions: list[dict[str, Any]]) -> tuple[list[dict], dict[str, dict[str, dict]], dict[str, dict]]:
    """One index per observed element; each operation has its own valid target choices.

    Screen elements (click / fill) share one index per AX node. Apps and keys are targets too,
    keyed by their own names, so the model can only ever pick something the executor offered.
    """
    elements: list[dict[str, Any]] = []
    indices: dict[int, str] = {}
    targets: dict[str, dict[str, dict]] = {}
    controls: dict[str, dict] = {}
    for action in actions:
        kind = action["kind"]
        if kind not in OPERATIONS:
            controls[action["id"].upper()] = action
            continue
        operation = OPERATIONS[kind]
        group = targets.setdefault(operation, {})
        if kind == "open_app":
            group[action["app"]] = action
            continue
        if kind == "key":
            group[action["key"]] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "subrole", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].removeprefix("Focus "), operations=[])
            elements.append(element)
        index = indices[node]
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        group[index] = action
    return elements, targets, controls


_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"[$€£]?\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|K|CAD|USD|EUR))?")
_LABELLED = re.compile(
    r"\b(?:postal\s*code|postcode|zip(?:\s*code)?|email|e-mail|username|user\s*name|password|phone|search\s+(?:term|query)|query|keyword|address|name)"
    r"\s*(?:is|:|=|of)?\s*(?P<v>[A-Za-z0-9@._+-]+(?:\s+[A-Za-z0-9@._+-]+){0,3}?)(?=[,.;]|\s+(?:and|then|in|on|for|to)\b|$)",
    re.I,
)
_LEAD_VERB = re.compile(
    r"^(?:please\s+)?(?:find|search(?:\s+for)?|look(?:\s+up|\s+for)?|google|show(?:\s+me)?|get(?:\s+me)?|pull\s+up|open|go\s+to)\s+(?:me\s+)?",
    re.I,
)


def text_candidates(goal: str) -> dict[str, str]:
    """Spans cut from the goal by code. Jev selects one; nothing is generated.

    Besides the single-command cuts (quoted text, 'type X', 'search for X'), every sentence of a
    multi-sentence goal is a candidate, with and without its leading verb, so a paragraph goal
    still yields a usable query such as 'a used 2020 CLA for sale in Ontario under $15,000'."""
    from .brain import text_candidates as cut

    spans: list[str] = list(cut(goal).values())

    def add(span: str) -> None:
        span = span.strip(" .,;:")
        if len(span) >= 2 and span not in spans:
            spans.append(span)

    # Labelled values: "postal code M5V 3L9", "zip 94107", "email x@y.z", "username kev", "search term ..."
    for m in _LABELLED.finditer(goal):
        add(m.group("v"))
    # Amounts, years and numbers, as written and as plain digits: "$15,000 CAD" → "$15,000", "15,000", "15000".
    for m in _NUMBER.finditer(goal):
        raw = m.group(0)
        add(raw)
        digits = re.sub(r"[^\d.]", "", raw).rstrip(".")
        if digits:
            add(digits)
            if "," in raw:
                add(raw.replace("$", "").strip())
    for sentence in _SENTENCE.split(goal.strip()):
        sentence = sentence.strip(" .")
        if len(sentence) < 3:
            continue
        add(sentence)
        add(_LEAD_VERB.sub("", sentence))
        # Comma clauses of a long sentence are often the individual field values.
        if "," in sentence:
            for clause in sentence.split(","):
                add(_LEAD_VERB.sub("", clause))
    return {f"c{i}": span for i, span in enumerate(spans[:18])}


STAGED_TARGETS = 8
STAGED_OPERATIONS = {
    "CLICK": "click a button, link, tab, video, or other control on the screen",
    "TYPE_TEXT": "type text into a field",
    "OPEN_APP": "open or switch to a different application",
    "PRESS_KEY": "press a key such as enter, escape, tab, or an arrow",
    "SCROLL_DOWN": "scroll down to find something not visible yet",
    "SCROLL_UP": "scroll up to find something not visible yet",
    "WAIT": "wait for the page or window to finish loading",
    "BLOCKED": "nothing on this screen can make progress",
}
_STAGED_DONE_RULES = "Do `recent_actions` already accomplish everything `goal` asks for?"
_STAGED_CLICK_RULES = "Is clicking one of `matching_controls` the next step to do `goal`?"
_STAGED_OPERATION_RULES = ("`goal` is what the user asked the computer to do. `recent_actions` is what has been done so far, "
                           "`app` and `window` what is in front now, and `matching_controls` the on-screen controls whose "
                           "names best match the goal. Which single next step advances the goal?")
_STAGED_TARGET_RULES = {
    "CLICK": "Which of these on-screen controls should be clicked to do `goal`?",
    "TYPE_TEXT": "Which of these fields should the text for `goal` be typed into?",
    "OPEN_APP": "Which application does `goal` need?",
    "PRESS_KEY": "Which key should be pressed to do `goal`?",
}
_WORD = re.compile(r"[a-z0-9]+")
_PROVIDER: Any = None


def _provider() -> Any:
    global _PROVIDER
    if _PROVIDER is None:
        from .providers import LayaProvider

        _PROVIDER = LayaProvider()
    return _PROVIDER


_COMMAND_WORDS = {"play", "click", "open", "watch", "tap", "press", "select", "choose", "start", "the", "this", "that",
                  "video", "song", "button", "link", "one", "please", "can", "you", "for", "and", "with", "from"}
_DURATION = re.compile(r"\s*\b\d+ (?:hours?|minutes?|seconds?)(?:,? \d+ (?:minutes?|seconds?))*\b", re.I)


def _words(text: str) -> list[str]:
    import unicodedata

    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return [w for w in _WORD.findall(plain) if len(w) > 2 or w.isdigit()]


def _shared(content: list[str], label: str) -> int:
    tokens = _words(label)
    return sum(any(t.startswith(w[:5]) or w.startswith(t[:5]) for t in tokens) for w in content)


CLICK_GOAL = re.compile(r"^(?:please\s+)?(?:click|tap|select|choose|play|watch|"
                        r"open\s+(?:the|that|this|first|second|third|fourth|fifth|last|top))\b", re.I)
_ORDINALS = {"first": 1, "1st": 1, "top": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
             "fifth": 5, "5th": 5, "last": -1}
_PLAYER_TIME = re.compile(r"\bof \d+ (?:hours?|minutes?|seconds?)", re.I)
_ORDINAL = re.compile(r"\b(" + "|".join(_ORDINALS) + r")\s+(?:video|result|one|link|song|item)s?\b", re.I)
_PLAY_GENERIC = {"music", "something", "again", "next", "previous", "track", "playback", "media"}
_ELSEWHERE = re.compile(r"\s(?:on|in)\s+([a-z][\w ]*)$", re.I)


def _names_place(text: str) -> bool:
    from . import actions
    from .brain import named_engines

    name = text.strip().lower().removeprefix("the ")
    return bool(named_engines(name)) or name.replace(" ", "_") in actions.SITES \
        or any(name == app.lower() for app in actions.installed_apps())


def ordinal_target(goal: str, acts: list[dict[str, Any]]) -> str | None:
    """The id of 'the second video': visible links that carry a duration (how YouTube labels a
    video) in reading order, one per title since thumbnail and title links repeat it."""
    m = _ORDINAL.search(goal) if CLICK_GOAL.match(goal) else None
    if not m:
        return None
    videos = [a for a in acts if a.get("kind") == "click" and a.get("rect") and _DURATION.search(a.get("label", ""))
              and not _PLAYER_TIME.search(a["label"])]
    videos.sort(key=lambda a: (round(a["rect"]["y"] / 60), a["rect"]["x"]))
    ids, seen = [], set()
    for a in videos:
        title = _DURATION.sub("", a["label"]).strip().lower()[:40]
        if len(title) >= 4 and title not in seen:
            seen.add(title)
            ids.append(a["id"])
    n = _ORDINALS[m.group(1).lower()]
    if not ids or n > len(ids):
        return None
    return ids[n - 1] if n > 0 else ids[-1]


def click_target(goal: str, labels: dict[str, str], where: str = "") -> str | None:
    """The id of the control a 'click/play X' command names on screen, or None when the command
    is not about the screen: no content words ('play'), a media command ('play music'), another
    app or site ('play X on spotify' while `where` is YouTube), or no label sharing a word."""
    if not CLICK_GOAL.match(goal):
        return None
    body = CLICK_GOAL.sub("", goal).strip().rstrip(".!?,;:")
    tail = _ELSEWHERE.search(body)
    if tail and _names_place(tail.group(1)):
        if tail.group(1).strip().lower() not in where.lower():
            return None
        body = body[:tail.start()]
    content = [w for w in _words(body) if w not in _COMMAND_WORDS]
    if not content or set(content) <= _PLAY_GENERIC or _ORDINAL.search(goal):
        return None
    from difflib import SequenceMatcher

    counts = {key: _shared(content, label) for key, label in labels.items()}
    best = max(labels, default=None,
               key=lambda key: (counts[key], SequenceMatcher(None, body.lower(), labels[key].lower()).ratio()))
    return best if best is not None and counts[best] else None


def shortlist(goal: str, labels: dict[str, str], k: int = STAGED_TARGETS) -> list[str]:
    """The ids whose labels best match the goal, at most k.

    Command words ('play', 'click', 'video') are ignored when the goal has other words, so 'play X
    by Y' matches the X link rather than the player's Play button. When some labels share several
    words with the goal, only those close to the best are kept: two Phil Collins videos, not eight
    mixed controls."""
    from difflib import SequenceMatcher

    words = _words(goal)
    content = [w for w in words if w not in _COMMAND_WORDS] or words
    counts = {key: _shared(content, labels[key]) for key in labels}
    ratio = {key: SequenceMatcher(None, goal.lower(), labels[key].lower()).ratio() for key in labels}
    ranked = sorted(labels, key=lambda key: (counts[key] + ratio[key]), reverse=True)
    best = counts[ranked[0]] if ranked else 0
    if best >= 2:
        ranked = [key for key in ranked if counts[key] >= best - 1]
    return ranked[:k]


def option_label(label: str, goal: str = "", limit: int = 60) -> str:
    """The label without durations, cut to `limit` around the first goal word it contains, so a long
    'Artist, Artist | Sunset Lake Mix' label still shows the part the user said."""
    import unicodedata

    folded = unicodedata.normalize("NFKD", label.removeprefix("Focus ")).encode("ascii", "ignore").decode()
    text = _DURATION.sub("", folded).strip()
    if len(text) <= limit:
        return text
    plain = text.lower()
    hits = [plain.find(w) for w in _words(goal) if w not in _COMMAND_WORDS and plain.find(w) >= 0]
    start = max(0, min(hits) - 12) if hits and min(hits) + 12 > limit else 0
    return ("…" if start else "") + text[start:start + limit]


def _recent(history: list[dict[str, Any]]) -> list[str]:
    return [f"{h.get('kind') or ''} {h.get('action') or ''}".strip() + ("" if h.get("screen_changed") else " (no change)")
            for h in history[-4:]]


STAGED_SURE = 0.5
STAGED_TARGET_MIN = {"CLICK": 0.4, "OPEN_APP": 0.3}


def _staged_target(provider: Any, goal: str, operation: str, group: dict[str, dict]) -> tuple[str, dict, str | None, dict]:
    from .brain import shortlist_apps

    if operation == "OPEN_APP":
        criteria: dict[str, Any] = {name: None for name in shortlist_apps(goal, list(group), STAGED_TARGETS)}
    elif operation == "PRESS_KEY":
        keep = shortlist(goal, {key: f"{key}: {a['label']}" for key, a in group.items()})
        criteria = {key: group[key]["label"] for key in keep}
    else:
        keep = shortlist(goal, {i: a["label"].removeprefix("Focus ") for i, a in group.items()})
        criteria = {i: f"{option_label(group[i]['label'], goal)} ({group[i].get('role', '')})" for i in keep}
    asked: dict[str, Any] = {"target": {"type": "choice", "criteria": criteria, "instructions": _STAGED_TARGET_RULES[operation]}}
    cands: dict[str, str] = {}
    if operation == "TYPE_TEXT":
        cands = dict(list(text_candidates(goal).items())[:STAGED_TARGETS])
        if cands:
            asked["text_value"] = {"type": "choice", "criteria": cands, "instructions": TEXT_SELECT}
    answers = provider.answer({"goal": goal}, asked)
    target_answer = validate_choice(answers.get("target", {}), criteria)
    text = cands[validate_choice(answers.get("text_value", {}), cands)["choice"]] if cands else None
    return target_answer["choice"], target_answer, text, {k: answers.get(k) for k in asked}


def choose_staged(state: dict[str, Any], goal: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    """Laya edition of `choose`: Laya reads at most 512 tokens and a few options per question, so instead
    of one request over the whole screen it answers the operation first, then a target from a short list."""
    provider = _provider()
    elements, targets, controls = action_space(state["actions"])
    operations = {k: v for k, v in STAGED_OPERATIONS.items()
                  if (k in targets or k in controls or k == "BLOCKED") and (k != "WAIT" or history)}
    clickable = {i: a["label"].removeprefix("Focus ") for i, a in targets.get("CLICK", {}).items()}
    view = {"goal": goal, "app": state["app"], "window": state["title"][:120], "recent_actions": _recent(history),
            "matching_controls": [option_label(clickable[i], goal, 50) for i in shortlist(goal, clickable, 4)]}
    started = time.perf_counter()
    raw: dict[str, Any] = {}
    asked: dict[str, Any] = {"operation": {"type": "choice", "criteria": operations, "instructions": _STAGED_OPERATION_RULES}}
    if history:
        # Nothing can be done before a step has run, so DONE is only a question once there is history.
        asked["done"] = {"type": "noul", "instructions": _STAGED_DONE_RULES}
    if "CLICK" in targets:
        asked["click"] = {"type": "noul", "instructions": _STAGED_CLICK_RULES}
    raw.update(provider.answer(view, asked))
    done_p = float(raw.get("done", {}).get("noul", 0.0)) if history else 0.0
    click_p = float(raw.get("click", {}).get("noul", 0.0)) if "CLICK" in targets else 0.0
    picked = validate_choice(raw.get("operation", {}), operations)
    target = target_answer = selected_text = None
    probabilities: dict[str, float] = {}
    if done_p > 0.5:
        operation_answer = {"choice": "DONE", "confidence": done_p, "probabilities": {"DONE": done_p}}
    elif picked["choice"] not in ("CLICK", "BLOCKED") and picked["confidence"] >= STAGED_SURE:
        operation_answer = picked
    else:
        operation_answer = {"choice": "CLICK", "confidence": click_p, "probabilities": {"CLICK": click_p}}
        if click_p <= 0.5:
            operation_answer = picked if picked["choice"] != "CLICK" else {"choice": "BLOCKED", "confidence": 1 - click_p,
                                                                            "probabilities": {"BLOCKED": 1 - click_p}}
    operation = operation_answer["choice"]
    tried: set[str] = set()
    while operation in targets and operation not in tried:
        tried.add(operation)
        target, target_answer, selected_text, raw_target = _staged_target(provider, goal, operation, targets[operation])
        raw.update(raw_target)
        if target_answer["confidence"] >= STAGED_TARGET_MIN.get(operation, 0.0):
            break
        # A weak best match means the thing the goal names is not on this screen (or not installed).
        fallback = picked["choice"] if picked["choice"] not in tried | {"CLICK"} else "BLOCKED"
        operation, target, target_answer, selected_text = fallback, None, None, None
        operation_answer = {"choice": fallback, "confidence": picked["probabilities"].get(fallback, 1.0),
                            "probabilities": picked["probabilities"]}
    if operation in targets and target is not None:
        group = targets[operation]
        choice = group[target]["id"]
        probabilities = {group[k]["id"]: v for k, v in target_answer["probabilities"].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"].get(operation, operation_answer["confidence"])
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "selected_text": selected_text,
        "raw_answers": raw,
        "model": provider.model,
        "usage": {},
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": {"state": view},
    }


def choose(state: dict[str, Any], goal: str, history: list[dict[str, Any]], *, text_model: bool | None = None) -> dict[str, Any]:
    if config.DECISION_PROVIDER == "laya":
        return choose_staged(state, goal, history)
    elements, targets, controls = action_space(state["actions"])
    operations = {key: LABELS[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions: dict[str, Any] = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        if operation == "OPEN_APP":
            questions["open_app_target"] = {
                "type": "choice",
                "criteria": {name: None for name in candidates},
                "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, APP_TARGET]},
            }
        elif operation == "PRESS_KEY":
            questions["press_key_target"] = {
                "type": "choice",
                "criteria": {key: a["label"] for key, a in candidates.items()},
                "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, KEY_TARGET]},
            }
        else:
            questions[operation.lower() + "_target"] = {
                "type": "choice",
                "criteria": {
                    index: {
                        "element": f"[{index}] {a['label']}",
                        "current_value": a.get("value", ""),
                        **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                    }
                    for index, a in candidates.items()
                },
                "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
            }
    use_llm = bool(os.environ.get("TEXT_MODEL_API_KEY")) if text_model is None else text_model
    candidates: dict[str, str] = {}
    if "TYPE_TEXT" in targets and not use_llm:
        candidates = text_candidates(goal)
        questions["text_value"] = {"type": "choice", "criteria": dict(candidates), "instructions": TEXT_SELECT}
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", config.JEV_MODEL),
        "state": {
            "screen": {"app": state["app"], "window": state["title"], "text": state["text"]},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "screen_changed")} for h in history[-10:]
            ],
            **({"candidates": candidates} if candidates else {}),
        },
        "questions": questions,
    }
    started = time.perf_counter()
    key = os.environ.get("TYPESAFE_API_KEY") or config.TYPESAFE_API_KEY
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set (put it in .env)")
    result = post_json(config.TYPESAFE_URL, key, body)
    answers = result["answers"]
    operation_answer = validate_choice(answers.get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities: dict[str, float] = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate only the head the operation selected.
        head = operation.lower() + "_target"
        target_answer = validate_choice(answers.get(head, {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    selected_text = None
    if operation == "TYPE_TEXT" and candidates:
        text_answer = validate_choice(answers.get("text_value", {}), candidates)
        selected_text = candidates[text_answer["choice"]]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "selected_text": selected_text,
        "raw_answers": answers,
        "model": result.get("model"),
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal: str, action: dict[str, Any], screen: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "screen": {"app": screen["app"], "window": screen["title"], "text": screen["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Generate a field value with the small text model. Requires TEXT_MODEL_API_KEY."""
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY (or a Jev-selected candidate); nothing is guessed by the executor.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "inception/mercury-2.5")
    reasoning: dict[str, Any] = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {"role": "user", "content": json.dumps(context)},
            ],
        },
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
