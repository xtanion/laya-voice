"""Training data for fine-tuning Laya on the staged decision path.

    uv run python -m laya_voice.training                 # write data/train/v1.jsonl and print stats
    uv run python -m laya_voice.training --no-paraphrases

An intent is a label in the eval `expect` shape plus the extra labels training needs
(`amount`, `submit`, `kind`, `title`, `in_app`, `compound`). Each labelled utterance
becomes one laya-train row `{state, questions, expected}` whose questions are rendered
exactly as Brain renders them for a staged provider (`for_staged`, `stage_keys`, the
shortcut group round, the app shortlist), so training and inference text match.
Utterances too close to any eval case are dropped (see `near_duplicate`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator

from . import config, evaluate
from .brain import QUESTIONS_VERSION, SHORTCUT_GROUPS, Brain, for_staged, questions_fingerprint, shortlist_apps, stage_keys, text_candidates
from .providers import parse_blocked

TRAIN = config.ROOT / "data" / "train"
PARAPHRASES = TRAIN / "paraphrases.v1.jsonl"
OUT = TRAIN / "v1.jsonl"
DUPLICATE_RATIO = 0.85

# Installed-app pool for randomised shortlists; targets are always added when missing.
APP_POOL = [
    "1Password", "Activity Monitor", "Affinity Photo", "Alfred", "App Store", "Arc", "Audacity", "Automator", "Bear",
    "Blender", "Books", "Brave Browser", "Calculator", "Calendar", "Chess", "ChatGPT", "Claude", "Contacts", "Cursor",
    "Dictionary", "Discord", "Docker", "Dropbox", "FaceTime", "Figma", "Finder", "Firefox", "GarageBand", "GitHub Desktop",
    "Google Chrome", "Home", "iMovie", "iTerm", "Keynote", "Linear", "Logic Pro", "Mail", "Maps", "Messages",
    "Microsoft Excel", "Microsoft Outlook", "Microsoft Teams", "Microsoft Word", "Music", "News", "Notes", "Notion",
    "Numbers", "Obsidian", "OBS", "Pages", "Photos", "Podcasts", "Postman", "Preview", "QuickTime Player", "Raycast",
    "Reminders", "Safari", "Shortcuts", "Signal", "Simulator", "Sketch", "Slack", "Spotify", "Stocks", "Sublime Text",
    "System Settings", "Telegram", "Terminal", "TextEdit", "TablePlus", "Things", "TV", "Visual Studio Code", "VLC",
    "Voice Memos", "Warp", "Weather", "WhatsApp", "Xcode", "Zed", "Zoom",
]
# Spoken forms that differ from the app name.
APP_SPOKEN = {
    "Google Chrome": ["chrome", "google chrome"], "Visual Studio Code": ["vs code", "visual studio code", "vscode"],
    "System Settings": ["settings", "system settings", "system preferences"], "iTerm": ["iterm", "i term"],
    "Microsoft Word": ["word", "microsoft word"], "Microsoft Excel": ["excel"], "Microsoft Teams": ["teams"],
    "Microsoft Outlook": ["outlook"], "QuickTime Player": ["quicktime"], "Activity Monitor": ["activity monitor"],
    "Brave Browser": ["brave"], "App Store": ["the app store", "app store"], "TV": ["the tv app"],
}
OPEN_APP_TARGETS = ["Spotify", "Google Chrome", "Slack", "Notes", "Terminal", "Visual Studio Code", "System Settings", "Safari",
                    "Mail", "Messages", "Calendar", "Finder", "Zoom", "Discord", "Figma", "Notion", "Obsidian", "Xcode",
                    "Cursor", "Photos", "Music", "Microsoft Word", "Microsoft Excel", "WhatsApp", "Calculator", "Preview",
                    "iTerm", "Reminders", "Maps", "TextEdit", "Firefox", "Arc", "Telegram", "Linear", "Postman"]
NOT_INSTALLED = ["photoshop", "premiere", "final cut", "illustrator", "lightroom", "steam", "minecraft", "tableau"]

SITE_SPOKEN = {
    "youtube": ["youtube"], "gmail": ["gmail", "my gmail", "my email inbox"], "google_calendar": ["google calendar"],
    "google_drive": ["google drive", "my drive"], "google_docs": ["google docs"], "google_maps": ["google maps"],
    "github": ["github"], "twitter_x": ["twitter", "x dot com"], "reddit": ["reddit"], "amazon": ["amazon"],
    "netflix": ["netflix"], "chatgpt": ["chat gpt", "chatgpt"], "linkedin": ["linkedin"], "instagram": ["instagram"],
    "facebook": ["facebook"], "facebook_marketplace": ["facebook marketplace"], "wikipedia": ["wikipedia"],
    "hacker_news": ["hacker news"], "twitch": ["twitch"], "google": ["google", "google dot com"],
}
ENGINE_SPOKEN = {"google": "google", "youtube": "youtube", "amazon": "amazon", "wikipedia": "wikipedia", "github": "github",
                 "google_maps": "google maps", "reddit": "reddit", "spotify": "spotify", "perplexity": "perplexity"}
QUERIES = ["lofi hip hop", "best ramen near me", "usb c cable", "how to tie a tie", "weather tomorrow", "python list comprehension",
           "cheap flights to tokyo", "taylor swift live", "noise cancelling headphones", "banana bread recipe",
           "the french revolution", "rust async tutorial", "coffee shops open now", "mechanical keyboard", "marie curie",
           "standing desk", "how to fix a leaky faucet", "nba highlights", "jazz for studying", "react hooks",
           "running shoes", "black holes explained", "vegan lasagna", "electric toothbrush", "the roman empire",
           "chess openings for beginners", "sourdough starter", "ipad pro case", "yoga for back pain", "kubernetes basics"]
TEXTS = ["hello world", "buy milk", "see you tomorrow", "on my way", "thanks so much", "running late", "sounds good to me",
         "meeting moved to three", "happy birthday", "call me when you can", "i will be there at noon", "lunch is ready",
         "please review the pull request", "good morning team", "the build is green", "let me check and get back to you",
         "dinner at seven", "pick up the kids", "ship it", "no worries", "sorry for the delay", "great work everyone",
         "remember the dentist", "eggs and bread", "love you"]
TITLES = ["groceries", "ideas", "meeting notes", "todo", "travel plans", "book list", "workout", "recipes", "budget", "gift ideas"]
FOLDER_SPOKEN = {"downloads": ["downloads", "my downloads", "the downloads folder"], "desktop": ["the desktop folder", "my desktop folder"],
                 "documents": ["documents", "my documents", "the documents folder"], "pictures": ["pictures", "my pictures folder"],
                 "movies": ["the movies folder", "my movies"], "applications": ["applications", "the applications folder"],
                 "home": ["my home folder", "the home folder"]}

SHORTCUT_PHRASES = {
    "enter": ["press enter", "hit enter", "press return", "hit the enter key", "submit it"],
    "escape": ["press escape", "hit escape", "escape", "dismiss this", "close this popup"],
    "tab": ["press tab", "hit tab", "tab over", "next field"],
    "space": ["press space", "hit the space bar", "press the spacebar"],
    "backspace": ["backspace", "delete the last character", "press backspace", "hit delete"],
    "arrow_up": ["press up", "arrow up", "up arrow", "press the up arrow key"],
    "arrow_down": ["press down", "arrow down", "down arrow", "press the down arrow key"],
    "arrow_left": ["press left", "arrow left", "left arrow"],
    "arrow_right": ["press right", "arrow right", "right arrow"],
    "copy": ["copy", "copy this", "copy that text", "copy it to the clipboard"],
    "paste": ["paste", "paste that", "paste from the clipboard", "paste it in"],
    "cut": ["cut", "cut this", "cut the selected text"],
    "undo": ["undo", "undo the last change", "take that back", "undo last action"],
    "redo": ["redo", "redo that", "redo the last change"],
    "select_all": ["select all", "select all the text", "highlight everything"],
    "save": ["save", "save it", "save the document", "save my work"],
    "find": ["find on this page", "search this page", "open find", "find in page"],
    "close_tab_or_window": ["close this tab", "close the tab", "close this window", "close it"],
    "reopen_closed_tab": ["reopen the last tab", "reopen closed tab", "bring back the tab i closed"],
    "quit_app": ["quit this app", "quit the app", "close this application completely", "exit the app"],
    "minimize_window": ["minimize", "minimize this window", "minimize the window"],
    "hide_app": ["hide this app", "hide the app", "hide this"],
    "fullscreen": ["full screen", "go full screen", "make it full screen", "exit full screen"],
    "next_tab": ["next tab", "switch to the next tab", "go to the next tab"],
    "previous_tab": ["previous tab", "go to the previous tab", "tab to the left"],
    "browser_back": ["go back", "back", "previous page", "go back a page"],
    "browser_forward": ["go forward", "forward", "forward one page"],
    "reload": ["reload", "refresh the page", "reload this page", "refresh"],
    "address_bar": ["focus the address bar", "go to the url bar", "select the address bar"],
    "spotlight": ["open spotlight", "spotlight search", "bring up spotlight"],
    "switch_app": ["switch apps", "command tab", "go to the previous app", "switch to the last app"],
    "next_window": ["next window", "switch windows", "cycle windows"],
    "delete_word": ["delete the last word", "delete word", "remove that word"],
    "delete_line": ["delete the line", "clear this line", "delete the whole line"],
    "zoom_in": ["zoom in", "make the text bigger", "bigger text"],
    "zoom_out": ["zoom out", "make the text smaller", "smaller text"],
    "bold": ["bold", "make it bold", "bold that"],
    "italic": ["italic", "italicize that", "make it italic"],
    "send_message": ["send the message", "send this message", "send that message now"],
    "emoji_picker": ["open the emoji picker", "emoji picker", "show emojis"],
}
# `new` and `new_tab` overlap new_item; those phrasings are labelled new_item instead.

PHRASES: dict[str, tuple[dict[str, Any], list[str]]] = {
    "media.play_pause": ({"action": "media", "op": "play_pause"}, [
        "pause the music", "pause", "play", "resume", "stop the song", "pause the song", "play the music", "unpause",
        "pause playback", "resume the music", "hold the music", "stop the music", "play it again", "pause the video"]),
    "media.next": ({"action": "media", "op": "next"}, [
        "next song", "skip", "next track", "skip this song", "play the next song", "skip ahead to the next track", "next"]),
    "media.previous": ({"action": "media", "op": "previous"}, [
        "previous song", "previous track", "last song", "go back one song", "play the previous track", "back a track"]),
    "volume.up": ({"action": "volume", "op": "up"}, ["volume up", "louder", "turn it up", "turn up the volume", "raise the volume", "a bit louder"]),
    "volume.down": ({"action": "volume", "op": "down"}, ["volume down", "quieter", "turn it down", "lower the volume", "turn the volume down", "softer"]),
    "volume.mute": ({"action": "volume", "op": "mute"}, ["mute", "mute the sound", "mute audio", "silence", "mute the volume"]),
    "volume.unmute": ({"action": "volume", "op": "unmute"}, ["unmute", "unmute the sound", "turn the sound back on", "sound on"]),
    "volume.max": ({"action": "volume", "op": "max"}, ["max volume", "full volume", "volume all the way up", "maximum volume"]),
    "volume.half": ({"action": "volume", "op": "half"}, ["half volume", "set the volume to half", "medium volume"]),
    "system.lock": ({"action": "system", "op": "lock"}, ["lock the screen", "lock my mac", "lock the computer", "lock screen"]),
    "system.sleep_display": ({"action": "system", "op": "sleep_display"}, [
        "put the display to sleep", "turn off the screen", "sleep the display", "screen off"]),
    "system.show_desktop": ({"action": "system", "op": "show_desktop"}, [
        "show the desktop", "show desktop", "go to the desktop", "clear the screen and show the desktop"]),
    "system.toggle_dark_mode": ({"action": "system", "op": "toggle_dark_mode"}, [
        "toggle dark mode", "dark mode", "switch to light mode", "turn on dark mode", "turn off dark mode"]),
    "screenshot": ({"action": "screenshot"}, [
        "take a screenshot", "screenshot", "grab a screenshot", "capture the screen", "screen capture", "snap the screen"]),
    "stop": ({"action": "stop"}, ["stop listening", "exit voice control", "quit listening", "end voice control", "go to sleep",
                                  "goodbye", "bye", "stop the assistant", "exit the assistant", "stop voice control"]),
    "scroll.down": ({"action": "scroll", "direction": "down", "amount": "page"}, ["scroll down", "page down", "scroll down please", "move down the page"]),
    "scroll.down.little": ({"action": "scroll", "direction": "down", "amount": "little"}, ["scroll down a bit", "scroll down a little", "down a little"]),
    "scroll.down.a_lot": ({"action": "scroll", "direction": "down", "amount": "a_lot"}, ["scroll way down", "scroll down a lot", "scroll down a bunch"]),
    "scroll.up": ({"action": "scroll", "direction": "up", "amount": "page"}, ["scroll up", "page up", "scroll back up", "move up the page"]),
    "scroll.up.little": ({"action": "scroll", "direction": "up", "amount": "little"}, ["scroll up a bit", "scroll up a little", "up a little"]),
    "scroll.top": ({"action": "scroll", "direction": "top"}, ["scroll to the top", "go to the top", "back to the top of the page", "jump to the top"]),
    "scroll.bottom": ({"action": "scroll", "direction": "bottom"}, [
        "scroll to the bottom", "go to the bottom", "jump to the end of the page", "all the way down"]),
    "reject.chatter": ({"reject": True, "chatter": True}, [
        "what time is dinner tonight", "i told her we'd be late", "hmm let me think", "did you see the game last night",
        "okay so where was i", "yeah that sounds good", "can you believe it's already friday", "honestly i'm so tired",
        "wait what did you say", "that movie was amazing", "i'll call you back later", "no no the other one",
        "mom said she's coming over", "how was your weekend", "uh huh", "right right", "we should get pizza",
        "i think it's going to rain", "did you feed the cat", "that's hilarious", "um so anyway", "i forgot my keys again",
        "where did i put my phone", "let's talk about it tomorrow", "this coffee is cold", "he never replied to me",
        "okay okay", "i don't know man", "that's what i said", "see you at the office"]),
    "reject.unsafe": ({"reject": True}, [
        "delete all my files", "empty the trash", "empty the bin", "format the hard drive", "send fifty dollars to john",
        "reply to the last email from sam saying yes", "buy the cheapest flight to london", "transfer money to my savings",
        "delete my downloads folder", "uninstall all my apps", "shut down the computer", "restart my mac",
        "post this on twitter", "email my boss that i quit", "call mom", "order a pizza", "wipe my laptop",
        "delete this repository", "pay the electricity bill", "send my password to alex", "book a table for two tonight",
        "erase everything on the desktop", "cancel my subscription", "forward this email to everyone",
        "permanently delete my photos", "venmo sarah twenty bucks", "turn off the firewall", "change my password",
        "message the team that the meeting is cancelled", "find the cheapest flight on google flights and book it"]),
}


def _spoken(app: str) -> list[str]:
    return APP_SPOKEN.get(app, [app.lower()])


def template_utterances(rng: random.Random) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """(family, utterance, expect) from hand-written phrases and slot templates."""
    for family, (expect, phrases) in PHRASES.items():
        for p in phrases:
            yield family, p, expect
    for shortcut, phrases in SHORTCUT_PHRASES.items():
        for p in phrases:
            yield f"shortcut.{shortcut}", p, {"action": "shortcut", "shortcut": shortcut}
    for app in OPEN_APP_TARGETS:
        for verb in rng.sample(["open", "launch", "switch to", "bring up", "start", "go to", "open up", "pull up"], 3):
            yield "open_app", f"{verb} {rng.choice(_spoken(app))}", {"action": "open_app", "app": app}
    for name in NOT_INSTALLED:
        yield "open_app.none", f"open {name}", {"action": "open_app", "app": "none"}
    for site, spoken in SITE_SPOKEN.items():
        for verb in rng.sample(["go to", "open", "pull up", "take me to", "visit", "load"], 3):
            yield "open_website", f"{verb} {rng.choice(spoken)}", {"action": "open_website", "site": site}
    search_templates = [("search {e} for {q}", None), ("{e} search {q}", None), ("search for {q}", "google"), ("google {q}", "google"),
                        ("look up {q}", "google"), ("find {q} on {e}", None), ("search {e} for {q} please", None), ("look up {q} on {e}", None)]
    for q in QUERIES:
        for tpl, fixed in rng.sample(search_templates, 3):
            engine = fixed or rng.choice(list(ENGINE_SPOKEN))
            yield "web_search", tpl.format(e=ENGINE_SPOKEN[engine], q=q), {"action": "web_search", "engine": engine, "query": q}
    type_templates = [("type {t}", False), ("write {t}", False), ("dictate {t}", False), ("type out {t}", False),
                      ("type {t} and press enter", True), ("write {t} and hit return", True), ("type {t} and send it", True)]
    for t in TEXTS:
        for tpl, submit in rng.sample(type_templates, 3):
            yield "type_text", tpl.format(t=t), {"action": "type_text", "text": t, "submit": submit}
    for t in rng.sample(TEXTS, 8):
        app = rng.choice(["Notes", "Slack", "Messages", "TextEdit"])
        yield "type_text.in_app", f"type {t} in {app.lower()}", {"action": "type_text", "text": t, "submit": False, "in_app": app}
    for title in TITLES:
        yield "new_item.title", rng.choice([f"make a new note called {title}", f"new note titled {title}", f"create a note called {title}"]), \
            {"action": "new_item", "kind": "item", "title": title}
    for p in ["new note", "create a new document", "make a new file", "start a new email", "new message", "create a new note"]:
        yield "new_item", p, {"action": "new_item", "kind": "item"}
    for p in ["new tab", "open a new tab", "make a new tab"]:
        yield "new_item.tab", p, {"action": "new_item", "kind": "tab"}
    for p in ["new window", "open a new window", "make a new window"]:
        yield "new_item.window", p, {"action": "new_item", "kind": "window"}
    for app in ["Notes", "TextEdit", "Pages"]:
        yield "new_item.in_app", f"new note in the {app.lower()} app", {"action": "new_item", "kind": "item", "in_app": app}
    for folder, spoken in FOLDER_SPOKEN.items():
        for s in spoken:
            yield "open_folder", f"{rng.choice(['open', 'show', 'show me'])} {s}", {"action": "open_folder", "folder": folder}
    for _ in range(40):
        app = rng.choice(OPEN_APP_TARGETS)
        site = rng.choice(list(SITE_SPOKEN))
        yield "compound", f"open {rng.choice(_spoken(app))} and go to {rng.choice(SITE_SPOKEN[site])}", \
            {"action": "open_app", "app": app, "compound": True}


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split())


def near_duplicate(u: str, protected: list[str], ratio: float = DUPLICATE_RATIO) -> bool:
    n = _norm(u)
    return any(n == p or SequenceMatcher(None, n, p).ratio() > ratio for p in protected)


def _shortcut_group(shortcut: str) -> str:
    return next(g for g, (_, keys) in SHORTCUT_GROUPS.items() if shortcut in keys)


def _text_key(cands: dict[str, str], value: str) -> str | None:
    return next((k for k, v in cands.items() if _norm(v) == _norm(value)), None)


def build_row(utterance: str, expect: dict[str, Any], installed: list[str], blocked: set[tuple[str, str]],
              misses: Counter | None = None) -> dict[str, Any] | None:
    """One laya-train row, or None when the true label is not an option at inference."""
    misses = misses if misses is not None else Counter()
    reject = bool(expect.get("reject"))
    action = "none" if reject else expect["action"]
    apps = shortlist_apps(utterance, installed, config.LAYA_MAX_APPS)
    cands = text_candidates(utterance)
    qs = for_staged(Brain.__new__(Brain)._questions(cands, apps), utterance, blocked)
    if action not in qs["action"]["criteria"]:
        misses[f"action_not_offered:{action}"] += 1
        return None

    labels: dict[str, Any] = {}
    if expect.get("chatter"):
        labels["addressed"] = False
    elif not reject:
        labels["addressed"] = True
    text_value = expect.get("text") or expect.get("query") or expect.get("title")
    for key in stage_keys(action, utterance):
        if key == "addressed":
            continue
        value: Any = None
        if key in ("media_op", "volume_op", "system_op"):
            value = expect.get("op")
        elif key == "app":
            value = expect.get("in_app") if action != "open_app" else expect.get("app")
            if value is not None and value != "none" and value not in apps:
                misses["app_not_shortlisted"] += 1
                value = None
        elif key == "site":
            value = expect.get("site")
        elif key == "engine":
            value = expect.get("engine")
        elif key == "text":
            if text_value:
                value = _text_key(cands, text_value)
                if value is None:
                    misses["text_not_a_candidate"] += 1
        elif key == "submit":
            value = bool(expect.get("submit"))
        elif key == "new_kind":
            value = expect.get("kind")
        elif key == "has_title":
            value = bool(expect.get("title"))
        elif key == "scroll_dir":
            value = expect.get("direction")
        elif key == "scroll_amount":
            value = expect.get("amount")
        elif key == "folder":
            value = expect.get("folder")
        elif key == "in_app":
            value = bool(expect.get("in_app"))
        elif key == "compound":
            value = bool(expect.get("compound"))
        elif key == "shortcut_group":
            value = _shortcut_group(expect["shortcut"])
        if value is not None:
            labels[key] = value

    questions = {"action": qs["action"], **{k: qs[k] for k in labels}}
    expected = {"action": action, **labels}
    if action == "shortcut":
        group = SHORTCUT_GROUPS[labels["shortcut_group"]][1]
        questions["shortcut"] = {**qs["shortcut"], "criteria": {k: v for k, v in qs["shortcut"]["criteria"].items() if k in group}}
        expected["shortcut"] = expect["shortcut"]
    for k, v in expected.items():
        crit = questions[k].get("criteria")
        if questions[k]["type"] == "choice" and v not in crit:
            misses[f"label_not_offered:{k}"] += 1
            return None
    return {"state": {"utterance": utterance}, "questions": questions, "expected": expected}


def load_paraphrases(path: Path = PARAPHRASES) -> list[tuple[str, str, dict[str, Any]]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out += [(f"para:{rec['family']}", p, rec["expect"]) for p in rec["phrasings"]]
    return out


def build(seed: int = 0, paraphrases: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    blocked = parse_blocked(config.LAYA_BLOCKED)
    protected = [_norm(c.utterance) for c in evaluate.load_cases()]
    pool = list(template_utterances(rng)) + (load_paraphrases() if paraphrases else [])
    seen: set[str] = set()
    rows, stats, misses = [], Counter(), Counter()
    for family, utterance, expect in pool:
        n = _norm(utterance)
        if not n or n in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(n)
        if near_duplicate(utterance, protected):
            stats["dropped_near_eval"] += 1
            continue
        installed = rng.sample(APP_POOL, rng.randint(20, 60))
        for app in (expect.get("app"), expect.get("in_app")):
            if app and app != "none" and app not in installed:
                installed.append(app)
        row = build_row(utterance, expect, installed, blocked, misses)
        if row is None:
            stats["dropped_unrenderable"] += 1
            continue
        row["id"] = f"{family}:{hashlib.sha1(n.encode()).hexdigest()[:8]}"
        row["source"] = "paraphrase" if family.startswith("para:") else "template"
        rows.append(row)
    rng.shuffle(rows)
    items = Counter(k for r in rows for k in r["expected"])
    actions = Counter(r["expected"]["action"] for r in rows)
    return rows, {"rows": len(rows), "items": sum(items.values()), "by_source": dict(Counter(r["source"] for r in rows)),
                  "by_action": dict(actions.most_common()), "items_by_question": dict(items.most_common()),
                  "dropped": dict(stats), "misses": dict(misses)}


def main() -> None:
    p = argparse.ArgumentParser(prog="laya-train-data", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-paraphrases", action="store_true")
    args = p.parse_args()
    rows, stats = build(args.seed, paraphrases=not args.no_paraphrases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats["questions_version"], stats["questions_fingerprint"] = QUESTIONS_VERSION, questions_fingerprint()
    stats["sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]
    (args.out.with_suffix(".stats.json")).write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
