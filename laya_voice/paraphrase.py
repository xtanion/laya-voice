"""Generate paraphrases of training seeds with Claude (one-time, offline data step).

    uv run python -m laya_voice.paraphrase --dry-run      # list seeds and the prompt, no API calls
    uv run python -m laya_voice.paraphrase                # call Claude for seeds not yet cached
    uv run python -m laya_voice.paraphrase --import F     # cache phrasings from a {family: [phrasings]} JSON file

Results are appended to data/train/paraphrases.v1.jsonl, which is committed so the training
set rebuilds without the API. Seeds already in that file are skipped. Each phrasing must keep
its slot values verbatim and pass the same gates the runtime uses (submit, stop), otherwise it
is dropped before it is cached. API calls need ANTHROPIC_API_KEY (environment or .env); --import
does not, and records the given --source instead of the API model.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from . import training
from .brain import _STOP_HINT, _SUBMIT_HINT, SHORTCUT_CRITERIA

MODEL = "claude-opus-5-5"
DEFAULT_N = 25
REJECT_N = 60

MEANINGS = {
    "media.play_pause": "pause, resume, play, or stop the music or video that is currently playing",
    "media.next": "skip to the next track or song",
    "media.previous": "go back to the previous track or song",
    "volume.up": "make the system volume louder by a step",
    "volume.down": "make the system volume quieter by a step",
    "volume.mute": "mute the system sound",
    "volume.unmute": "unmute the system sound",
    "volume.max": "set the system volume to maximum",
    "volume.half": "set the system volume to about half",
    "system.lock": "lock the Mac's screen",
    "system.sleep_display": "put the display to sleep (screen off, computer stays on)",
    "system.show_desktop": "show the desktop by moving windows aside",
    "system.toggle_dark_mode": "switch between dark mode and light mode",
    "screenshot": "take a screenshot of the screen",
    "stop": "tell the voice assistant itself to stop listening or exit",
    "scroll.down": "scroll the current page down about one screen",
    "scroll.up": "scroll the current page up about one screen",
    "scroll.top": "jump to the very top of the page",
    "scroll.bottom": "jump to the very bottom of the page",
}
REJECT_MEANINGS = {
    "reject.chatter": "everyday speech that is NOT a command for the computer: conversation with another person, thinking "
                      "out loud, reactions, small talk, phone-call fragments. None of it should ask the computer to do anything.",
    "reject.unsafe": "requests to the computer that a simple voice-control tool must refuse: deleting or erasing data, sending "
                     "money, sending or replying to messages or email on the user's behalf, purchases, bookings, posting online, "
                     "shutting down or restarting, changing security settings, multi-step errands in apps or websites.",
}


@dataclass
class Seed:
    family: str
    seed: str
    expect: dict[str, Any]
    meaning: str
    n: int = DEFAULT_N
    keep: list[str] = field(default_factory=list)


def seeds() -> list[Seed]:
    out = []
    for family, (expect, phrases) in training.PHRASES.items():
        if family in REJECT_MEANINGS:
            out.append(Seed(family, phrases[0], expect, REJECT_MEANINGS[family], REJECT_N))
        elif family in MEANINGS:
            out.append(Seed(family, phrases[0], expect, MEANINGS[family]))
    for shortcut, phrases in training.SHORTCUT_PHRASES.items():
        out.append(Seed(f"shortcut.{shortcut}", phrases[0], {"action": "shortcut", "shortcut": shortcut},
                        f"press a key or keyboard shortcut: {SHORTCUT_CRITERIA[shortcut]}"))
    for app, spoken in [("Spotify", "spotify"), ("Slack", "slack"), ("Visual Studio Code", "vs code"), ("Google Chrome", "chrome")]:
        out.append(Seed(f"open_app.{app}", f"open {spoken}", {"action": "open_app", "app": app},
                        f"open, launch, or switch to the {app} application", keep=[spoken]))
    for site, spoken in [("reddit", "reddit"), ("github", "github"), ("netflix", "netflix")]:
        out.append(Seed(f"open_website.{site}", f"go to {spoken}", {"action": "open_website", "site": site},
                        f"open the {spoken} website in the browser (no search)", keep=[spoken]))
    for engine, spoken, q in [("youtube", "youtube", "jazz for studying"), ("amazon", "amazon", "standing desk"),
                              ("google", "google", "banana bread recipe"), ("wikipedia", "wikipedia", "the roman empire")]:
        out.append(Seed(f"web_search.{engine}", f"search {spoken} for {q}", {"action": "web_search", "engine": engine, "query": q},
                        f"search {spoken} for exactly this query: '{q}'", keep=[spoken, q] if engine != "google" else [q]))
    for t, submit in [("thanks so much", False), ("running late", False), ("sounds good to me", True)]:
        seed = f"type {t}" + (" and press enter" if submit else "")
        meaning = f"type exactly this text into the focused field: '{t}'" + (", then press enter to submit or send it" if submit else ", without submitting it")
        out.append(Seed(f"type_text.{'submit' if submit else 'plain'}.{t}", seed,
                        {"action": "type_text", "text": t, "submit": submit}, meaning, keep=[t]))
    out.append(Seed("new_item.title", "make a new note called recipes", {"action": "new_item", "kind": "item", "title": "recipes"},
                    "create a new note titled exactly 'recipes'", keep=["recipes"]))
    out.append(Seed("open_folder.downloads", "open my downloads", {"action": "open_folder", "folder": "downloads"},
                    "open the Downloads folder in Finder", keep=["download"]))
    return out


def prompt(s: Seed) -> str:
    rules = [
        f"Write {s.n} different things a person might say out loud to a voice assistant on their Mac.",
        f"Every one must mean exactly: {s.meaning}.",
        f"Example: \"{s.seed}\".",
        "Vary wording, length, and tone: terse two-word commands, casual, polite, indirect (\"can you...\", \"i need...\"), "
        "with and without filler words. Write how people talk, not how they type.",
        "Lowercase, no punctuation except apostrophes, no assistant name or wake word, at most 14 words each.",
        "Every phrasing must be unambiguous: it must not also fit a different command.",
    ]
    if s.keep:
        rules.append("Each phrasing must contain these words verbatim: " + ", ".join(f"'{k}'" for k in s.keep) + ".")
    if s.expect.get("action") == "type_text":
        rules.append("Use a verb like type, write, enter, or dictate so it is clear the words should be typed.")
    if s.family == "stop":
        rules.append("Each phrasing must contain one of: 'stop listening', 'quit listening', 'exit voice control', "
                     "'end voice control', 'stop the assistant', 'go to sleep', or start with 'goodbye' or 'bye'.")
    return "\n".join(f"- {r}" for r in rules)


_CLEAN = re.compile(r"[^a-z0-9' ]+")


def clean(s: Seed, phrasings: list[str]) -> list[str]:
    out = []
    for p in phrasings:
        p = " ".join(_CLEAN.sub(" ", p.lower()).split())
        if not p or len(p.split()) > 14 or p in out:
            continue
        if any(k.lower() not in p for k in s.keep):
            continue
        if s.expect.get("action") == "type_text" and bool(_SUBMIT_HINT.search(p)) != bool(s.expect.get("submit")):
            continue
        if s.family == "stop" and not _STOP_HINT.search(p):
            continue
        out.append(p)
    return out


SCHEMA = {"type": "object", "properties": {"phrasings": {"type": "array", "items": {"type": "string"}}},
          "required": ["phrasings"], "additionalProperties": False}


def generate(client: Any, s: Seed) -> list[str]:
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        system="You write varied, realistic training utterances for a voice-command classifier.",
        messages=[{"role": "user", "content": prompt(s)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"refused: {response.stop_details}")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["phrasings"]


def cached_families() -> set[str]:
    if not training.PARAPHRASES.exists():
        return set()
    return {json.loads(line)["family"] for line in training.PARAPHRASES.read_text().splitlines() if line.strip()}


def record(s: Seed, raw: list[str], source: str) -> dict[str, Any]:
    return {"family": s.family, "seed": s.seed, "expect": s.expect, "model": source, "requested": s.n,
            "returned": len(raw), "phrasings": clean(s, raw)}


def import_file(path: str, todo: list[Seed], source: str) -> None:
    given = json.loads(open(path).read())
    unknown = set(given) - {s.family for s in seeds()}
    if unknown:
        raise SystemExit(f"unknown families: {sorted(unknown)}")
    with training.PARAPHRASES.open("a") as f:
        for s in todo:
            if s.family in given:
                rec = record(s, given[s.family], source)
                f.write(json.dumps(rec) + "\n")
                print(f"  {s.family}: kept {len(rec['phrasings'])}/{rec['returned']}")


def main() -> None:
    p = argparse.ArgumentParser(prog="laya-paraphrase", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--only", nargs="*", help="limit to these families")
    p.add_argument("--import", dest="import_path", help="cache phrasings from a {family: [phrasings]} JSON file")
    p.add_argument("--source", default=f"{MODEL} (session)", help="model label recorded for --import")
    args = p.parse_args()

    done = cached_families()
    todo = [s for s in seeds() if s.family not in done and (not args.only or s.family in args.only)]
    total = sum(s.n for s in todo)
    print(f"{len(todo)} seeds to paraphrase ({len(done)} cached), {total} phrasings requested, model {MODEL}")
    if args.dry_run:
        for s in todo:
            print(f"\n[{s.family}] -> {s.expect}\n{prompt(s)}")
        return
    training.PARAPHRASES.parent.mkdir(parents=True, exist_ok=True)
    if args.import_path:
        import_file(args.import_path, todo, args.source)
        return

    import anthropic

    client = anthropic.Anthropic()
    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(generate, client, s): s for s in todo}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                raw = fut.result()
            except (anthropic.APIError, RuntimeError, ValueError, KeyError) as e:
                print(f"  ! {s.family}: {e}")
                continue
            rec = record(s, raw, MODEL)
            with lock, training.PARAPHRASES.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {s.family}: kept {len(rec['phrasings'])}/{len(raw)}")


if __name__ == "__main__":
    main()
