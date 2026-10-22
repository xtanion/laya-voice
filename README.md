# Laya Voice

Talk to your Mac. You speak, it opens apps, types, searches, scrolls, presses keys.

A decision model turns the transcript into a typed action plus typed arguments. It
never generates text: code produces candidate values and the model *selects*. Code
owns execution. Two providers sit behind one interface (`DECISION_PROVIDER`):

- **Laya** (local, what this fork targets): a fine-tuned [Laya](https://huggingface.co/convaiinnovations/laya)
  checkpoint running in-process on MPS. Nothing leaves the machine.
- **Jev** (hosted, TypeSafe's System One model): one ~250 ms fan-out request per utterance.

```
mic ─► energy VAD ─► whisper.cpp small.en (Metal, ~330 ms) ─► Laya (local, ~200 ms) ─► macOS actions ─► chime / `say`
```

## Performance

Measured on a Mac mini M4 (16 GB), Laya on MPS, dev split of `data/eval` (62 labelled
cases: 56 commands, 6 that must be rejected), threshold 0 (the runtime default).

**Decision model, typed text**

| Model | Command accuracy | Action accuracy | False actions | Latency p50 / p95 |
| --- | --- | --- | --- | --- |
| Laya base checkpoint (zero-shot, threshold 0.35) | 57.1% | | 12.9% | |
| Laya fine-tuned pilot (`laya-voice-v1-pilot`) | **96.4%** | 100% | 4.8% | 195 / 324 ms |

The 4.8% false actions are 3 of 62 cases: one should-reject request that ran and two
commands executed with the wrong arguments. A 0.1 threshold keeps 96.4% accuracy with
1.8% of commands held back.

**End to end on speech** (`laya-eval --asr`): every case spoken by four macOS voices
(Samantha, Daniel, Rishi, Karen), transcribed by whisper.cpp, decided by Laya.

| Whisper | Prompt | Command accuracy | Action accuracy | False actions |
| --- | --- | --- | --- | --- |
| base.en | none | 85.7% | 92.9% | 7.7% |
| base.en | `WHISPER_PROMPT` | 90.2% | 96.0% | 7.3% |
| small.en | none | 87.9% | 92.9% | 8.1% |
| small.en | `WHISPER_PROMPT` | 91.5% | 97.8% | 4.8% |
| small.en + prompt, pilot at threshold 0 | `WHISPER_PROMPT` | **94.2%** | 97.8% | 6.9% |

The first four rows use a 0.35 threshold; the last is the current runtime setting. Most
remaining errors are keyboard shortcuts (40 of 48 right) and unsafe requests that should
be refused (4 of 12 run). Synthetic voices only so far, not a real mic.

**Latency per stage**

| Stage | Time |
| --- | --- |
| End-of-speech detection | 550 ms of silence (tune `VADConfig.end_silence_ms`) |
| whisper.cpp base.en / small.en (2.5 s clip, model loaded) | ~120 ms / ~330 ms |
| Laya decision (staged: action, then that action's questions) | p50 195 ms, p95 324 ms |
| Jev fan-out (hosted alternative) | 170–420 ms |
| Front browser tab URL via Accessibility | 35–50 ms |
| Execute + feedback | ~50–100 ms |

Reproduce with `uv run laya-eval --provider laya` (add `--asr Samantha,Daniel,Rishi,Karen`
for speech). The test split is held out until the final model.

## Setup (macOS, Apple Silicon)

```sh
cp .env.example .env
./scripts/setup.sh
uv sync --extra laya                       # local decision model
curl -L -o models/ggml-small.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
```

In `.env`, set `DECISION_PROVIDER=laya`, `LAYA_REPO` to the fine-tuned checkpoint
(a local path such as `models/laya-voice-v1-pilot`, or `convaiinnovations/laya` for
the zero-shot base; the fine-tuned checkpoint is not published yet, train it with
`docs/finetune-plan.md`) and `WHISPER_MODEL=models/ggml-small.en.bin`. For Jev instead, set
`DECISION_PROVIDER=jev` and `TYPESAFE_API_KEY` from console.typesafe.ai. The whisper
prompt is fixed in `laya_voice/config.py`.

The script installs whisper-cpp + ffmpeg, downloads the model, syncs the Python
env, remaps **Caps Lock → F18** with `hidutil` (persisted by a LaunchAgent so it
survives reboots), installs a `laya` launcher in `~/.local/bin`, and opens the
three permission panes. Grant the terminal app you launch from (Cursor / Terminal /
iTerm) **Microphone**, **Accessibility** and **Input Monitoring**. If a permission
is missing at launch, Laya Voice prompts for it and waits.

Undo the Caps Lock remap any time: `./scripts/uninstall-capslock.sh`.

## Run

```sh
laya                                 # hands-free: "Alfred, open chrome" (or tap CAPS LOCK, then speak)
laya --hold                          # hold CAPS LOCK to talk, release to run; no wake word
laya --always-on                     # open mic, EVERY utterance is a command (no wake word)
laya --ptt                           # push-to-talk in the terminal: Enter start / Enter stop
laya --device "RØDE"                 # pick a mic (uv run python -m sounddevice)
laya --remote                        # hold the button on the ESP32 voice module (Bluetooth LE)
laya --text "open chrome and go to youtube" --dry-run   # test routing, no mic
```

**Voice module (`--remote`):** an ESP32 with an INMP441 mic streams IMA-ADPCM audio
over Bluetooth LE while its button is held (`firmware/voice_module`). Wiring: INMP441
VDD→3V3, GND→GND, L/R→GND, SCK→GPIO26, WS→GPIO25, SD→GPIO33; button between GPIO0
(the BOOT button by default) and GND. Flash with
`cd firmware/voice_module && uvx --from platformio pio run -t upload -t monitor`.
The module only streams over an encrypted link, so accept the pairing request on first
connect; the terminal app needs the Bluetooth permission. Once connected, set
`BLE_DEVICE` in `.env` to the printed address so no other device can stand in for it.
At boot the firmware probes SCK/WS both ways round and both L/R slots and keeps the one that yields data;
if it prints `mic probe: no data on any setup`, check that the mic header is soldered and that SD, VDD and GND reach it.

**Hands-free mode (default):** the mic stays open and whisper transcribes every
utterance locally (~100 ms, nothing leaves the machine). Only utterances that name
the assistant (`WAKE_WORDS` in `.env`, default Alfred / Jarvis) are decided. After a
command you have `FOLLOWUP_SECONDS` (8) to chain more without the name: "Alfred,
open chrome" … "go to youtube" … "scroll down". Saying just "Alfred" chimes and
arms the next utterance. A Caps Lock tap does the same.

**Caps Lock modes (`--hold`):** hold it while speaking (Tink = recording, Pop = sent). A
short tap (<250 ms) latches hands-free recording; tap again to send. Caps Lock no
longer toggles capitals while the remap is installed.

## What you can say

| Say | Does |
| --- | --- |
| "open cursor", "switch to chrome" | `open -a` the matching installed app (the model picks from the real app list) |
| "go to youtube", "go to stripe dot com" | opens the site |
| "search youtube for lofi hip hop", "google best ramen near me" | site-specific search |
| "open youtube and search veritasium" | one search on that site |
| "search mrbeast" (with YouTube in front) | searches the site in the front tab, in the same tab |
| "open the first video", "play the second one" | clicks that result on screen |
| "click subscribe", "play the lofi mix" | clicks the on-screen element whose label matches |
| "type hello world and hit enter" | types into the focused field, optional submit |
| "close this tab", "select all and copy", "undo", "go back", "reload" | ~45 keyboard shortcuts |
| "scroll down a lot", "go to the top" | real scroll-wheel events |
| "volume up", "mute", "pause the music", "next song" | system volume / media keys |
| "take a screenshot", "open my downloads", "lock the screen", "toggle dark mode" | misc |
| "open notes and type buy milk and press enter" | compound: the model flags it, code splits it, each step runs in order |

### Browser awareness and voice clicks

The front browser tab is read through Accessibility (`AXWebArea` → `AXURL`, 35–50 ms in
Arc), so no Automation permission is needed. When you launch from a terminal or editor,
the browser highest in the window stack is used. A search with no named site goes to the
site in the front tab if it has search (YouTube stays YouTube), otherwise Google; when
that site is already in front, the search URL is typed into the same tab.

Utterances that ask to click, play, select or "open the first …" skip the model: code
reads the on-screen elements and clicks the match directly. Position commands rank results
that carry a duration by row, then column. Named targets match on shared words, with a
fuzzy tie-break. A trailing "on youtube" is ignored when YouTube is the current site.

## Multi-step tasks: the ultrafast loop on the desktop

Single commands above are one decision and one action, and voice clicks go straight to
the matched element. Typed goals that need several on-screen steps run the agent loop from
[jev-ultrafast](https://github.com/browser-use/jev-ultrafast), ported from the DOM to the
macOS Accessibility tree. With `DECISION_PROVIDER=laya` the desktop agent makes a staged
Laya decision over shortlisted targets; the browser driver, recommendations, chess and the
modes page below still use Jev.

```
AX tree of the frontmost window ─► indexed element table ─► one Jev request ─► executor
                                   [1] button   Back                │ operation
                                   [2] textfield Address and search │ click_target
                                   [3] link     Home                │ type_text_target
                                   ...                              │ open_app_target
                                                                    │ press_key_target
                                                            use the matching head only
```

```sh
laya-agent --goal "open notes and write buy milk"          # from the terminal
laya-agent --goal "..." --choose --elements                # pause before each step, show the table
laya --goal "reply to the last email from Sam saying yes"  # same loop, spoken reply
```

By voice, Jev routes an utterance to `task` when it needs several on-screen steps
("Alfred, find the cheapest flight to London on google flights"). The pill shows each step.

Operations are `CLICK`, `TYPE_TEXT`, `OPEN_APP`, `PRESS_KEY`, `SCROLL_UP`, `SCROLL_DOWN`,
`WAIT`, `DONE`, `BLOCKED`. Only operations with a valid target on this screen are offered.
Every target is an observed `AXUIElement` held by code; the model never emits selectors,
coordinates, scripts, or shell. Before input the executor re-reads the target's role, label,
value and enabled state, re-resolves its geometry, and hit-tests its centre so a covered
control is never clicked. `TYPE_TEXT` values come from a small text model if
`TEXT_MODEL_API_KEY` is set, otherwise Jev *selects* the value from spans cut out of the goal
(nothing generated). Runs are bounded: 40 actions, 80 Jev calls, and three actions in a row
that change nothing stop the run as `BLOCKED`. A `DONE` choice is the model's claim, not proof.

### Browser driver (default) and recommendations

Browser goals run on [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) unmodified:
Chrome over CDP through browser-harness, in a new window on `AGENT_DISPLAY`. One-time setup:
tick **Allow remote debugging for this browser instance** at `chrome://inspect/#remote-debugging`
and click Allow. `TASK_DRIVER=desktop` switches to the experimental Accessibility-tree port.

```sh
laya-agent --url https://www.autotrader.ca --recommend \
  --goal "Find a used 2020 Mercedes-Benz CLA under \$15,000 CAD near Toronto. Postal code M5V 3L9. Stop when the filtered listings are visible."
```

`--recommend` (or saying "recommend me…" by voice) adds one more Jev choice after the run:
code reads the listing cards (title, price, mileage, distance) from the results and following
pages, Jev picks one against your goal, Jev picks the reason from facts code verified
(cheapest, lowest mileage, requested year, closest), and the agent opens that listing.
Nothing is generated. A typical AutoTrader run: ~13 s, ~30 Jev calls, about one cent.

Around ultrafast, all code-owned: controls behind an open modal are withdrawn; an action
repeated twice without leaving the page is withdrawn; a BLOCKED under 50% executes the
runner-up operation from the same request; a DONE chosen twice stands on live pages;
an unchanged page reuses the decision for free; three rejections of one choice stop the run.
Without `TEXT_MODEL_API_KEY`, TYPE_TEXT values are Jev choices over spans cut from the goal
(sentences, comma clauses, labelled values like "postal code M5V 3L9", numbers as "$15,000"/"15000").

### Chess on chess.com (Stockfish proposes, Jev chooses, code moves)

```sh
uv run laya-agent --chess --url https://www.chess.com --goal "On chess.com, start a game and stop when a live board is on screen."
```

Jev gets the tab to a live game (Play, matchmaking, or resuming a game already open). From there, each turn:
code reads the board from the DOM and keeps the exact position with python-chess (the opponent's move is
inferred from the changed squares); Stockfish returns its top five moves with evaluations; Jev selects one
(strongest by default, `CHESS_STYLE` to bias it); code clicks the piece and the square and confirms the board.
About 1.3 s per move. It has won rated 10-minute games by checkmate.

**Teaching mode** (on by default): true facts about the chosen move are computed in code, Jev picks the one worth
stressing, Haiku writes a coach's sentence from those facts only, and the local open-source Kokoro voice speaks
it. The move waits for the narration to finish (and at least `CHESS_MOVE_DELAY` seconds). The inspector shows the
decision tree under the live feed: every candidate in plain words ("pawn takes pawn on e5"), its rank, eval,
line and Jev's probability, plus the narration. `CHESS_VOICE=0` mutes; `CHESS_EXPLAIN=template` skips the LLM.

Engine play in rated games is against chess.com's fair-play rules; use bots or unrated games for an account you care about.

### Inspector

```sh
uv run laya-inspect          # open http://127.0.0.1:8766 on the monitor the agent is NOT confined to
```

The inspector is the jev-ultrafast demo page for the desktop: the confined monitor with numbered
element badges, the operation and target probabilities of each Jev request, the text value
(selected by Jev or written by the text helper), the executed-step checklist with elapsed
seconds and median decision latency, and an exportable decision trail. **Choose next** pauses
before execution; **Run automatically** loops. Pick the monitor in the form; `AGENT_DISPLAY`
in `.env` sets the default. Left/right follow the macOS Displays arrangement, so use `main`
or an index when that differs from the physical layout.

### Browser-use modes

```sh
uv run laya-modes            # open http://127.0.0.1:8767
```

A second, separate page (own port, own static files) that drives Chrome in one of three modes:

| Mode | What runs |
| --- | --- |
| **Ultrafast** | jev-ultrafast exactly as it shipped: the stock `Agent` on a plain CDP tab, Jev chooses, the OpenAI-compatible text helper writes `TYPE_TEXT` (needs `TEXT_MODEL_API_KEY`), 60-step budget, no guards, no Claude. Its Google Flights and fixture scenarios are kept. |
| **Jev + guards** | `WebAgent` with Claude switched off: the window confined to `AGENT_DISPLAY`, modal/cycle/futile pruning, hesitant BLOCKED, provider retries, `TYPE_TEXT` selected by Jev from the goal when no text model is set. |
| **Agent** | The full agent: Claude planner, Jev → Haiku → Fable tie-breaks, DONE verifier, learned site rules and per-site trust. Needs `ANTHROPIC_API_KEY`. |

Step through with **Choose next / Execute choice**, or **Run automatically** (a background job the page polls; **Pause** stops after the current step). Agent modes accept a follow-up goal on the same tab. The right column shows Jev's operation and target probabilities, the text value, run cost, and, in agent mode, every Claude escalation.

Read [docs/design.md](docs/design.md) for the freshness guards and the differences from the browser version.

## How the decision layer works (`laya_voice/brain.py`, `laya_voice/providers.py`)

**Laya** asks in two stages, with only the utterance as state: first the `action`
question, then only the questions that action needs. Shortlists keep option lists short
(installed apps are cut to `LAYA_MAX_APPS` by name similarity), answers are validated
fail-closed, and `LAYA_BLOCKED` options can never be selected. The fine-tuned checkpoint
is trained on templates plus paraphrases built with the same question text as runtime
(`QUESTIONS_VERSION` is frozen); see [docs/finetune-plan.md](docs/finetune-plan.md).

**Jev** sends one request per utterance with ~15 speculative questions evaluated in parallel:

- `action` — Choice over 13 action kinds.
- `app` — Choice over your installed apps (+ `none`); `site`, `engine`, `folder`,
  `shortcut`, `scroll_dir`, `volume_op`, `media_op`, `system_op` — Choices over
  closed sets whose keys are exactly what the executor accepts.
- `text` — Choice over **candidate spans** cut from the transcript by regex
  ("type X", "search for X", quoted text, whole utterance). Jev picks the one that
  is exactly the payload. This is the "select instead of generate" pattern.
- `submit`, `compound` — Nouls.

Both providers answer the same questions.

Code reads only the answers the chosen action needs. Plan confidence is the
minimum over the judgements used. Low-confidence plans still run (`ACTION_MIN_CONFIDENCE`
defaults to 0), except disruptive ones (quit app, close tab, system actions, stop listening),
which say "not sure" below `RISKY_MIN_CONFIDENCE` (0.35). Thresholds live in `laya_voice/config.py`.

## Floating transcription pill

A small always-on-top bar at the top-center of the screen shows what whisper
heard, what was decided, and the result (gray idle · red listening · yellow
heard · blue thinking · green done · orange error). The idle text names the
mode's trigger, and in hands-free mode "Listening" only shows once armed. It
never takes keyboard focus. `OVERLAY=0` or `--no-overlay` hides it. Ctrl-C stops
the service and its whisper-server.

## Feedback

`FEEDBACK=ding` (default) plays a chime when an action completes and a low buzz
on failure. `FEEDBACK=voice` gives spoken replies from a posh butler persona
(`PERSONA=alfred`, or `cowboy`) using the best British voice installed, or
ElevenLabs if `ELEVENLABS_API_KEY` is set (phrases cached to disk, so repeats are
instant).

## Layout

```
laya_voice/
  main.py     loop, CLI, compound handling, task hand-off
  brain.py    questions, candidate extraction, engine resolution, Plan (single commands)
  providers.py decision providers: Jev (hosted) and Laya (local, staged)
  evaluate.py laya-eval: labelled dev/test eval, threshold sweep, speech eval (--asr)
  training.py training-data builder (templates + paraphrases); paraphrase.py generates them
  remote.py   ESP32 voice module over Bluetooth LE
  agent.py    multi-step loop: observe → choose → act (port of jev-ultrafast/agent.py)
  desktop.py  AX-tree snapshot, indexed controls, freshness guards, execution (snapshot.js + browser.py)
  policy.py   dynamic operation/target heads, text helper (model.py)
  questions.py model instructions for the agent
  inspector.py loopback inspector server (demo.py); static/ holds the page
  modes.py    browser-use modes page (ultrafast / jev + guards / agent); static_modes/ holds it
  web.py      browser driver: jev-ultrafast Agent/Browser on a confined Chrome window + guards
  recommend.py listing harvest → Jev pick → verified reason → open the listing
  chess_play.py board reader, Stockfish candidates → Jev choice, click-click moves, teaching lines
  escalate.py  Haiku/Fable escalation: plan, text values, tie-break, verify, progress, move explanations
  costs.py    Jev / text-helper cost accounting
  actions.py  macOS execution (open, keystrokes, scroll, volume, media keys…)
  audio.py    mic + VAD endpointing
  stt.py      whisper-server client
  tts.py      macOS `say`
  config.py   env / thresholds
  hotkey.py   Caps Lock (remapped to F18) global key tap
  overlay.py  floating transcription pill (AppKit)
  persona.py  butler / cowboy phrasing
scripts/
  setup.sh    one-shot install: deps, model, Caps Lock remap, launcher, permissions
  hf_train.py fine-tune Laya on Hugging Face hardware
data/
  eval/       labelled commands (dev/test) and a fixed machine context
  train/      v1 training set and paraphrases
tests/
  test_agent.py  offline contracts for the agent loop (no screen, no paid APIs)
```

## Development

```sh
uv run ruff check .
uv run pytest          # offline
```

## License

MIT
