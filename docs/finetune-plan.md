# Fine-tuning plan: Laya for laya-voice

Status: step 0 done; step 1 data built except ASR noise (2026-10-21). Baseline: commit `f85c219`, `data/eval` v1, threshold 0.35.

| | dev (62) | test (53, held out) |
|---|---|---|
| Command accuracy (executed correctly) | 53.6% | 43.5% |
| Action accuracy | 64.3% | 58.7% |
| Arg accuracy (given right action) | 94.4% | 88.9% |
| Reject rate | 100% | 85.7% |
| False action rate | 11.3% | 13.2% |
| Latency mean | 242 ms | 238 ms |

No threshold brings dev false actions under 2% (floor 6.5%, from `stop` bypassing the
confidence gate).

## Why fine-tuning is plausible

- Laya's model card calls it "a fast base to specialise, not a zero-shot decision
  engine"; its typed-decisions fine-tune moved 0.362 -> 0.766 on its benchmark.
- Most remaining failures are decision-model errors (action confusion, shortcuts), not
  transcription or schema errors.

## What Laya ships for training (laya 0.4.12)

- `laya-train` CLI / `laya.train.finetune(data, model_dir, output_dir, TrainConfig)`.
- Data: JSONL rows `{state, questions, expected}` in the same question format we send.
- Device: `auto | cpu | cuda | mps`. Output loads with `laya.load(path)`, so
  `LAYA_REPO=models/...` plugs into `LayaProvider`.
- Calibration built in: holds out 10% of items (max 400), fits temperatures with
  `fit_temperature_map`; warns under 50 items per question type (`CALIB_WARN_N`) and
  on collapse to the class prior (#963).
- Options: `--shuffle-options` (option-order augmentation), `--freeze-encoder`
  (heads only), `--loss soft-ce` (their #741 measured no gain from the default
  `rlcd` term), `--dry-run` (builds items without loading weights).
- Option head budget is 192 tokens by default; options that do not fit are skipped
  (`options_collapsed`, `options_beyond_max_len`) and counted by `--dry-run`.
- License: Apache 2.0 (fine-tuning and keeping the checkpoint are fine).

Hardware: Apple M4, 16 GB. Full fine-tune of 421M params needs ~7 GB for weights and
Adam state plus activations; plausible at micro-batch 4-8 since inputs are short.
Fallbacks: `--freeze-encoder`, or the card's Kaggle 2x T4 notebook.

## Plan

### 0. Before training

- Code-gate `stop` (offered only when the utterance matches stop phrases) and make it
  pass the confidence gate like every other action.
- Code-gate `submit` (asked only when the utterance mentions enter/return/send/submit).
- Check the option head budget for every staged question; shorten descriptions that
  are truncated or collapsed.
- Freeze the question text (instructions, criteria, descriptions) under a version;
  the model is trained on those exact strings, so changing them means retraining.
- Re-run `laya-eval` on dev to get the post-fix baseline.

**Step 0 results** (dev, threshold 0.35, questions v1 `ed9da008fbc33d1f`):

- `stop` is offered only for stop phrases and now passes the confidence gate
  (`main.execute`, `evaluate.executes`). `submit` is asked only when the utterance
  mentions enter/return/send/submit.
- `scripts/head_budget.py` showed `action` was the only cut question: 493 option
  tokens squeezed to 11 per option, instruction cut to 29/37. Staged providers now get
  `ACTIONS_SHORT` and `ACTION_INSTRUCTIONS_SHORT` (159 option tokens, nothing cut).
  Jev keeps the long text.
- Question text is frozen as `QUESTIONS_VERSION = 1`; `tests/test_brain.py` pins its
  fingerprint.

| Variant (gates on) | cmd acc | action acc | false actions | reject |
|---|---|---|---|---|
| Baseline before step 0 | 53.6% | 64.3% | 11.3% | 100% |
| Long options, long instruction | 53.6% | 67.9% | 4.8% | 100% |
| Long options, short instruction | 55.4% | 71.4% | 4.8% | 100% |
| Short options, long instruction | 57.1% | 73.2% | 9.7% | 83.3% |
| **Short options, short instruction (v1)** | 57.1% | 73.2% | 12.9% | 83.3% |

The gates did most of the safety work. Short options raise accuracy but make Laya
confidently wrong more often zero-shot. v1 keeps them anyway for training: truncated
long options render differently depending on how many options are offered (blocking
`task`/`stop` changes the per-option cap), so training and inference text would drift.
If fine-tuning is not adopted, revert to long options + short instruction (4.8%).
Neither variant meets the 2% bar zero-shot.

### 1. Training set (`scripts/build_train_data.py` -> `data/train/v1.jsonl`)

Separate from `data/eval`; never trained on dev or test. Target ~2-3k utterances
(~6-10k items: one action item plus the staged argument items per utterance).

- Templates per action with slot fillers (apps, sites, engines, queries, text), fixed seed.
- Rows built through Brain's own question builder and staged rounds, with state
  `{"utterance": ...}`, so training matches inference exactly.
- Hard negatives from the baseline failures: "stop the song" / "skip" / "mute" are
  media or volume, not `stop`; "copy that" is a shortcut, not typing; "send fifty
  dollars to john" is `none`.
- Randomized app shortlists (names and order) so `app` learns name matching.
- ASR noise generated locally: macOS `say` -> whisper.cpp on a subset.
- Rejects: chatter, unsafe, and blocked-action requests labelled `none`.
- Dedupe against dev and test (normalized fuzzy ratio > 0.85 dropped).
- Open decision: Claude-generated paraphrases for diversity (offline, one-time data
  step, not runtime inference). Templates written by the same author as the eval set
  risk sharing its phrasing.

**Step 1 status:** `laya_voice/training.py` (templates, row builder, dedupe) and
`laya_voice/paraphrase.py` (77 seeds, ~2,000 phrasings requested from `claude-opus-5-5`,
cached to `data/train/paraphrases.v1.jsonl`) are in place; see `data/train/README.md`.
The v1 paraphrases (1,984, all passing cleaning) were written in-session by Claude and
imported with `--import` instead of the API. Templates plus paraphrases: 2,391 rows /
8,337 items (653 template, 1,738 paraphrase rows), 142 dropped as near-eval, 190
duplicates, 152 rows with no matching text candidate; `laya-train --dry-run` accepts all
items (0 skipped) and plans 500 optimizer updates over 4 epochs. Imbalance to watch:
shortcut is 1,013 rows, rejects (`none`) 147, stop 22, screenshot 19. ASR-noise
generation is not built yet.

### 2. Dry run

`laya-train --dry-run`: item counts per question type (>= 50 each), zero
skipped/truncated options, optimizer update count (watch the collapse warning).

### 3. Pilot

One epoch, `--freeze-encoder`, subset, on MPS: confirm it runs, measure time per
epoch and memory. No training-time estimate until measured.

**Pilot result (2026-10-22):** ran as a full fine-tune instead (encoder not frozen), one
epoch over all of v1, `--device mps --micro-batch 2 --grad-accum 32 --shuffle-options
--loss soft-ce`. The default micro-batch 8 thrashes swap on the M4 16 GB; micro-batch 2
works (~3 GB RSS). 125 optimizer updates in 50 min. Log: `results/pilot-train.log`.

| dev, threshold 0.35 | step 0 | pilot |
|---|---|---|
| command accuracy | 57.1% | 94.6% |
| action accuracy | 73.2% | 100.0% |
| false action rate | 12.9% | 0.0% |
| reject rate | 83.3% | 100.0% |
| missed rate | | 5.4% |
| mean latency | ~250 ms | 370 ms |

Dev failures (3, all held, none executed): "close the window" -> minimize_window (0.33),
"find lofi hip hop on youtube" -> query cut wrong by the code-side candidates (0.03),
"scroll dawn" right but 0.29. Caveat: training paraphrases, templates and the eval set
were all written by Claude, so dev may share their phrasing; test is still unused.

### 4. Runs (all with `--shuffle-options`)

- (a) `--freeze-encoder`, `soft-ce`
- (b) full, `soft-ce`
- (c) full, `rlcd`

Kaggle fallback if (b)/(c) do not fit in memory; the data is synthetic.

### 5. Evaluate

- Select among runs on dev only with `laya-eval` (the staged pipeline).
- Re-sweep the threshold on dev with the calibrated confidences.
- One final test run.
- Proposed bar: dev false actions <= 2% at some threshold with command accuracy
  >= 85%; test within a few points of dev; reject rate not lower; latency unchanged.
- Independent check: 30-50 utterances written or recorded by the user as `data/eval` v2.

### 6. Integrate

- Save to `models/laya-voice-v1/` (gitignored), point `LAYA_REPO` at it.
- Commit the generator, training data, `data/train/README.md`, and a run record
  (data hash, question version, TrainConfig, eval metrics).
- Keep the base checkpoint selectable for comparison.

## When not to adopt it

- If, after step 0, remaining failures are candidate-span extraction ("X on youtube"),
  fix the regexes instead.
- If dev improves but test or the user's own utterances do not, the model learned the
  templates. Keep the base checkpoint plus code gates.
