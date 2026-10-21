# Evaluation data

`eval/commands.jsonl` is the labelled set for the single-command decision path.
`eval/context.json` fixes the installed apps and frontmost app so results do not
depend on the machine. Run it with `uv run laya-eval` (see `laya_voice/evaluate.py`).

Each line is one case:

```json
{"id": "media-pause-1", "group": "media-pause", "split": "dev", "category": "media",
 "utterance": "pause the music", "expect": {"action": "media", "op": "play_pause"}}
```

- `expect` is either an action with the plan arguments to check (only the listed
  ones are compared; text is compared case- and whitespace-insensitively), or
  `{"reject": true}` for requests that must not execute.
- `group` ties paraphrases of the same command together. A group lives in exactly
  one split, so near-duplicates never leak from `dev` into `test`.
- `split`: `dev` is for tuning (descriptions, candidate rules, thresholds);
  `test` is held out. Report on `test`, never tune against it. `laya-eval` only
  sweeps thresholds on `dev`.
- `category` groups results in the report. Every category has cases in both splits.

Rules for changing the set:

- Add cases as new groups; do not move an existing group between splits.
- Note dataset changes in the commit message, since metrics are only comparable
  on the same version of the file.
- `tests/test_evaluate.py` checks ids, splits, groups and that every expected
  value is an option the executor offers.
