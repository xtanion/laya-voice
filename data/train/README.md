# Training data

Rows for `laya-train` (`{state, questions, expected}`), built by `laya_voice/training.py`.

```sh
uv run python -m laya_voice.paraphrase --dry-run   # seeds and prompts, no API calls
uv run python -m laya_voice.paraphrase             # Claude paraphrases -> paraphrases.v1.jsonl (needs ANTHROPIC_API_KEY)
uv run python -m laya_voice.training               # templates + paraphrases -> v1.jsonl and v1.stats.json
USE_TF=0 uv run laya-train --data data/train/v1.jsonl --dry-run --shuffle-options --loss soft-ce
```

- **Sources:** hand-written phrases and slot templates (`template_utterances`), plus
  Claude paraphrases of one seed per label (`paraphrase.seeds`). Paraphrases are cached in
  `paraphrases.v1.jsonl` and committed, so `v1.jsonl` rebuilds without the API.
- **Rendering:** every row's questions come from Brain's staged path (`for_staged`,
  `stage_keys`, the shortcut group round, the app shortlist), with state
  `{"utterance": ...}`. `tests/test_training.py` checks that a row's questions are exactly
  what Brain asks for that utterance. Rows are tied to `QUESTIONS_VERSION`; the stats file
  records the version and fingerprint.
- **Labels:** only questions Brain would ask for the true action are labelled. A text
  question is labelled only when one of the code-cut candidates equals the payload
  (`misses.text_not_a_candidate` counts the rest). Rejects are `action: none`, and chatter
  is also `addressed: false`.
- **Held-out protection:** utterances whose normalized text matches or is close
  (ratio > 0.85) to any `data/eval` case are dropped (`dropped.dropped_near_eval`).
- **Paraphrase cleaning:** each phrasing must keep its slot words verbatim and pass the
  runtime gates (submit mentioned iff the label submits; stop phrases for `stop`).
- Do not hand-edit `v1.jsonl`; change the templates or seeds and rebuild.
