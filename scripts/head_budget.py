"""Show what Laya actually sees of each staged question after its option head budget.

    uv run python scripts/head_budget.py            # summary per question
    uv run python scripts/head_budget.py -v         # plus the surviving text of every option

Loads only the tokenizer (no weights). Laya caps each option at 48 tokens, and when all
options together exceed head_max_len - 16 it cuts every option to (head_max_len - 16) / n
tokens and squeezes the instruction into what is left.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")

from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import build_head  # noqa: E402
from laya.train import resolve_checkpoint_dir, to_internal  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from laya_voice import config, evaluate  # noqa: E402
from laya_voice.brain import SHORTCUT_GROUPS, Brain, for_staged, shortlist_apps, text_candidates  # noqa: E402


def staged_questions() -> dict[str, dict]:
    ctx = evaluate.load_context()
    apps = shortlist_apps("open spotify", ctx["apps"], config.LAYA_MAX_APPS)
    qs = Brain.__new__(Brain)._questions(text_candidates("search youtube for lofi hip hop"), apps)
    qs = for_staged(qs, "stop listening", set())
    for group, (_, keys) in SHORTCUT_GROUPS.items():
        qs[f"shortcut[{group}]"] = {**qs["shortcut"], "criteria": {k: v for k, v in qs["shortcut"]["criteria"].items() if k in keys}}
    del qs["shortcut"]
    return qs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--head-max-len", type=int, default=None)
    args = p.parse_args()

    root = resolve_checkpoint_dir(config.LAYA_REPO if not config.LAYA_SUBFOLDER else os.path.join(config.LAYA_REPO, config.LAYA_SUBFOLDER))
    _fix_tokenizer_config(root)
    tok = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
    with open(os.path.join(root, "rl_agent_config.json"), encoding="utf-8") as f:
        head_max_len = args.head_max_len or json.load(f).get("head_max_len", 192)

    bad = 0
    print(f"head_max_len={head_max_len}\n")
    print(f"{'question':<22}{'opts':>5}{'opt tokens':>11}{'cap/opt':>8}{'instr kept':>12}{'distinct':>9}")
    for key, question in staged_questions().items():
        q = to_internal(key, question)
        ids, markers, stats = build_head(tok, q, head_max_len)
        full_instr = len(tok(f"{q['t']} question: {q['ins']}", add_special_tokens=False)["input_ids"])
        kept_instr = markers[0] - 2 if markers else len(ids) - 2
        full_opts = sum(1 + len(tok(" " + o, add_special_tokens=False)["input_ids"]) for o in _options(q))
        cut = stats["tokens_per_option"] is not None or kept_instr < full_instr
        collapsed = stats["options_distinct"] < stats["options"]
        bad += cut or collapsed
        flag = "  CUT" if cut else ""
        flag += "  COLLAPSED" if collapsed else ""
        cap = stats["tokens_per_option"] or "-"
        print(f"{key:<22}{stats['options']:>5}{full_opts:>11}{cap:>8}{f'{kept_instr}/{full_instr}':>12}"
              f"{stats['options_distinct']:>9}{flag}")
        if args.verbose and cut:
            print(f"    instruction: {tok.decode(ids[1:markers[0] - 1])!r}")
            for a, b in zip(markers, markers[1:] + [len(ids) - 1]):
                print(f"    option: {tok.decode(ids[a + 1:b])!r}")
    print(f"\n{bad} question(s) cut or collapsed")
    return 1 if bad else 0


def _options(q: dict) -> list[str]:
    from laya.common import render_options
    return render_options(q)


if __name__ == "__main__":
    sys.exit(main())
