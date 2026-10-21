"""Evaluate the single-command decision path on the labelled set in data/eval.

    uv run laya-eval --provider laya                      # dev split, with threshold sweep
    uv run laya-eval --provider laya --split test         # held-out: report only, no sweep
    uv run laya-eval --provider laya --out results.json   # keep the per-case rows

A case either expects an action with arguments, or expects to be rejected (not
executed). "Executes" mirrors main.execute: stop always runs; anything else runs
only without an error, when not `none`, and at or above the confidence threshold.
The installed apps and frontmost app come from data/eval/context.json so results
do not depend on the machine.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import actions, config
from .brain import QUESTIONS_VERSION, Brain, Plan, questions_fingerprint
from .providers import make_provider

DATA = config.ROOT / "data" / "eval"
SPLITS = ("dev", "test")


@dataclass
class Case:
    id: str
    group: str
    split: str
    category: str
    utterance: str
    expect: dict[str, Any]

    @property
    def reject(self) -> bool:
        return bool(self.expect.get("reject"))


@dataclass
class Result:
    case: Case
    plan: Plan


def load_cases(path: Path = DATA / "commands.jsonl", split: str | None = None) -> list[Case]:
    cases = [Case(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    return [c for c in cases if split in (None, "all") or c.split == split]


def load_context(path: Path = DATA / "context.json") -> dict[str, Any]:
    return json.loads(path.read_text())


@contextmanager
def fixed_context(ctx: dict[str, Any]) -> Iterator[None]:
    saved = actions.installed_apps, actions.frontmost_app
    actions.installed_apps = lambda: list(ctx["apps"])
    actions.frontmost_app = lambda: ctx["frontmost_app"]
    try:
        yield
    finally:
        actions.installed_apps, actions.frontmost_app = saved


def executes(plan: Plan, threshold: float) -> bool:
    if plan.error or plan.action == "none":
        return False
    return plan.confidence >= threshold


def _norm(v: Any) -> Any:
    return " ".join(v.lower().split()) if isinstance(v, str) else v


def matches(plan: Plan, expect: dict[str, Any]) -> bool:
    if expect.get("reject") or plan.action != expect["action"]:
        return False
    return all(_norm(plan.args.get(k)) == _norm(v) for k, v in expect.items() if k != "action")


def outcome(r: Result, threshold: float) -> str:
    """correct | false_action (executed something not wanted) | held (did not execute)."""
    if not executes(r.plan, threshold):
        return "held"
    return "false_action" if r.case.reject or not matches(r.plan, r.case.expect) else "correct"


def run(brain: Brain, cases: list[Case], ctx: dict[str, Any], warmup: bool = True) -> list[Result]:
    with fixed_context(ctx):
        if warmup:
            brain.evaluate("warm up")
        return [Result(c, brain.evaluate(c.utterance)) for c in cases]


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def score(results: list[Result], threshold: float) -> dict[str, Any]:
    commands = [r for r in results if not r.case.reject]
    rejects = [r for r in results if r.case.reject]
    out = Counter(outcome(r, threshold) for r in results)
    right_action = [r for r in commands if r.plan.action == r.case.expect["action"]]
    lat = sorted(r.plan.latency_ms for r in results)
    cats: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        cats[r.case.category][outcome(r, threshold)] += 1
    return {
        "threshold": threshold,
        "cases": len(results),
        "commands": len(commands),
        "rejects": len(rejects),
        "command_accuracy": _rate(sum(outcome(r, threshold) == "correct" for r in commands), len(commands)),
        "action_accuracy": _rate(len(right_action), len(commands)),
        "arg_accuracy": _rate(sum(matches(r.plan, r.case.expect) for r in right_action), len(right_action)),
        "reject_rate": _rate(sum(outcome(r, threshold) == "held" for r in rejects), len(rejects)),
        "missed_rate": _rate(sum(outcome(r, threshold) == "held" for r in commands), len(commands)),
        "false_action_rate": _rate(out["false_action"], len(results)),
        "errors": sum(bool(r.plan.error) for r in results),
        "latency_ms": {
            "mean": round(statistics.mean(lat)) if lat else None,
            "p50": lat[len(lat) // 2] if lat else None,
            "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
        },
        "by_category": {c: {"n": sum(v.values()), **dict(v)} for c, v in sorted(cats.items())},
    }


def sweep(results: list[Result], thresholds: list[float]) -> list[dict[str, Any]]:
    rows = []
    for t in thresholds:
        s = score(results, t)
        rows.append({k: s[k] for k in ("threshold", "command_accuracy", "false_action_rate", "reject_rate", "missed_rate")})
    return rows


def suggest_threshold(results: list[Result], max_false_rate: float, thresholds: list[float]) -> dict[str, Any] | None:
    """Lowest threshold whose false-action rate is within budget (most commands kept)."""
    for row in sweep(results, thresholds):
        if (row["false_action_rate"] or 0.0) <= max_false_rate:
            return row
    return None


def _row(r: Result, threshold: float) -> dict[str, Any]:
    p = r.plan
    return {"id": r.case.id, "split": r.case.split, "category": r.case.category, "utterance": r.case.utterance,
            "expect": r.case.expect, "action": p.action, "args": {k: v for k, v in p.args.items() if k not in ("addressed",)},
            "confidence": round(p.confidence, 3), "latency_ms": p.latency_ms, "error": p.error, "outcome": outcome(r, threshold)}


def _pct(v: float | None) -> str:
    return "  n/a" if v is None else f"{v * 100:5.1f}%"


def report(results: list[Result], threshold: float, split: str, max_false_rate: float, show: str) -> str:
    s = score(results, threshold)
    lines = [
        f"{s['cases']} cases ({s['commands']} commands, {s['rejects']} should be rejected)  split={split}  threshold={threshold}",
        f"  command accuracy  {_pct(s['command_accuracy'])}   executed with the right action and arguments",
        f"  action accuracy   {_pct(s['action_accuracy'])}   right action, ignoring threshold",
        f"  arg accuracy      {_pct(s['arg_accuracy'])}   right arguments when the action was right",
        f"  reject rate       {_pct(s['reject_rate'])}   should-reject cases that were not executed",
        f"  missed rate       {_pct(s['missed_rate'])}   commands held back",
        f"  false action rate {_pct(s['false_action_rate'])}   executed something unwanted (all cases)",
        f"  errors            {s['errors']}",
        f"  latency           mean {s['latency_ms']['mean']}ms  p50 {s['latency_ms']['p50']}ms  p95 {s['latency_ms']['p95']}ms",
        "",
        f"  {'category':<16}{'n':>4}{'correct':>9}{'false':>7}{'held':>6}",
    ]
    for c, v in s["by_category"].items():
        lines.append(f"  {c:<16}{v['n']:>4}{v.get('correct', 0):>9}{v.get('false_action', 0):>7}{v.get('held', 0):>6}")
    if split == "dev":
        grid = [round(t * 0.05, 2) for t in range(0, 20)]
        lines += ["", f"  {'threshold':>9}{'accuracy':>10}{'false':>8}{'reject':>8}{'missed':>8}"]
        for row in sweep(results, grid):
            lines.append(f"  {row['threshold']:>9}{_pct(row['command_accuracy']):>10}{_pct(row['false_action_rate']):>8}"
                         f"{_pct(row['reject_rate']):>8}{_pct(row['missed_rate']):>8}")
        best = suggest_threshold(results, max_false_rate, grid)
        lines.append("")
        if best:
            lines.append(f"  lowest threshold with false actions <= {max_false_rate:.0%}: {best['threshold']}"
                         f"  (accuracy {_pct(best['command_accuracy']).strip()}, missed {_pct(best['missed_rate']).strip()})")
        else:
            lines.append(f"  no threshold keeps false actions <= {max_false_rate:.0%}")
    if show != "none":
        picked = [r for r in results if show == "all" or outcome(r, threshold) != "correct"]
        if picked:
            lines += ["", f"  {'outcome':<13}{'conf':>5}  utterance -> got  (want)"]
        for r in picked:
            p, o = r.plan, outcome(r, threshold)
            if o == "held" and r.case.reject:
                continue
            got = f"rejected: {p.error}" if p.error else f"{p.action} {_brief(p.args)}"
            want = "reject" if r.case.reject else " ".join(f"{k}={v}" for k, v in r.case.expect.items())
            lines.append(f"  {o:<13}{p.confidence:5.2f}  {r.case.utterance!r} -> {got}  ({want})")
    return "\n".join(lines)


def _brief(args: dict[str, Any]) -> str:
    skip = {"addressed", "compound", "recommend", "message_seller"}
    return " ".join(f"{k}={v}" for k, v in args.items() if k not in skip and v not in (None, False))


def main() -> None:
    p = argparse.ArgumentParser(prog="laya-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default=None, help="decision provider (default: DECISION_PROVIDER)")
    p.add_argument("--split", choices=[*SPLITS, "all"], default="dev")
    p.add_argument("--threshold", type=float, default=config.ACTION_MIN_CONFIDENCE)
    p.add_argument("--max-false-rate", type=float, default=0.02, help="false-action budget for the suggested threshold")
    p.add_argument("--show", choices=["failures", "all", "none"], default="failures")
    p.add_argument("--data", type=Path, default=DATA / "commands.jsonl")
    p.add_argument("--out", type=Path, help="write metrics and per-case rows as JSON")
    args = p.parse_args()

    cases = load_cases(args.data, args.split)
    brain = Brain(make_provider(args.provider))
    t0 = time.perf_counter()
    results = run(brain, cases, load_context())
    print(f"{brain.label}: {len(cases)} cases in {time.perf_counter() - t0:.1f}s  (questions v{QUESTIONS_VERSION} {questions_fingerprint()})\n")
    print(report(results, args.threshold, args.split, args.max_false_rate, args.show))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"provider": brain.label, "split": args.split, "questions_version": QUESTIONS_VERSION,
                                        "questions_fingerprint": questions_fingerprint(), "metrics": score(results, args.threshold),
                                        "cases": [_row(r, args.threshold) for r in results]}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
