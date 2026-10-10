"""Score KMMLU with Claude models through the Message Batches API, generative protocol.

    uv run python scripts/api_eval.py estimate --model claude-haiku-4-5-20251001 [--subjects ...]  # token count on a sample -> USD
    uv run python scripts/api_eval.py submit --model claude-haiku-4-5-20251001 [--subjects ...]    # one batch per subject
    uv run python scripts/api_eval.py collect                                                      # fetch ended batches, score, write results/api_models.md

The user turn is kmmlu_task.kmmlu_prompt(row).query, the string lm-eval and the lighteval task score; a system prompt
asks for the letter only and a structured-output schema (enum A-D) holds the model to it, since a plain instruction did
not (both models started a worked solution instead). This is a generative protocol. lm-eval and
lighteval instead compare the log-likelihood of the four choices, which the API does not expose, so these numbers are
not comparable with the README table or the published KMMLU scores. Per-request outputs (results/api/*.results.jsonl)
hold the model's reply, not the question; results/ stays gitignored except results/api_models.md.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import anthropic
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kmmlu_task import KMMLU_REVISION, KMMLU_SUBSETS, kmmlu_prompt  # noqa: E402

OUT = ROOT / "results" / "api"
KEY_FILE = Path.home() / ".anthropic" / "text2cad_api.env"  # one line; read here, passed to the client only
SYSTEM = "This is a Korean multiple-choice exam. Reply with the letter of the correct choice only: A, B, C or D."
FORMAT = {"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "string", "enum": list("ABCD")}},
                                            "required": ["answer"], "additionalProperties": False}}
MAX_TOKENS = 32  # {"answer": "B"} is about 10 tokens
ATTEMPT1 = ("A first attempt the same day (results/api/attempt1_free_text/, $0.27) asked for the letter in free text with "
            "`max_tokens=8` and no schema: Haiku 4.5 began a worked solution on 689 of 700 questions and Sonnet 5.5 on 340, so "
            "the replies held no letter. A stronger instruction changed nothing on a 12-question check; the schema did.")
# Message Batches prices, USD per million tokens (input, output), 50% of list;
# platform.claude.com/docs/en/about-claude/pricing read 2026-10-10.
PRICE = {"claude-haiku-4-5-20251001": (0.5, 2.5), "claude-sonnet-5-5": (1.0, 5.0)}
README_SUBJECTS = ["Health", "Education", "Korean-History", "Accounting", "Math"]
LETTER = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=KEY_FILE.read_text(encoding="utf-8").strip())


def rows(subject: str):
    return load_dataset("HAERAE-HUB/KMMLU", subject, split="test", revision=KMMLU_REVISION)


def request(model: str, query: str) -> dict:
    p = {"model": model, "max_tokens": MAX_TOKENS, "system": SYSTEM, "messages": [{"role": "user", "content": query}],
         "output_config": {"format": FORMAT}}
    if model.startswith("claude-sonnet-5-5"):  # no thinking pass: the same direct-answer protocol as Haiku 4.5
        p["thinking"] = {"type": "between_tools"}
        p["output_config"]["effort"] = "low"  # Haiku 4.5 rejects the effort parameter
    return p


def parse_letter(text: str) -> int | None:
    m = LETTER.search(text)
    return "ABCD".index(m.group(1)) if m else None


def estimate(model: str, subjects: list[str]) -> None:
    c = client()
    queries = [kmmlu_prompt(r).query for s in subjects for r in rows(s)]
    chars = sum(map(len, queries))
    sample = random.Random(0).sample(queries, min(100, len(queries)))
    toks = sum(c.messages.count_tokens(model=model, system=SYSTEM, messages=[{"role": "user", "content": q}]).input_tokens
               for q in sample)
    ratio = toks / sum(map(len, sample))
    pin, pout = PRICE[model]
    tin = chars * ratio
    print(f"{model}: {len(queries)} questions in {len(subjects)} subjects, {chars} chars, {ratio:.3f} tokens/char on "
          f"{len(sample)} sampled prompts -> about {tin / 1e6:.2f} M input tokens, "
          f"about ${(tin * pin + len(queries) * MAX_TOKENS * pout) / 1e6:.2f}")


def submit(model: str, subjects: list[str]) -> None:
    c = client()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        reqs = [{"custom_id": f"{s}__{i}", "params": request(model, kmmlu_prompt(r).query)} for i, r in enumerate(rows(s))]
        b = c.messages.batches.create(requests=reqs)
        (OUT / f"{b.id}.json").write_text(json.dumps({"batch": b.id, "model": model, "subject": s, "n": len(reqs),
                                                       "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1), newline="\n")
        print(f"{s}: {b.id} ({len(reqs)} requests)", flush=True)


def collect() -> None:
    c = client()
    metas = sorted((json.loads(p.read_text()) for p in OUT.glob("msgbatch_*.json")), key=lambda m: m["created"])
    for m in metas:
        res = OUT / f"{m['batch']}.results.jsonl"
        if res.exists():
            continue
        b = c.messages.batches.retrieve(m["batch"])
        if b.processing_status != "ended":
            print(f"{m['batch']} {m['model']} {m['subject']}: {b.processing_status} {b.request_counts}")
            continue
        out = []
        for r in c.messages.batches.results(m["batch"]):
            rec = {"custom_id": r.custom_id, "status": r.result.type}
            if r.result.type == "succeeded":
                msg = r.result.message
                rec |= {"text": "".join(x.text for x in msg.content if x.type == "text"), "stop": msg.stop_reason,
                        "in": msg.usage.input_tokens, "out": msg.usage.output_tokens}
            out.append(rec)
        res.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8", newline="\n")
        print(f"{m['batch']} {m['model']} {m['subject']}: {len(out)} results")
    report(metas)


def score(m: dict) -> dict | None:
    res = OUT / f"{m['batch']}.results.jsonl"
    if not res.exists():
        return None
    gold = [r["answer"] - 1 for r in rows(m["subject"])]
    got = {x["custom_id"]: x for x in map(json.loads, res.read_text(encoding="utf-8").splitlines())}
    st = {"n": len(gold), "correct": 0, "unparsed": 0, "failed": 0, "in": 0, "out": 0, "batch": m["batch"]}
    for i, g in enumerate(gold):
        x = got.get(f"{m['subject']}__{i}")
        if not x or x["status"] != "succeeded":
            st["failed"] += 1  # an errored or missing request counts as wrong, never as excluded
            continue
        st["in"] += x["in"]
        st["out"] += x["out"]
        p = parse_letter(x["text"])
        st["unparsed"] += p is None
        st["correct"] += p == g
    pin, pout = PRICE[m["model"]]
    st["usd"] = (st["in"] * pin + st["out"] * pout) / 1e6
    return st


def report(metas: list[dict]) -> None:
    per: dict[str, dict[str, dict]] = {}
    for m in metas:
        if st := score(m):
            per.setdefault(m["model"], {})[m["subject"]] = st  # a later batch for the same pair replaces the earlier
    lines = [f"# KMMLU through the Claude API, generative protocol (collected {time.strftime('%Y-%m-%d')})", "",
             "**Protocol.** Each question is sent once, zero-shot, as the user turn `kmmlu_task.kmmlu_prompt(row).query` "
             "(the exact lm-eval `kmmlu` prompt, full-width colon included), with the system prompt "
             f"\"{SYSTEM}\", a structured-output schema `{{\"answer\": A|B|C|D}}` and `max_tokens={MAX_TOKENS}`; the first "
             "A-D in the reply is the prediction, a reply with no letter counts as wrong. "
             f"{ATTEMPT1} No thinking pass (Haiku 4.5 has none unless given a budget; Sonnet 5.5 runs with "
             "`thinking: between_tools`, effort low). **This differs from the log-likelihood protocol of lm-eval and the "
             "lighteval task in the README**, which score the log-probability of each of the four choices and take the "
             "highest; the API does not expose those, so the accuracies here are not comparable with the README table or "
             "the published KMMLU numbers (they are closer to lm-eval's `kmmlu_direct` variant, which is not run here). "
             f"Dataset `HAERAE-HUB/KMMLU` at `{KMMLU_REVISION[:7]}`, test split, Message Batches API. Spend is computed "
             "from each reply's `usage` at the batch prices in `scripts/api_eval.py`.", ""]
    for model, subs in per.items():
        lines += [f"## {model}", "", "| Subject | n | acc | unparsed | failed | batch |", "|---|---:|---:|---:|---:|---|"]
        for s in [x for x in KMMLU_SUBSETS if x in subs]:
            st = subs[s]
            lines.append(f"| {s} | {st['n']} | {st['correct'] / st['n']:.4f} | {st['unparsed']} | {st['failed']} | `{st['batch']}` |")
        sub5 = [subs[s] for s in README_SUBJECTS if s in subs]
        if len(sub5) == len(README_SUBJECTS):
            n5 = sum(x["n"] for x in sub5)
            lines.append(f"| **README 5 subjects (micro)** | {n5} | {sum(x['correct'] for x in sub5) / n5:.4f} | "
                         f"{sum(x['unparsed'] for x in sub5)} | {sum(x['failed'] for x in sub5)} | |")
        if len(subs) == len(KMMLU_SUBSETS):
            n = sum(x["n"] for x in subs.values())
            lines.append(f"| **all 45 (micro)** | {n} | {sum(x['correct'] for x in subs.values()) / n:.4f} | "
                         f"{sum(x['unparsed'] for x in subs.values())} | {sum(x['failed'] for x in subs.values())} | |")
            lines.append(f"| **all 45 (macro, mean of subjects)** | 45 | "
                         f"{sum(x['correct'] / x['n'] for x in subs.values()) / 45:.4f} | | | |")
        lines += ["", f"Spend: **${sum(x['usd'] for x in subs.values()):.2f}** "
                  f"({sum(x['in'] for x in subs.values()) / 1e6:.2f} M input, {sum(x['out'] for x in subs.values())} output tokens) "
                  f"over {len(subs)} subjects.", ""]
    lines += ["- **acc**: correct / n; unparsed replies and failed requests are in n and count as wrong.",
              "- **unparsed**: replies with no A-D letter. **failed**: requests the batch did not complete.",
              "- micro pools all questions; macro is the mean of the per-subject accuracies (the KMMLU paper reports macro)."]
    (ROOT / "results" / "api_models.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


def selfcheck() -> None:
    assert parse_letter("B") == 1 and parse_letter("정답: C") == 2 and parse_letter("**D**.") == 3
    assert parse_letter("ABC") is None and parse_letter("답은 없다") is None and parse_letter("A.") == 0
    assert parse_letter('{"answer": "D"}') == 3 and parse_letter('{"answer":"A"}') == 0
    assert request("claude-haiku-4-5-20251001", "q")["output_config"] == {"format": FORMAT}
    assert request("claude-sonnet-5-5", "q")["output_config"] == {"format": FORMAT, "effort": "low"}
    print("selfcheck ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["estimate", "submit", "collect", "selfcheck"])
    p.add_argument("--model", choices=sorted(PRICE), default="claude-haiku-4-5-20251001")
    p.add_argument("--subjects", nargs="+", default=README_SUBJECTS, help="HF config names; 'all' = the 45")
    a = p.parse_args()
    subjects = KMMLU_SUBSETS if a.subjects == ["all"] else a.subjects
    {"estimate": lambda: estimate(a.model, subjects), "submit": lambda: submit(a.model, subjects),
     "collect": collect, "selfcheck": selfcheck}[a.command]()


if __name__ == "__main__":
    main()
