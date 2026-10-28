"""Run KMMLU subjects through lighteval (kmmlu_task.py) and lm-evaluation-harness, then compare per question.

    uv run python reproduce.py                       # both tools + compare, default subjects
    uv run python reproduce.py --only compare        # re-read saved outputs only
    uv run python reproduce.py --subjects all --device cuda   # all 45 subjects on a GPU

Outputs go to results/ (gitignored: per-question files hold KMMLU text, CC BY-ND 4.0).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
# Five subjects with few prompt tokens, picked so both tools finish within about 3 hours on a laptop CPU.
DEFAULT_SUBJECTS = ["Health", "Education", "Korean-History", "Accounting", "Math"]


def _plain(x):
    """Tensors (or lists of them) from lighteval details -> JSON-ready lists."""
    return x.tolist() if hasattr(x, "tolist") else [_plain(v) for v in x] if isinstance(x, (list, tuple)) else x


def run_lighteval(subjects: list[str], out_dir: Path, limit: int | None = None, device: str = "cpu") -> None:
    """Same steps as `lighteval accelerate` (lighteval/main_accelerate.py), with two Windows workarounds.

    lighteval 0.13.0 puts the task name "kmmlu:x|0" into cache and details paths; '|' and ':' are not
    valid in Windows file names (maintainers: Windows is not supported, lighteval issue #632). So the
    sample cache is switched off and per-question details are written here instead of --save-details.
    """
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.transformers.transformers_model import TransformersModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    from kmmlu_task import subset_suffix

    tasks = ",".join(f"kmmlu:{subset_suffix(s)}|0" for s in subjects)
    model_config = TransformersModelConfig(
        model_name=MODEL,
        revision=MODEL_REVISION,
        dtype="float32",
        device=device,
        batch_size=1,
        override_chat_template=False,  # lm-eval scores the raw prompt unless --apply_chat_template
    )
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=PipelineParameters(
            launcher_type=ParallelismManager.ACCELERATE,
            custom_tasks_directory=str(ROOT / "kmmlu_task.py"),
            max_samples=limit,
        ),
        evaluation_tracker=EvaluationTracker(output_dir=str(out_dir), save_details=False),
        model_config=model_config,
    )
    pipeline.model._cache = None  # see docstring
    pipeline.evaluate()
    pipeline.show_results()

    out_dir.mkdir(parents=True, exist_ok=True)
    for task_name, details in pipeline.get_details().items():
        suffix = task_name.split(":", 1)[1].split("|", 1)[0]
        with open(out_dir / f"{suffix}.jsonl", "w", encoding="utf-8") as f:
            for d in details:
                row = {
                    "doc_id": int(d.doc.id),
                    "gold": d.doc.gold_index,
                    "logprobs": [float(x) for x in d.model_response.logprobs],
                    "acc": float(d.metric["acc"]),
                    "query": d.doc.query,
                    "input_tokens": _plain(d.model_response.input_tokens),
                }
                f.write(json.dumps(row) + "\n")
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(pipeline.get_results(), f, indent=2, default=str)


def run_lm_eval(subjects: list[str], out_dir: Path, limit: int | None = None, device: str = "cpu") -> None:
    from kmmlu_task import subset_suffix

    cmd = [
        sys.executable, "-m", "lm_eval", "run",
        "--model", "hf",
        "--model_args", f"pretrained={MODEL},revision={MODEL_REVISION},dtype=float32",
        "--tasks", ",".join(f"kmmlu_{subset_suffix(s)}" for s in subjects),
        "--device", device,
        "--batch_size", "1",
        "--output_path", str(out_dir),
        "--log_samples",
    ]  # fmt: skip
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True, cwd=ROOT)


def load_lm_eval(out_dir: Path, suffix: str) -> dict[int, dict]:
    files = sorted(out_dir.rglob(f"samples_kmmlu_{suffix}_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no lm-eval samples for kmmlu_{suffix} under {out_dir}")
    rows = {}
    with open(files[-1], encoding="utf-8") as f:  # newest run
        for line in f:
            r = json.loads(line)
            rows[r["doc_id"]] = {
                "gold": int(r["target"]),  # lm-eval logs values as strings
                "logprobs": [float(x[0]) for x in r["filtered_resps"]],
                "acc": r["acc"],
                "query": r["arguments"]["gen_args_0"]["arg_0"],
            }
    return rows


def load_lighteval(out_dir: Path, suffix: str) -> dict[int, dict]:
    with open(out_dir / f"{suffix}.jsonl", encoding="utf-8") as f:
        return {r["doc_id"]: r for r in map(json.loads, f)}


def argmax(xs: list[float]) -> int:
    return max(range(len(xs)), key=lambda i: (xs[i], -i))  # first index on ties, like np.argmax


def compare(subjects: list[str], le_dir: Path, lm_dir: Path) -> list[dict]:
    from transformers import AutoTokenizer

    from kmmlu_task import subset_suffix

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    enc = lambda text: tok(text, add_special_tokens=False)["input_ids"]  # noqa: E731

    def split_tokens_differ(query: str) -> bool:
        # lighteval 0.13 always tokenizes context and choice separately (pairwise=True in _loglikelihood_tokens);
        # lm-eval tokenizes context+choice together and splits. They agree when no token crosses the boundary.
        return any(enc(query + c) != enc(query) + enc(c) for c in (" A", " B", " C", " D"))

    table = []
    for s in subjects:
        suffix = subset_suffix(s)
        le, lm = load_lighteval(le_dir, suffix), load_lm_eval(lm_dir, suffix)
        if le.keys() != lm.keys():
            raise ValueError(f"{s}: question ids differ ({len(le)} lighteval vs {len(lm)} lm-eval)")
        ids = sorted(le)
        gold_mismatch = [i for i in ids if le[i]["gold"] != lm[i]["gold"]]
        if gold_mismatch:
            raise ValueError(f"{s}: gold answers differ on doc_ids {gold_mismatch[:10]}")
        prompt_diff = [i for i in ids if le[i]["query"] != lm[i]["query"]]
        pred_diff = [i for i in ids if argmax(le[i]["logprobs"]) != argmax(lm[i]["logprobs"])]
        max_abs = max(abs(a - b) for i in ids for a, b in zip(le[i]["logprobs"], lm[i]["logprobs"], strict=True))
        token_diff = [i for i in ids if split_tokens_differ(lm[i]["query"])]
        # Smallest lm-eval gap between the top two choices; a flip needs this gap below 2 x max_abs_logprob_diff.
        min_margin = min(sorted(lm[i]["logprobs"])[-1] - sorted(lm[i]["logprobs"])[-2] for i in ids)
        acc_le = sum(le[i]["acc"] for i in ids) / len(ids)
        acc_lm = sum(lm[i]["acc"] for i in ids) / len(ids)
        table.append({
            "subject": s, "n": len(ids), "acc_lighteval": acc_le, "acc_lm_eval": acc_lm,
            "diff_pp": 100 * (acc_le - acc_lm), "pred_differs": len(pred_diff),
            "prompt_differs": len(prompt_diff), "tokens_differ": len(token_diff), "pred_differs_ids": pred_diff,
            "max_abs_logprob_diff": max_abs, "min_top2_margin": min_margin,
        })  # fmt: skip
    return table


def print_table(table: list[dict]) -> None:
    print(
        "| Subject | n | lighteval acc | lm-eval acc | diff (pp) | prompts differ | tokens differ "
        "| predictions differ | max abs log-prob diff | min top-2 margin |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in table:
        print(
            f"| {r['subject']} | {r['n']} | {r['acc_lighteval']:.4f} | {r['acc_lm_eval']:.4f} | "
            f"{r['diff_pp']:+.2f} | {r['prompt_differs']} | {r['tokens_differ']} | {r['pred_differs']} | "
            f"{r['max_abs_logprob_diff']:.2e} | {r['min_top2_margin']:.2e} |"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS, help="HF config names, e.g. Korean-History, or 'all' for the 45")
    p.add_argument("--device", default="cpu", help="passed to both tools, e.g. cuda")
    p.add_argument("--only", choices=["lighteval", "lm_eval", "compare"])
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--limit", type=int, help="smoke test only: lighteval draws N random questions, lm-eval takes the first N, so ids do not match")
    a = p.parse_args()
    if a.subjects == ["all"]:
        from kmmlu_task import KMMLU_SUBSETS

        a.subjects = KMMLU_SUBSETS
    le_dir, lm_dir = a.results / "lighteval", a.results / "lm_eval"
    for name, fn, out in [("lighteval", run_lighteval, le_dir), ("lm_eval", run_lm_eval, lm_dir)]:
        if a.only in (None, name):
            t = time.time()
            fn(a.subjects, out, a.limit, a.device)
            print(f"[{name}] wall time {time.time() - t:.0f} s for {a.subjects}")
    if a.only in (None, "compare"):
        table = compare(a.subjects, le_dir, lm_dir)
        print_table(table)
        (a.results / "comparison.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
