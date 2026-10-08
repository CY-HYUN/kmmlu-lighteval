"""Checks that kmmlu_task.py matches lm-eval's kmmlu task files (no model, no download)."""

from pathlib import Path

import lm_eval
import yaml
from lm_eval.utils import apply_template

from kmmlu_task import KMMLU_SUBSETS, TASKS_TABLE, kmmlu_prompt, subset_suffix

LM_EVAL_KMMLU = Path(lm_eval.__file__).parent / "tasks" / "kmmlu" / "default"


def _lm_eval_subjects() -> dict[str, str]:
    """lm-eval task name -> HF config name, read from the installed yaml files."""
    out = {}
    for path in LM_EVAL_KMMLU.glob("kmmlu_*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        out[cfg["task"]] = cfg["dataset_name"]
    return out


def test_same_45_subjects_and_names():
    lm = _lm_eval_subjects()
    assert len(lm) == 45
    ours = {f"kmmlu_{subset_suffix(s)}": s for s in KMMLU_SUBSETS}
    assert ours == lm
    assert len(TASKS_TABLE) == 45


def test_prompt_matches_lm_eval_template():
    base = yaml.safe_load((LM_EVAL_KMMLU / "_default_kmmlu_yaml").read_text(encoding="utf-8"))
    line = {"question": "  다음 중 옳은 것은? \n", "A": "가", "B": "나 ", "C": "3", "D": "다", "answer": 3}
    doc = kmmlu_prompt(line, "kmmlu:test")
    assert doc.query == apply_template(base["doc_to_text"], line)
    # lm-eval scores target_delimiter (" ") + choice; gold is answer-1.
    assert doc.choices == [" " + c for c in base["doc_to_choice"]]
    assert doc.gold_index == int(apply_template(base["doc_to_target"], line)) == 2
    assert base.get("target_delimiter", " ") == " " and base["output_type"] == "multiple_choice"


def test_lm_eval_default_is_zero_shot_loglikelihood():
    base = yaml.safe_load((LM_EVAL_KMMLU / "_default_kmmlu_yaml").read_text(encoding="utf-8"))
    assert "num_fewshot" not in base
    assert [m["metric"] for m in base["metric_list"]] == ["acc"]
