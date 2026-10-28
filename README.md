# KMMLU in lighteval, checked against lm-evaluation-harness

KMMLU is a Korean multiple-choice benchmark: 35,030 questions from real Korean
exams in 45 subjects ([paper](https://arxiv.org/abs/2402.11548),
[dataset](https://huggingface.co/datasets/HAERAE-HUB/KMMLU)).
EleutherAI's lm-evaluation-harness ("lm-eval") ships it. Hugging Face's
lighteval does not. This repo adds KMMLU to lighteval as a custom task and
shows, question by question, that it gives the same answers as lm-eval's
`kmmlu` task for the same model.

## Is KMMLU already in lighteval?

No, as of 2026-10-08:

- Installed `lighteval==0.13.0` (latest release, 2025-11-24): no `.py` file
  mentions `kmmlu`, `haerae` or `kobest` (0 files; the same search for `mmlu`
  finds 21).
- `huggingface/lighteval` main at `7fa19dc` (2026-10-06): no path in the tree
  matches `kmmlu|haerae|kobest|korean`, and GitHub code search finds none.
- GitHub issues and pull requests: no hit for KMMLU, HAE-RAE or KoBEST. The
  only Korean pull request, #1179, fixes Korean translation literals.

Proposed upstream on 2026-10-10:
[huggingface/lighteval#1424](https://github.com/huggingface/lighteval/issues/1424) (issue),
then [huggingface/lighteval#1425](https://github.com/huggingface/lighteval/pull/1425) (pull request, the same task as one file).

lighteval does score Korean through multilingual sets that include it
(`global_mmlu`, `openai_mmlu`, `mkqa`, `paws_x`, `tydiqa`). These are
translated from English or built across many languages. What is missing is the
Korean-native benchmarks.

## How the task matches lm-eval

lm-eval's `kmmlu` task (`lm_eval/tasks/kmmlu/default/_default_kmmlu_yaml`,
lm-eval 0.4.13) works like this, and `kmmlu_task.py` copies each part:

| Part | lm-eval `kmmlu` | `kmmlu_task.py` |
|---|---|---|
| Prompt | `{{question.strip()}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n정답：` (the colon is full-width, U+FF1A) | same string |
| Choices scored | `" A"`, `" B"`, `" C"`, `" D"` (choice letter after `target_delimiter=" "`) | same list |
| Gold | `answer - 1` (the dataset is 1-based) | same |
| Method | log-likelihood of each choice, pick the highest, no length normalisation (`acc`) | `Metrics.loglikelihood_acc` (no normalisation) |
| Shots | 0 | 0 (`kmmlu:<subject>\|0`) |
| Chat template | off by default | forced off (`override_chat_template=False`); lighteval would turn it on for this instruct model |

`test_kmmlu_task.py` checks the first three rows against the installed lm-eval
files, by rendering lm-eval's own template. It also checks that all 45
subjects map to lm-eval task names (`kmmlu:korean_history` ↔
`kmmlu_korean_history`). Both checks fail when broken on purpose (ASCII colon,
one subject removed).

One difference cannot be configured away. lighteval 0.13.0 always tokenizes
the prompt and the choice separately (`pairwise=True` is hard-coded in
`TransformersModel._loglikelihood_tokens`). lm-eval tokenizes prompt plus
choice together and splits them. The two agree only when no token crosses the
boundary. `reproduce.py` counts the questions where they disagree for this
tokenizer ("tokens differ" below).

## Results

Model `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae5576`, float32, CPU,
batch size 1. Dataset `HAERAE-HUB/KMMLU` at revision `d61b3f1`. Whole test
split of each subject, 0-shot.

| Subject | n | lighteval acc | lm-eval acc | diff (pp) | prompts differ | tokens differ | predictions differ | max abs log-prob diff | min top-2 margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Health | 100 | 0.3300 | 0.3300 | +0.00 | 0 | 0 | 0 | 5.15e-05 | 7.76e-03 |
| Education | 100 | 0.3600 | 0.3600 | +0.00 | 0 | 0 | 0 | 5.53e-05 | 1.30e-03 |
| Korean-History | 100 | 0.2300 | 0.2300 | +0.00 | 0 | 0 | 0 | 4.01e-05 | 1.79e-02 |
| Accounting | 100 | 0.2500 | 0.2500 | +0.00 | 0 | 0 | 0 | 5.05e-05 | 1.15e-02 |
| Math | 300 | 0.2733 | 0.2733 | +0.00 | 0 | 0 | 0 | 4.24e-05 | 5.36e-04 |

All 700 questions: same prompt string, same tokens, same predicted choice.
Log-probabilities differ by at most 5.53e-05. That size looks like float32
rounding: lighteval runs the four choices as one batch of four, lm-eval runs
them one by one, so the arithmetic order differs (not tested further). Each tool's
own summary (lm-eval `results_*.json`, lighteval `show_results`) prints the
same accuracies as the table.

- **n**: questions in the subject's test split.
- **acc**: share of questions where the highest-scoring choice is the gold one.
- **predictions differ**: questions where the two tools picked different choices.
- **max abs log-prob diff**: largest gap, over every question and choice,
  between the two tools' log-probability for the same choice.
- **min top-2 margin**: smallest gap between the best and second-best choice
  (lm-eval scores). A prediction can only flip if this is below twice the
  previous column.

Versions: lighteval 0.13.0, lm-eval 0.4.13, transformers 5.19.0, torch
2.14.1+cpu, datasets 5.1.0, xxhash 3.8.1, Python 3.12.6 (all in `uv.lock`).
The model runs used commit `bdead50`. The table comes from `reproduce.py
--only compare` at the commit that adds this README; between the two commits
only the comparison code (two columns added) and the default subject list changed.

Wall time on a laptop CPU (Intel i5-1235U, other programs running):

| Run | lighteval | lm-eval |
|---|---:|---:|
| Health, Education (200 questions) | 1,599 s | 635 s |
| Korean-History, Accounting, Math (500 questions) | 2,629 s | 787 s |

Other programs were using the CPU and their load changed during the runs, so
these times are not a speed benchmark. Not measured on a GPU.

## Reproduce

```sh
uv sync --frozen
uv run python reproduce.py
```

The second command runs lighteval, then lm-eval, on the
five subjects above, then prints the table (about 2 hours on the CPU above;
`--subjects Math` runs one subject, `--subjects all --device cuda` runs all 45 on a GPU;
on Windows, PyPI's torch wheel is CPU-only, so install a CUDA build first).
Per-question outputs go to `results/` (not committed, see Limits).
`uv run pytest` runs the task checks without a model.

## Claude models through the API (a different protocol)

`scripts/api_eval.py` sends the same prompt string to Claude models through the
Message Batches API and parses the letter the model writes back (a
structured-output schema limits the reply to one of A-D; asked in plain
text, both models started a worked solution instead). That is a
generative protocol: the API does not expose the per-choice log-likelihoods
that lm-eval and the lighteval task compare, so its accuracies are not
comparable with the table above or with published KMMLU scores. The numbers,
batch ids and spend are in `results/api_models.md` (the only file under
`results/` that is committed). lighteval's `litellm` endpoint was not used: it
is generative only, has no Batch API, and needs the `litellm` extra.

```sh
uv run python scripts/api_eval.py estimate --model claude-haiku-4-5-20251001 --subjects all   # token count on a sample -> USD
uv run python scripts/api_eval.py submit --model claude-haiku-4-5-20251001 [--subjects all]   # one batch per subject
uv run python scripts/api_eval.py collect                                                     # fetch ended batches, rewrite results/api_models.md
```

The API key is read from `%USERPROFILE%/.anthropic/text2cad_api.env` (one line)
inside the script and passed to the client only.

## Limits

- Five of 45 subjects, 700 questions, to fit a CPU budget of about 3 hours.
  They are five of the six subjects with the fewest prompt tokens (Economics
  was left out for Korean-History, the most Korea-specific one), not a random
  sample. Agreement on them supports the prompt and scoring logic; it is not a
  full-benchmark score. The accuracies (23% to 36%, chance is 25%) are for a
  0.5B model; the point is the agreement, not the score.
- One model and one tokenizer. The "tokens differ" count is for the Qwen2.5
  tokenizer only; another tokenizer could merge `：` and ` A` and then the
  tools would differ.
- Windows. lighteval 0.13.0 puts the task name (`kmmlu:health|0`) into file
  paths for its sample cache and for `--save-details`. `|` and `:` are not
  allowed in Windows file names, so `lighteval accelerate` fails at the end
  of the run. Seen here on 2026-10-08 after 89 minutes of scoring:
  `NotADirectoryError: [WinError 267] ... lighteval/Qwen/Qwen2.5-0.5B-Instruct/08667eda30ecdbae/kmmlu:korean_history|0`.
  Same cause as issue #632, where a maintainer wrote "We do not support
  windows in lighteval for now".
  `reproduce.py` builds the same `Pipeline` as `lighteval accelerate`, turns
  the sample cache off and writes per-question results itself.
- `xxhash` 4 breaks lighteval 0.13.0 (`xxh64(str)` raises `TypeError`), so
  `pyproject.toml` pins `xxhash<4`.
- lm-eval's `kmmlu` group reports the micro average (all questions pooled);
  the KMMLU paper uses the macro average (mean over subjects). This repo
  reports per subject only.
- The dataset is CC BY-ND 4.0 (from its Hugging Face card). Per-question files
  contain the questions, so they stay out of git; rerun to get them.
- Only the default log-likelihood variant is ported. lm-eval also has
  `kmmlu_direct` (generation), `kmmlu_hard`, and chain-of-thought variants.

## Next steps for a lighteval contribution

1. Done on 2026-10-10: [lighteval#1424](https://github.com/huggingface/lighteval/issues/1424),
   opened with their "Evaluation/task request" template, as their README asks ("If
   you're adding a new feature, please open an issue first"). It proposes KMMLU as
   `src/lighteval/tasks/tasks/kmmlu.py`, links this comparison, asks whether they want
   the lm-eval prompt or their multilingual template style, and asks about HAE-RAE Bench
   and KoBEST.
2. Done on 2026-10-10, before a maintainer answered: [lighteval#1425](https://github.com/huggingface/lighteval/pull/1425) adds
   `src/lighteval/tasks/tasks/kmmlu.py` (this repo's `kmmlu_task.py`, reformatted with their ruff
   config), keeps the lm-eval prompt and offers to switch to the template style. The PR
   says an AI coding assistant was used. Next: answer review comments; the 45-subject GPU run. lighteval has no CONTRIBUTING file (checked
   2026-10-08), but `huggingface/transformers` CONTRIBUTING.md asks
   first-time contributors not to use code agents to create issues or PRs.
   Assume the same norm here: write the issue and any PR by hand.
3. If they say yes: move the task into their layout (the file docstring already
   follows their header format), run the full 45 subjects on a GPU, and
   report micro and macro averages next to lm-eval's.
