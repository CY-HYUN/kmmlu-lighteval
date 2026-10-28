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
these times are not a speed benchmark. The GPU run below has its own times.

## All 45 subjects on a GPU

On 2026-10-28, team A of the 2026 Open-Source AI Contributor Challenge
(@GeonRyoung, @ojm6135, @hyeongrae-kim) ran `reproduce.py --subjects all --device cuda` at commit
`c731c1e`, with no code or setting changed: same model and revision, float32,
batch size 1, same dataset revision, 0-shot. GPU: NVIDIA GeForce RTX 4070 Ti
SUPER 16 GB, on Windows 11 under WSL2 Ubuntu (so the Windows file-name problem
in Limits did not apply). Wall time: lighteval 1,129 s, lm-eval 695 s, about
31 minutes with the comparison. Their `comparison.json` is committed as
`results/gpu_45_comparison.json` (per-subject counts and scores, no question
text).

| Subject | n | lighteval acc | lm-eval acc | diff (pp) | prompts differ | tokens differ | predictions differ | max abs log-prob diff | min top-2 margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accounting | 100 | 0.2500 | 0.2500 | +0.00 | 0 | 0 | 0 | 4.10e-05 | 1.15e-02 |
| Agricultural-Sciences | 1,000 | 0.3100 | 0.3100 | +0.00 | 0 | 0 | 0 | 2.34e-04 | 1.41e-03 |
| Aviation-Engineering-and-Maintenance | 1,000 | 0.3450 | 0.3450 | +0.00 | 0 | 0 | 0 | 1.65e-04 | 9.98e-04 |
| Biology | 1,000 | 0.2470 | 0.2470 | +0.00 | 0 | 0 | 0 | 5.72e-05 | 6.64e-04 |
| Chemical-Engineering * | 1,000 | 0.3140 | 0.3140 | +0.00 | 0 | 0 | 0 | 2.93e-04 | 6.39e-05 |
| Chemistry | 600 | 0.3000 | 0.3000 | +0.00 | 0 | 0 | 0 | 4.96e-05 | 2.00e-03 |
| Civil-Engineering | 1,000 | 0.3410 | 0.3410 | +0.00 | 0 | 0 | 0 | 7.25e-05 | 9.90e-04 |
| Computer-Science * | 1,000 | 0.4520 | 0.4520 | +0.00 | 0 | 0 | 0 | 2.60e-04 | 8.11e-05 |
| Construction | 1,000 | 0.3190 | 0.3190 | +0.00 | 0 | 0 | 0 | 1.88e-04 | 4.13e-04 |
| Criminal-Law | 200 | 0.2900 | 0.2900 | +0.00 | 0 | 0 | 0 | 5.63e-05 | 5.76e-04 |
| Ecology | 1,000 | 0.3500 | 0.3500 | +0.00 | 0 | 0 | 0 | 6.48e-05 | 6.39e-04 |
| Economics | 130 | 0.3077 | 0.3077 | +0.00 | 0 | 0 | 0 | 3.96e-05 | 9.94e-03 |
| Education | 100 | 0.3600 | 0.3600 | +0.00 | 0 | 0 | 0 | 5.58e-05 | 1.29e-03 |
| Electrical-Engineering * | 1,000 | 0.3300 | 0.3300 | +0.00 | 0 | 0 | 0 | 3.64e-04 | 3.66e-04 |
| Electronics-Engineering | 1,000 | 0.3740 | 0.3740 | +0.00 | 0 | 0 | 0 | 5.39e-05 | 3.70e-04 |
| Energy-Management | 1,000 | 0.2960 | 0.2960 | +0.00 | 0 | 0 | 0 | 6.20e-05 | 3.02e-04 |
| Environmental-Science | 1,000 | 0.3110 | 0.3110 | +0.00 | 0 | 0 | 0 | 5.39e-05 | 4.64e-04 |
| Fashion | 1,000 | 0.3140 | 0.3140 | +0.00 | 0 | 0 | 0 | 5.44e-05 | 9.52e-04 |
| Food-Processing | 1,000 | 0.2910 | 0.2910 | +0.00 | 0 | 0 | 0 | 5.58e-05 | 5.92e-04 |
| Gas-Technology-and-Engineering | 1,000 | 0.3020 | 0.3020 | +0.00 | 0 | 0 | 0 | 6.01e-05 | 6.69e-04 |
| Geomatics * | 1,000 | 0.3450 | 0.3450 | +0.00 | 0 | 0 | 0 | 1.51e-04 | 8.39e-05 |
| Health | 100 | 0.3300 | 0.3300 | +0.00 | 0 | 0 | 0 | 4.63e-05 | 7.74e-03 |
| Industrial-Engineer | 1,000 | 0.3400 | 0.3400 | +0.00 | 0 | 0 | 0 | 6.01e-05 | 4.22e-04 |
| Information-Technology | 1,000 | 0.4310 | 0.4310 | +0.00 | 0 | 0 | 0 | 5.20e-05 | 7.06e-04 |
| Interior-Architecture-and-Design | 1,000 | 0.3720 | 0.3720 | +0.00 | 0 | 0 | 0 | 5.01e-05 | 2.16e-04 |
| Korean-History | 100 | 0.2300 | 0.2300 | +0.00 | 0 | 0 | 0 | 3.72e-05 | 1.79e-02 |
| Law | 1,000 | 0.2970 | 0.2970 | +0.00 | 0 | 0 | 0 | 6.53e-05 | 1.56e-04 |
| Machine-Design-and-Manufacturing | 1,000 | 0.3240 | 0.3240 | +0.00 | 0 | 0 | 0 | 1.12e-04 | 7.92e-04 |
| Management * | 1,000 | 0.3120 | 0.3120 | +0.00 | 0 | 0 | 0 | 3.48e-03 | 2.01e-03 |
| Maritime-Engineering * | 600 | 0.3217 | 0.3217 | +0.00 | 0 | 0 | 0 | 5.05e-05 | 5.82e-05 |
| Marketing | 1,000 | 0.3820 | 0.3820 | +0.00 | 0 | 0 | 0 | 5.91e-05 | 1.82e-03 |
| Materials-Engineering * | 1,000 | 0.3010 | 0.3010 | +0.00 | 0 | 0 | 0 | 8.16e-04 | 8.68e-04 |
| Math | 300 | 0.2733 | 0.2733 | +0.00 | 0 | 0 | 0 | 3.77e-05 | 5.41e-04 |
| Mechanical-Engineering * | 1,000 | 0.3200 | 0.3200 | +0.00 | 0 | 0 | 0 | 3.90e-04 | 3.53e-05 |
| Nondestructive-Testing | 1,000 | 0.2960 | 0.2960 | +0.00 | 0 | 0 | 0 | 1.01e-04 | 8.82e-04 |
| Patent | 100 | 0.2600 | 0.2600 | +0.00 | 0 | 0 | 0 | 6.77e-05 | 1.17e-02 |
| Political-Science-and-Sociology | 300 | 0.2833 | 0.2833 | +0.00 | 0 | 0 | 0 | 4.48e-05 | 1.70e-03 |
| Psychology | 1,000 | 0.2750 | 0.2750 | +0.00 | 0 | 0 | 0 | 4.91e-05 | 1.67e-04 |
| Public-Safety * | 1,000 | 0.3080 | 0.3080 | +0.00 | 0 | 0 | 0 | 5.58e-05 | 1.81e-05 |
| Railway-and-Automotive-Engineering * | 1,000 | 0.3170 | 0.3170 | +0.00 | 0 | 0 | 0 | 6.48e-05 | 6.58e-05 |
| Real-Estate | 200 | 0.2750 | 0.2750 | +0.00 | 0 | 0 | 0 | 5.05e-05 | 7.99e-04 |
| Refrigerating-Machinery * | 1,000 | 0.2830 | 0.2830 | +0.00 | 0 | 0 | 0 | 6.68e-05 | 2.96e-05 |
| Social-Welfare * | 1,000 | 0.2930 | 0.2930 | +0.00 | 0 | 0 | 0 | 7.10e-04 | 5.82e-04 |
| Taxation | 200 | 0.2450 | 0.2450 | +0.00 | 0 | 0 | 0 | 7.77e-05 | 9.98e-04 |
| Telecommunications-and-Wireless-Technology | 1,000 | 0.3890 | 0.3890 | +0.00 | 0 | 0 | 0 | 5.39e-05 | 2.26e-04 |
| **All 45, micro** | **35,030** | **0.3245** | **0.3245** | **+0.00** | **0** | **0** | **0** | **3.48e-03** | **1.81e-05** |
| **All 45, macro** | | **0.3157** | **0.3157** | **+0.00** | | | | | |

All 35,030 questions: same prompt string, same tokens, same predicted choice,
so every subject has the same accuracy in both tools. The five CPU subjects
get the same accuracies as in the CPU table. Micro is the mean over all
questions (what lm-eval's `kmmlu` group reports), macro the mean over subjects
(what the KMMLU paper reports).

The log-probability gaps are larger on this GPU than on the CPU: up to
3.48e-03 (Management) against 5.53e-05. In the 12 subjects marked `*`, the
smallest top-2 margin is below twice the largest gap, so those two columns
alone cannot rule out a flipped prediction there (the two values can come from
different questions). The question-by-question count can: 0 predictions
differ.

The team ran PyPI lighteval 0.13.0 with this repo's `kmmlu_task.py`, not the
file in #1425. On 2026-10-28 both files were loaded side by side: the #1425
file at `76d6194` defines the same 45 log-likelihood tasks (dataset revision,
subject list, every task setting, and the prompt, choices and gold for a test
row). It adds the 45 `kmmlu_direct` tasks on top.

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

- The CPU table has five of 45 subjects, 700 questions, to fit a CPU budget
  of about 3 hours. They are five of the six subjects with the fewest prompt
  tokens (Economics was left out for Korean-History, the most Korea-specific
  one), not a random sample. All 45 subjects were run once, on one GPU, by the
  team named above. The accuracies (23% to 45%, chance is 25%) are for a 0.5B
  model; the point is the agreement, not the score.
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
  the KMMLU paper uses the macro average (mean over subjects). The GPU table
  gives both; the CPU table is per subject only.
- The dataset is CC BY-ND 4.0 (from its Hugging Face card). Per-question files
  contain the questions, so they stay out of git; rerun to get them.
- Only the default log-likelihood variant is ported here. The upstream pull
  request #1425 also carries `kmmlu_direct` (generation, five-shot), added on
  2026-10-28 by team A of the 2026 Open-Source AI Contributor Challenge
  (CY-HYUN/lighteval#1). `kmmlu_hard` and chain-of-thought variants are in
  neither.

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
   says an AI coding assistant was used. The 45-subject GPU run was done on 2026-10-28 (above). Next: answer review comments. lighteval has no CONTRIBUTING file (checked
   2026-10-08), but `huggingface/transformers` CONTRIBUTING.md asks
   first-time contributors not to use code agents to create issues or PRs.
   Assume the same norm here: write the issue and any PR by hand.
3. If they say yes: move the task into their layout (the file docstring already
   follows their header format).
