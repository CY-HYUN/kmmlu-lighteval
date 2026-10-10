# 초안 — CY가 읽고 게시 (draft, CY reads and posts by hand)

> Target: a new issue on https://github.com/huggingface/lighteval with their "Evaluation/task request" template (title prefix `[EVAL]`, label `new task`; template fields are the headings below). lighteval's README says "If you're adding a new feature, please open an issue first." (read 2026-10-10), so this goes before any pull request.
> Checked 2026-10-10 with `gh`: `gh search issues --repo huggingface/lighteval KMMLU` → 0, `gh search prs --repo huggingface/lighteval KMMLU` → 0, `gh search code --repo huggingface/lighteval kmmlu` → 0, `gh search issues --repo huggingface/lighteval korean` → 0. `main` was `7fa19dc` (2026-10-06). Korean appears in 8 files (`Language.KOREAN`): `global_mmlu.py`, `openai_mmlu.py`, `mkqa.py`, `paws_x.py`, `tydiqa.py` and three utils.
> Before posting: re-run the four `gh search` commands (counts can change), replace `<repo link>` with the public URL of this repo, and keep the body in the first person.

---

**Title:** `[EVAL] KMMLU (Korean-native MMLU, 45 subjects) — would a PR be welcome?`

## Evaluation short description

KMMLU is a Korean multiple-choice benchmark: 35,030 expert-level questions in 45 subjects, collected from original Korean exams rather than translated from English (paper: arXiv 2402.11548, "KMMLU: Measuring Massive Multitask Language Understanding in Korean", Son et al.). EleutherAI's lm-evaluation-harness ships it as the `kmmlu` task.

- Why is this evaluation interesting? lighteval already scores Korean, but only through sets that are translated or built across many languages (`global_mmlu`, `openai_mmlu`, `mkqa`, `paws_x`, `tydiqa`; `Language.KOREAN` appears in 8 files on `main` at 7fa19dc, 2026-10-06). There is no Korean-native knowledge benchmark. KMMLU is the one most Korean model cards report, and it is the natural counterpart to the MMLU family that lighteval already has.
- How used is it in the community? It is the default Korean knowledge benchmark in lm-eval and on the HAERAE-HUB dataset page; the paper has the usage figures. I have not measured citation counts myself.

I checked for prior work here first: on 2026-10-10 GitHub search returned no issue, pull request or code hit for `KMMLU` in this repo, and no issue for `korean`. The only Korean-related PR I found is #1179, which fixes Korean translation literals.

## What I have already done

I wrote KMMLU as a custom task (`--custom-tasks`) that copies lm-eval's `kmmlu` default variant part by part: same prompt string (`{{question.strip()}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n정답：`, full-width colon), same four choices `" A"`–`" D"`, gold `answer - 1`, log-likelihood accuracy without length normalisation, 0-shot, chat template off. A small test renders lm-eval's own template for the first rows and checks that all 45 subjects map to lm-eval task names.

Then I ran both tools on the same model and compared per question. Model `Qwen/Qwen2.5-0.5B-Instruct` (revision `7ae5576`), float32, CPU, batch size 1; dataset `HAERAE-HUB/KMMLU` at `d61b3f1`; whole test split, 0-shot; lighteval 0.13.0 and lm-eval 0.4.13.

| Subject | n | lighteval acc | lm-eval acc | diff (pp) | prompts differ | tokens differ | predictions differ | max abs log-prob diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Health | 100 | 0.3300 | 0.3300 | +0.00 | 0 | 0 | 0 | 5.15e-05 |
| Education | 100 | 0.3600 | 0.3600 | +0.00 | 0 | 0 | 0 | 5.53e-05 |
| Korean-History | 100 | 0.2300 | 0.2300 | +0.00 | 0 | 0 | 0 | 4.01e-05 |
| Accounting | 100 | 0.2500 | 0.2500 | +0.00 | 0 | 0 | 0 | 5.05e-05 |
| Math | 300 | 0.2733 | 0.2733 | +0.00 | 0 | 0 | 0 | 4.24e-05 |

All 700 questions: same prompt, same tokens, same predicted choice. Log-probabilities differ by at most 5.53e-05, which looks like float32 rounding (lighteval batches the four choices, lm-eval scores them one by one). Code, test and a `reproduce.py` that runs both tools and prints this table: `<repo link>`.

Limits I know about: five of 45 subjects (the ones with the fewest prompt tokens, to fit a CPU budget), one model and one tokenizer, and only the log-likelihood variant (lm-eval also has `kmmlu_direct`, `kmmlu_hard` and chain-of-thought variants). lighteval always tokenizes prompt and choice separately (`pairwise=True` in `TransformersModel._loglikelihood_tokens`), lm-eval tokenizes them together and splits; the two agree only when no token crosses the boundary, and for the Qwen2.5 tokenizer none did in these 700 questions.

## Proposal

A new file `src/lighteval/tasks/tasks/kmmlu.py` following the "Adding a custom task" guide (prompt function → `Doc`, one `LightevalTaskConfig` per subject, `TASKS_TABLE`), task names `kmmlu:<subject>|0` for the 45 HF configs, metric `loglikelihood_acc`. Two questions before I write it:

1. Would a PR be welcome, and should it keep the lm-eval prompt (so the two tools stay comparable) or use the multilingual template style that `global_mmlu` uses?
2. Would you also want HAE-RAE Bench and KoBEST, the other two Korean-native sets lm-eval ships, in the same PR or as separate issues?

If yes, I would run all 45 subjects on a GPU and report micro and macro averages next to lm-eval's before opening the PR.

## Evaluation metadata

- Paper url: https://arxiv.org/abs/2402.11548
- Github url: `<repo link>` (comparison code); lm-eval task: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/kmmlu
- Dataset url: https://huggingface.co/datasets/HAERAE-HUB/KMMLU (CC BY-ND 4.0 per the dataset card)

---

## 한국어 미러 (CY용, 게시하지 않음)

- **무엇을 올리나**: lighteval에 KMMLU 태스크를 제안하는 이슈. 템플릿 "Evaluation/task request"(제목 `[EVAL]`, 라벨 `new task`)의 칸을 그대로 썼다. README가 "새 기능은 이슈 먼저"라고 하니 PR보다 이슈가 먼저다.
- **먼저 확인한 것(2026-10-10)**: 이 레포에 KMMLU 이슈 0·PR 0·코드 0, "korean" 이슈 0. 한국어는 번역·다국어 세트 5개(`global_mmlu`·`openai_mmlu`·`mkqa`·`paws_x`·`tydiqa`)에만 있다. 한국어 PR은 #1179(번역 문자열 수정) 하나.
- **내가 한 것**: lm-eval의 `kmmlu` 기본 변형을 그대로 옮긴 커스텀 태스크 + 같은 모델(Qwen2.5-0.5B-Instruct)로 5과목 700문항을 두 도구에서 돌려 문항 단위로 비교 → 예측 전부 일치, 로그확률 차이 최대 5.53e-05(float32 반올림 수준).
- **묻는 것 둘**: ① PR을 받을지, 받으면 lm-eval 프롬프트를 유지할지 lighteval 다국어 템플릿으로 갈지 ② HAE-RAE Bench·KoBEST도 같이 원하는지.
- **한계를 적은 이유**: 5/45 과목·모델 1개·토크나이저 1개·log-likelihood 변형만. 숨기면 리뷰어가 먼저 찾는다.
- **게시 전 할 일**: `gh search` 4개 다시 돌리기(숫자가 바뀔 수 있다), `<repo link>` 채우기, 레포 README의 표와 숫자가 같은지 한 번 대조(이 초안의 표는 README 2026-10-08 판을 옮긴 것).
