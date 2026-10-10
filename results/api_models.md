# KMMLU through the Claude API, generative protocol (collected 2026-10-10)

**Protocol.** Each question is sent once, zero-shot, as the user turn `kmmlu_task.kmmlu_prompt(row).query` (the exact lm-eval `kmmlu` prompt, full-width colon included), with the system prompt "This is a Korean multiple-choice exam. Reply with the letter of the correct choice only: A, B, C or D.", a structured-output schema `{"answer": A|B|C|D}` and `max_tokens=32`; the first A-D in the reply is the prediction, a reply with no letter counts as wrong. A first attempt the same day (results/api/attempt1_free_text/, $0.27) asked for the letter in free text with `max_tokens=8` and no schema: Haiku 4.5 began a worked solution on 689 of 700 questions and Sonnet 5.5 on 340, so the replies held no letter. A stronger instruction changed nothing on a 12-question check; the schema did. No thinking pass (Haiku 4.5 has none unless given a budget; Sonnet 5.5 runs with `thinking: between_tools`, effort low). **This differs from the log-likelihood protocol of lm-eval and the lighteval task in the README**, which score the log-probability of each of the four choices and take the highest; the API does not expose those, so the accuracies here are not comparable with the README table or the published KMMLU numbers (they are closer to lm-eval's `kmmlu_direct` variant, which is not run here). Dataset `HAERAE-HUB/KMMLU` at `d61b3f1`, test split, Message Batches API. Spend is computed from each reply's `usage` at the batch prices in `scripts/api_eval.py`.

## claude-haiku-4-5-20251001

| Subject | n | acc | unparsed | failed | batch |
|---|---:|---:|---:|---:|---|
| Accounting | 100 | 0.7000 | 0 | 0 | `msgbatch_012FDTsNJGUpEyQigwUPGfPY` |
| Agricultural-Sciences | 1000 | 0.5490 | 0 | 0 | `msgbatch_01JGrQVtJ3DZKEH9utLTRhLT` |
| Aviation-Engineering-and-Maintenance | 1000 | 0.6900 | 0 | 0 | `msgbatch_0142nUxHrbybn1v2u1UKJqVx` |
| Biology | 1000 | 0.6210 | 0 | 0 | `msgbatch_01EU2L8gTdvZZSUR8iu4uSK8` |
| Chemical-Engineering | 1000 | 0.6940 | 0 | 0 | `msgbatch_018PZEf3cnKQG7rKnriWjaC2` |
| Chemistry | 600 | 0.7733 | 0 | 0 | `msgbatch_01CsD8vhU7y9wyKP6zkuYVyW` |
| Civil-Engineering | 1000 | 0.6240 | 0 | 0 | `msgbatch_01SQjKMNRwMpMcbzBuhDjqKc` |
| Computer-Science | 1000 | 0.8770 | 0 | 0 | `msgbatch_0119UneWprk1bPTgpwjgkFFd` |
| Construction | 1000 | 0.5370 | 0 | 0 | `msgbatch_01AXySih6nQC21Ln6pEVdcqd` |
| Criminal-Law | 200 | 0.4850 | 0 | 0 | `msgbatch_01YNxWLGruhMkDEQnGMD9j1p` |
| Ecology | 1000 | 0.6690 | 0 | 0 | `msgbatch_01WF41QPSBeZ3wtAjmDRQFWT` |
| Economics | 130 | 0.7385 | 0 | 0 | `msgbatch_01NTqkneQfyBHGyJofBtPhSz` |
| Education | 100 | 0.8500 | 0 | 0 | `msgbatch_01W4cNSMtDUE34VRScYdawSv` |
| Electrical-Engineering | 1000 | 0.5110 | 0 | 0 | `msgbatch_01CNLsxUwZvCq4xdEGkAcX7c` |
| Electronics-Engineering | 1000 | 0.8110 | 0 | 0 | `msgbatch_01W2XK3QpGrhaBbQ84cpFtLn` |
| Energy-Management | 1000 | 0.5260 | 0 | 0 | `msgbatch_01GxsyQCggt2jtMx86aQkrNq` |
| Environmental-Science | 1000 | 0.5010 | 0 | 0 | `msgbatch_01BMUuY6rQCeLPvB99DJTpXM` |
| Fashion | 1000 | 0.6250 | 0 | 0 | `msgbatch_015tRTZEkhZzkRywJHCeSWzu` |
| Food-Processing | 1000 | 0.6410 | 0 | 0 | `msgbatch_014tJfXsBoC3qbDCyWqM8jH4` |
| Gas-Technology-and-Engineering | 1000 | 0.5550 | 0 | 0 | `msgbatch_01QkTBfbutX1pAuhBfnVox16` |
| Geomatics | 1000 | 0.5910 | 0 | 0 | `msgbatch_01Swq4JifChVCfwkP4Cygich` |
| Health | 100 | 0.7900 | 0 | 0 | `msgbatch_011X8ZbPhV2EGovPq4WnU7kS` |
| Industrial-Engineer | 1000 | 0.6560 | 0 | 0 | `msgbatch_01SsX3K8j7kP6FiJJZBb9erw` |
| Information-Technology | 1000 | 0.9100 | 0 | 0 | `msgbatch_019j61D2DAEtRWULDqE7mUSr` |
| Interior-Architecture-and-Design | 1000 | 0.7600 | 0 | 0 | `msgbatch_0175PbuQwurEoSCmrwqzfPrv` |
| Korean-History | 100 | 0.4800 | 0 | 0 | `msgbatch_01UadZBg2p2yZoEMvx92KQfW` |
| Law | 1000 | 0.6130 | 0 | 0 | `msgbatch_01TLBqsBdUNJqQ17k7zqeHKJ` |
| Machine-Design-and-Manufacturing | 1000 | 0.7260 | 0 | 0 | `msgbatch_01884XzpFkFEofBc5eUvP8Yv` |
| Management | 1000 | 0.7730 | 0 | 0 | `msgbatch_013kEEzRd4Yef1VFUCHtdmZQ` |
| Maritime-Engineering | 600 | 0.6883 | 0 | 0 | `msgbatch_01Qx2RrK9ttRKLnbfnJ9E5ri` |
| Marketing | 1000 | 0.8950 | 0 | 0 | `msgbatch_01Vh1B6YTfZWw1VGWH2EZd1W` |
| Materials-Engineering | 1000 | 0.7120 | 0 | 0 | `msgbatch_01GBYMpg4m4pzjN5y4qz62N6` |
| Math | 300 | 0.4367 | 0 | 0 | `msgbatch_01884t4jS9xR2FwtA2riq2Eb` |
| Mechanical-Engineering | 1000 | 0.6530 | 0 | 0 | `msgbatch_01DWi7GUSu8kvURWYnkih9YA` |
| Nondestructive-Testing | 1000 | 0.6380 | 0 | 0 | `msgbatch_0154dP2668FMvQn1zNVneq7r` |
| Patent | 100 | 0.4900 | 0 | 0 | `msgbatch_01SDV3mzQKdv4TZyAyKtQoYN` |
| Political-Science-and-Sociology | 300 | 0.7967 | 0 | 0 | `msgbatch_011Bh7KzQb3DcRUJPyWMWJzM` |
| Psychology | 1000 | 0.6630 | 0 | 0 | `msgbatch_01PnhTJTKciyE4FLMQXjdG5x` |
| Public-Safety | 1000 | 0.5700 | 0 | 0 | `msgbatch_01SmyzLubyrX5TGyDFbgKSfX` |
| Railway-and-Automotive-Engineering | 1000 | 0.5920 | 0 | 0 | `msgbatch_01Awjx6vEjPsw1tuw1e1fcHs` |
| Real-Estate | 200 | 0.5950 | 0 | 0 | `msgbatch_012fgWzUpWQb5vHq5Z95e1Yg` |
| Refrigerating-Machinery | 1000 | 0.5990 | 0 | 0 | `msgbatch_014qbTbDrEnzXXPqfGiQ7z1S` |
| Social-Welfare | 1000 | 0.8330 | 0 | 0 | `msgbatch_01VDeve5ZViLybXAYFZHTtxf` |
| Taxation | 200 | 0.4600 | 0 | 0 | `msgbatch_01X5zAPRjoY2BW7KGGTjzdmd` |
| Telecommunications-and-Wireless-Technology | 1000 | 0.8090 | 0 | 0 | `msgbatch_01MtWFM8uTre8sPZCqv3bVXW` |
| **README 5 subjects (micro)** | 700 | 0.5900 | 0 | 0 | |
| **all 45 (micro)** | 35030 | 0.6682 | 0 | 0 | |
| **all 45 (macro, mean of subjects)** | 45 | 0.6602 | | | |

Spend: **$6.89** (12.01 M input, 351505 output tokens) over 45 subjects.

## claude-sonnet-5-5

| Subject | n | acc | unparsed | failed | batch |
|---|---:|---:|---:|---:|---|
| Accounting | 100 | 0.9200 | 0 | 0 | `msgbatch_017t3zCeaUSHbsUD5min9SHb` |
| Education | 100 | 0.9500 | 0 | 0 | `msgbatch_01CbQcXzngnkEBGo1BB7PQNA` |
| Health | 100 | 0.9300 | 0 | 0 | `msgbatch_01Kbw14onRkF5QAM2f7VYg4i` |
| Korean-History | 100 | 0.8000 | 0 | 0 | `msgbatch_01K7voscPLSWPpnwESmGgZ4L` |
| Math | 300 | 0.4933 | 0 | 0 | `msgbatch_01AxcYpdNmdzAXzsH4QSUdLo` |
| **README 5 subjects (micro)** | 700 | 0.7257 | 0 | 0 | |

Spend: **$0.34** (0.31 M input, 7100 output tokens) over 5 subjects.

- **acc**: correct / n; unparsed replies and failed requests are in n and count as wrong.
- **unparsed**: replies with no A-D letter. **failed**: requests the batch did not complete.
- micro pools all questions; macro is the mean of the per-subject accuracies (the KMMLU paper reports macro).
