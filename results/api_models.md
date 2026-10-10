# KMMLU through the Claude API, generative protocol (collected 2026-10-10)

**Protocol.** Each question is sent once, zero-shot, as the user turn `kmmlu_task.kmmlu_prompt(row).query` (the exact lm-eval `kmmlu` prompt, full-width colon included), with the system prompt "This is a Korean multiple-choice exam. Reply with the letter of the correct choice only: A, B, C or D.", a structured-output schema `{"answer": A|B|C|D}` and `max_tokens=32`; the first A-D in the reply is the prediction, a reply with no letter counts as wrong. A first attempt the same day (results/api/attempt1_free_text/, $0.27) asked for the letter in free text with `max_tokens=8` and no schema: Haiku 4.5 began a worked solution on 689 of 700 questions and Sonnet 5.5 on 340, so the replies held no letter. A stronger instruction changed nothing on a 12-question check; the schema did. No thinking pass (Haiku 4.5 has none unless given a budget; Sonnet 5.5 runs with `thinking: between_tools`, effort low). **This differs from the log-likelihood protocol of lm-eval and the lighteval task in the README**, which score the log-probability of each of the four choices and take the highest; the API does not expose those, so the accuracies here are not comparable with the README table or the published KMMLU numbers (they are closer to lm-eval's `kmmlu_direct` variant, which is not run here). Dataset `HAERAE-HUB/KMMLU` at `d61b3f1`, test split, Message Batches API. Spend is computed from each reply's `usage` at the batch prices in `scripts/api_eval.py`.

## All 45 subjects, side by side

| Subject | n | claude-haiku-4-5-20251001 | claude-sonnet-5-5 |
|---|---:|---:|---:|
| Accounting | 100 | 0.7000 | 0.9200 |
| Agricultural-Sciences | 1000 | 0.5490 | 0.7270 |
| Aviation-Engineering-and-Maintenance | 1000 | 0.6900 | 0.8290 |
| Biology | 1000 | 0.6210 | 0.8470 |
| Chemical-Engineering | 1000 | 0.6940 | 0.8300 |
| Chemistry | 600 | 0.7733 | 0.8933 |
| Civil-Engineering | 1000 | 0.6240 | 0.7580 |
| Computer-Science | 1000 | 0.8770 | 0.9460 |
| Construction | 1000 | 0.5370 | 0.7380 |
| Criminal-Law | 200 | 0.4850 | 0.7150 |
| Ecology | 1000 | 0.6690 | 0.7850 |
| Economics | 130 | 0.7385 | 0.9462 |
| Education | 100 | 0.8500 | 0.9500 |
| Electrical-Engineering | 1000 | 0.5110 | 0.7350 |
| Electronics-Engineering | 1000 | 0.8110 | 0.9180 |
| Energy-Management | 1000 | 0.5260 | 0.7340 |
| Environmental-Science | 1000 | 0.5010 | 0.6770 |
| Fashion | 1000 | 0.6250 | 0.7660 |
| Food-Processing | 1000 | 0.6410 | 0.7890 |
| Gas-Technology-and-Engineering | 1000 | 0.5550 | 0.7360 |
| Geomatics | 1000 | 0.5910 | 0.7560 |
| Health | 100 | 0.7900 | 0.9300 |
| Industrial-Engineer | 1000 | 0.6560 | 0.7510 |
| Information-Technology | 1000 | 0.9100 | 0.9410 |
| Interior-Architecture-and-Design | 1000 | 0.7600 | 0.9220 |
| Korean-History | 100 | 0.4800 | 0.8000 |
| Law | 1000 | 0.6130 | 0.7840 |
| Machine-Design-and-Manufacturing | 1000 | 0.7260 | 0.8780 |
| Management | 1000 | 0.7730 | 0.8910 |
| Maritime-Engineering | 600 | 0.6883 | 0.8683 |
| Marketing | 1000 | 0.8950 | 0.9320 |
| Materials-Engineering | 1000 | 0.7120 | 0.8860 |
| Math | 300 | 0.4367 | 0.4933 |
| Mechanical-Engineering | 1000 | 0.6530 | 0.8440 |
| Nondestructive-Testing | 1000 | 0.6380 | 0.8070 |
| Patent | 100 | 0.4900 | 0.7800 |
| Political-Science-and-Sociology | 300 | 0.7967 | 0.9033 |
| Psychology | 1000 | 0.6630 | 0.8540 |
| Public-Safety | 1000 | 0.5700 | 0.7680 |
| Railway-and-Automotive-Engineering | 1000 | 0.5920 | 0.7530 |
| Real-Estate | 200 | 0.5950 | 0.7950 |
| Refrigerating-Machinery | 1000 | 0.5990 | 0.7960 |
| Social-Welfare | 1000 | 0.8330 | 0.9000 |
| Taxation | 200 | 0.4600 | 0.6750 |
| Telecommunications-and-Wireless-Technology | 1000 | 0.8090 | 0.9010 |
| **micro** | 35030 | 0.6682 | 0.8180 |
| **macro** | 45 | 0.6602 | 0.8189 |

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
| Agricultural-Sciences | 1000 | 0.7270 | 1 | 0 | `msgbatch_01VCwTBq1Xot7eWRVKYcaYWw` |
| Aviation-Engineering-and-Maintenance | 1000 | 0.8290 | 0 | 0 | `msgbatch_01YSjoJwKXR8LAQqcsCBcxhx` |
| Biology | 1000 | 0.8470 | 0 | 0 | `msgbatch_019X47Y9Lqgtyi6aWZQFaJZ7` |
| Chemical-Engineering | 1000 | 0.8300 | 0 | 0 | `msgbatch_01S4biCYNFtiJJoCVXDu7h38` |
| Chemistry | 600 | 0.8933 | 0 | 0 | `msgbatch_016rXhFwb91ESZoitpaM3CB6` |
| Civil-Engineering | 1000 | 0.7580 | 0 | 0 | `msgbatch_01QjT2jeqAk5EHEKQqZFGJD5` |
| Computer-Science | 1000 | 0.9460 | 0 | 0 | `msgbatch_01NEJenWdd67gvPwivmwXEyy` |
| Construction | 1000 | 0.7380 | 0 | 0 | `msgbatch_01C6Dhbndn7JnJLKHVncKJuB` |
| Criminal-Law | 200 | 0.7150 | 0 | 0 | `msgbatch_01EeSyKju2obXN7osrNjdDdZ` |
| Ecology | 1000 | 0.7850 | 0 | 0 | `msgbatch_01HNp5Ay8agadrhPZP9pDDnu` |
| Economics | 130 | 0.9462 | 0 | 0 | `msgbatch_0188A8JBQ94D2xUjvYsevaX1` |
| Education | 100 | 0.9500 | 0 | 0 | `msgbatch_01CbQcXzngnkEBGo1BB7PQNA` |
| Electrical-Engineering | 1000 | 0.7350 | 0 | 0 | `msgbatch_01Xc3Yy3jEnYinQUYrmVrcJU` |
| Electronics-Engineering | 1000 | 0.9180 | 0 | 0 | `msgbatch_016PdZnc8oVVz4dVrhAwSnor` |
| Energy-Management | 1000 | 0.7340 | 0 | 0 | `msgbatch_01DoReD5d2h9t5MjQvfUwsoj` |
| Environmental-Science | 1000 | 0.6770 | 0 | 0 | `msgbatch_011HgFbxHAFHPwBwsiVhLvcp` |
| Fashion | 1000 | 0.7660 | 0 | 0 | `msgbatch_019kFPSa4JfS4tvxnkGVUhpX` |
| Food-Processing | 1000 | 0.7890 | 0 | 0 | `msgbatch_01NjvcEJb53PsGiMAqS6oftD` |
| Gas-Technology-and-Engineering | 1000 | 0.7360 | 0 | 0 | `msgbatch_017Ji8icXaS9QsW7UXiLpNzM` |
| Geomatics | 1000 | 0.7560 | 0 | 0 | `msgbatch_01JR5qyJMA7CP8w6duDU95cH` |
| Health | 100 | 0.9300 | 0 | 0 | `msgbatch_01Kbw14onRkF5QAM2f7VYg4i` |
| Industrial-Engineer | 1000 | 0.7510 | 0 | 0 | `msgbatch_01JjbTP9rH7ekMzzA8aXFgFd` |
| Information-Technology | 1000 | 0.9410 | 0 | 0 | `msgbatch_01EqsKCJUGmfqv9MeWKRsPq5` |
| Interior-Architecture-and-Design | 1000 | 0.9220 | 0 | 0 | `msgbatch_017tHCK5iuVGPysm5NMjzX6D` |
| Korean-History | 100 | 0.8000 | 0 | 0 | `msgbatch_01K7voscPLSWPpnwESmGgZ4L` |
| Law | 1000 | 0.7840 | 0 | 0 | `msgbatch_01HbWLyJ2bAG4Dr2sJhsv757` |
| Machine-Design-and-Manufacturing | 1000 | 0.8780 | 0 | 0 | `msgbatch_0114eCyWfww1Zipyv2MNSNf2` |
| Management | 1000 | 0.8910 | 0 | 0 | `msgbatch_018oxzPCHQS872JmJEuCYKko` |
| Maritime-Engineering | 600 | 0.8683 | 0 | 0 | `msgbatch_01NgmCa2WxGFUxCrpS7jkw4Z` |
| Marketing | 1000 | 0.9320 | 0 | 0 | `msgbatch_01FBdTetk7TeN5ZpKUySrdSe` |
| Materials-Engineering | 1000 | 0.8860 | 0 | 0 | `msgbatch_019YjEWQ2EZAdJAfUnQDG8so` |
| Math | 300 | 0.4933 | 0 | 0 | `msgbatch_01AxcYpdNmdzAXzsH4QSUdLo` |
| Mechanical-Engineering | 1000 | 0.8440 | 0 | 0 | `msgbatch_01QWQuj1Ms1SXgjQVSBPckQG` |
| Nondestructive-Testing | 1000 | 0.8070 | 0 | 0 | `msgbatch_01RmnJEVsX3NJpQFphXn9RTe` |
| Patent | 100 | 0.7800 | 0 | 0 | `msgbatch_01RCccsoJFHyg83zDEeRZE7q` |
| Political-Science-and-Sociology | 300 | 0.9033 | 0 | 0 | `msgbatch_01RHLxc2LZa6NjtrmnpnbWoj` |
| Psychology | 1000 | 0.8540 | 0 | 0 | `msgbatch_013eKbSrfjrPatCY53GPiwq9` |
| Public-Safety | 1000 | 0.7680 | 0 | 0 | `msgbatch_01Vm9eJjPLjVVwAucugCfb1z` |
| Railway-and-Automotive-Engineering | 1000 | 0.7530 | 0 | 0 | `msgbatch_017n8Nqg77e8EVdu1U5fDGQ8` |
| Real-Estate | 200 | 0.7950 | 0 | 0 | `msgbatch_01KpP7nADMgJHWRituFT4REi` |
| Refrigerating-Machinery | 1000 | 0.7960 | 0 | 0 | `msgbatch_01RjULjhBQmUnU2714usjGe9` |
| Social-Welfare | 1000 | 0.9000 | 0 | 0 | `msgbatch_01MHM75TeTA7rufzhhL2yCg3` |
| Taxation | 200 | 0.6750 | 0 | 0 | `msgbatch_011r2YPyzJ3RvqXT89RwCXo7` |
| Telecommunications-and-Wireless-Technology | 1000 | 0.9010 | 0 | 0 | `msgbatch_011HTmShDR589qARtxuF5fJn` |
| **README 5 subjects (micro)** | 700 | 0.7257 | 0 | 0 | |
| **all 45 (micro)** | 35030 | 0.8180 | 1 | 0 | |
| **all 45 (macro, mean of subjects)** | 45 | 0.8189 | | | |

Spend: **$15.87** (14.11 M input, 351171 output tokens) over 45 subjects.

- **acc**: correct / n; unparsed replies and failed requests are in n and count as wrong.
- **unparsed**: replies with no A-D letter. **failed**: requests the batch did not complete.
- micro pools all questions; macro is the mean of the per-subject accuracies (the KMMLU paper reports macro).
