# Results reported in the paper

Source: Table 1 (page 7) of **Curriculum Proof Repair with Learned Counterexamples**.
Each cell is **Safe-Pass@1 / Safe-Pass@3 (%)**, transcribed as reported.
The paper evaluates seven benchmark suites and reports 471 test problems.

| Approach | VerusBench | DafnyBench | HumanEval | LeetCode | VeriCogen | ObfsBench | VeruSAGE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sonnet-4.5 | 73.1 / 84.6 | 70.6 / 76.5 | 13.3 / 13.3 | 40.0 / 40.0 | 26.7 / 30.0 | 62.0 / 75.2 | 75.5 / 84.9 |
| GPT-4o | 42.3 / 42.3 | 76.5 / 76.5 | 6.7 / 13.3 | 0.0 / 0.0 | 40.4 / 41.4 | 21.8 / 22.9 | 17.0 / 24.5 |
| Qwen3-Coder | 34.6 / 34.6 | 52.9 / 52.9 | 6.7 / 6.7 | 0.0 / 0.0 | 20.0 / 25.6 | 14.3 / 18.8 | 45.3 / 64.2 |
| DeepSeek-V4 | 23.1 / 38.5 | 47.1 / 82.4 | 13.3 / 20.0 | 0.0 / 0.0 | 28.9 / 36.7 | 12.8 / 29.3 | 58.5 / 66.0 |
| AutoVerus | 38.5 / 50.0 | 58.8 / 70.6 | 6.7 / 6.7 | 20.0 / 20.0 | 22.5 / 28.1 | 13.2 / 23.7 | 7.5 / 9.4 |
| CoT-Init | 46.2 / 50.0 | 82.4 / 100 | 20.0 / 26.7 | 0.0 / 0.0 | 38.2 / 48.3 | 42.9 / 60.9 | 15.1 / 24.5 |
| CTX-Init | 88.5 / 92.3 | 88.2 / 100 | 26.7 / 26.7 | 20.0 / 40.0 | 48.3 / 61.8 | 55.3 / 73.3 | 9.4 / 22.6 |
| CTX-Binary | 88.5 / 96.2 | 94.1 / 100 | 26.7 / 26.7 | 20.0 / 40.0 | 44.9 / 60.7 | 58.6 / 75.9 | 24.5 / 30.2 |
| CTX-Dep | 88.5 / 88.5 | 100 / 100 | 20.0 / 33.3 | 20.0 / 20.0 | 49.4 / 65.2 | 69.9 / 80.1 | 30.2 / 43.4 |

The paper reports weighted-average Safe-Pass@1 / Safe-Pass@3 of **61.6% / 72.2%**
for CTX-Dep (TopoVerus). Benchmark sizes reported in Section 4.1 are:

| Benchmark | Problems |
| --- | ---: |
| VerusBench | 26 |
| DafnyBench | 17 |
| HumanEval | 15 |
| LeetCode | 5 |
| VeriCogen | 89 |
| ObfsBench | 266 |
| VeruSAGE | 53 |
| Total | 471 |

The paper's standard model evaluation uses three independent repair chains with
up to ten attempts per chain and iterative verifier feedback. Its Safe-Pass
metric applies the paper's Lynette checks. These are the reported paper results.

[paper-results.csv](paper-results.csv) contains the same values in long form;
percentage columns use the 0–100 scale.
