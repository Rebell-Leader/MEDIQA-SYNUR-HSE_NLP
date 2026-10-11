# Generated Paper Tables

## Table 1: SYNUR Dataset Statistics
| Split   | Examples   | Avg Tokens   | Avg Obs   | Unique Concepts   | % SINGLE_SELECT   | % MULTI_SELECT   | % NUMERIC   | % STRING   |
|:--------|:-----------|:-------------|:----------|:------------------|:------------------|:-----------------|:------------|:-----------|
| Train   | 122        | 275.3        | 13.8      | 166               | 71.2%             | 11.6%            | 10.8%       | 6.4%       |
| Dev     | 101        | 279.9        | 13.0      | 170               | 72.3%             | 11.9%            | 10.4%       | 5.3%       |
| Test    | 199        | 259.3        | 12.8      | 164               | 72.6%             | 13.5%            | 10.1%       | 3.8%       |

## Table 2 & 4: System Performance Summary
| Model                      |   Precision |   Recall |     F1 | Type               |
|:---------------------------|------------:|---------:|-------:|:-------------------|
| GPT-4o 0-shot              |      0.741  |   0.7268 | 0.7338 | Baseline           |
| Qwen-235B 0-shot           |      0.765  |   0.7404 | 0.7525 | Baseline           |
| GPT-5 0-shot               |      0.789  |   0.7556 | 0.7719 | Baseline           |
| Qwen-235B 5-shot RAG       |      0.7801 |   0.8076 | 0.7936 | Baseline (Strong)  |
| V3 Union (Ensemble)        |      0.4618 |   0.7449 | 0.5702 | Ablation           |
| V4 Adjudication (Ensemble) |      0.7352 |   0.6582 | 0.6946 | Ablation           |
| V6 ACE Full                |      0.7397 |   0.6413 | 0.687  | Primary Submission |

## Table 5: Top Error Types (Test Set)
| type        |   obs_id |   count |
|:------------|---------:|--------:|
| FN          |        3 |      73 |
| FN          |      148 |      63 |
| Value Error |      177 |      51 |
| Value Error |      139 |      37 |
| FP          |      177 |      32 |
| FP          |       31 |      31 |
| Value Error |       13 |      31 |
| Value Error |      149 |      30 |
| FP          |       13 |      29 |
| Value Error |      148 |      29 |
| FP          |        3 |      25 |
| FP          |      139 |      24 |
| Value Error |       84 |      24 |
| FN          |      177 |      24 |
| FN          |        0 |      23 |
