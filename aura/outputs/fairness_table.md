# Group-Wise Fairness and Utility Evaluation Table

### Model: FairStudentPrioritizer (Calibrated Ensemble + Pareto Alignment)

- **Total Evaluated Rows**: 395
- **Support Capacity (Top-20%)**: 79 students
- **Overall Recall**: 60.77%
- **Worst-Group Recall (R_min)**: 58.67% (sex:F)
- **Overall Fairness Gap**: 4.97%
- **Brier Score / Calibration Utility**: 0.0374 / 0.9626
- **Total Judging Score**: 72.61 / 100

## Subgroup Performance Metrics

| Attribute | Group | Total Rows | Positive Rows | Selected Rows | Eligible? | Group Recall | Precision | Selection Rate |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| sex | F | 208 | 75 | 44 | Yes | 58.7% | 100.0% | 21.2% |
| sex | M | 187 | 55 | 35 | Yes | 63.6% | 100.0% | 18.7% |
| school | GP | 349 | 113 | 68 | Yes | 60.2% | 100.0% | 19.5% |
| school | MS | 46 | 17 | 11 | Yes | 64.7% | 100.0% | 23.9% |

---
*Criteria: A group is eligible when containing >= 10 rows and >= 3 positive examples.*
