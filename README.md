# AURA — Fair Student-Support Prioritization

**Equitable Student Prioritization powered by Explainable and Responsible AI**

AURA is a machine-learning-based student-support prioritization system designed to help educational institutions identify students who may benefit most from additional academic assistance. It combines calibrated risk estimation, fairness-aware ranking, cohort analytics, and capacity-constrained selection to support more equitable allocation of limited educational resources.

Rather than labeling students or making final decisions about their academic futures, AURA provides an evidence-based decision-support tool that helps educators prioritize tutoring, mentoring, and remedial learning opportunities.

## Overview

Educational institutions often have limited resources for academic support. Selecting students using a conventional ranking model can overlook struggling learners in smaller demographic or school groups.

AURA addresses this challenge by combining predictive modeling with fairness-aware prioritization. Under the benchmark's 20% support-capacity constraint, the system ranks students according to estimated support needs while auditing group-level outcomes.

### Key Objectives

- Identify students who may need additional academic support.
- Prioritize limited support resources using a defined capacity constraint.
- Reduce disparities in support-needed recall across eligible groups.
- Provide transparent cohort-level performance and fairness metrics.
- Preserve student privacy and keep educators involved in final decisions.

## Key Features

### 1. Fair Student Prioritization
Uses a calibrated ensemble and fairness-aware ranking mechanism to prioritize students for potential academic support.

### 2. Capacity-Constrained Selection
Selects exactly `ceil(0.20 × N)` students by default, where `N` is the number of evaluated students.

### 3. Cohort Analytics
Visualizes academic performance, student trajectories, grade changes, and cohort-level patterns.

### 4. Fairness Audit Dashboard
Compares group-level recall, selection rates, calibration, and fairness gaps to help evaluate how support allocation affects different groups.

### 5. Interactive Student Analytics
The AURA interface includes cohort visualization, demographic comparison, a what-if analysis area, and student-oriented academic trajectory views.

### 6. Prediction and Ranking Exports
Generates CSV files containing support probabilities, priority rankings, and selected-student indicators.

### 7. Responsible AI
Separates support prioritization from punitive academic decisions and emphasizes educator oversight, privacy, and transparent limitations.

## How It Works

1. **Input:** Load permitted student academic, demographic, and school-context features.
2. **Feature Engineering:** Prepare model inputs and derive relevant academic indicators.
3. **Risk Estimation:** Estimate each student's likelihood of requiring additional academic support.
4. **Fairness Alignment:** Apply fairness-aware ranking to reduce disparities across eligible groups.
5. **Capacity-Constrained Selection:** Select the highest-priority students within the configured support budget.
6. **Evaluation:** Measure recall, group fairness, calibration, and reproducibility.
7. **Output:** Export the ranked students and support-selection results for educator review.

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, LightGBM
- **Scientific Computing:** SciPy
- **Model Persistence:** Joblib
- **Testing:** Pytest
- **Outputs:** CSV reports and fairness tables
- **Interface:** Interactive AURA analytics dashboard

## Project Structure

```text
AURA/
├── data/
│   └── student-mat.csv
├── models/
│   └── fair_student_model.pkl
├── outputs/
│   ├── probability_and_ranking.csv
│   ├── selected_students.csv
│   └── fairness_table.md
├── src/
│   ├── model.py
│   └── metrics.py
├── tests/
├── train.py
├── inference.py
├── requirements.txt
├── README.md
└── RESPONSIBLE_USE_AND_LIMITATIONS.md
```

*This structure describes the expected project layout; include only files and folders that are actually present in your repository.*

## Installation

**Prerequisites:** Python 3.10 or a compatible version supported by the installed dependencies.

1. Clone the repository:

```bash
git clone https://github.com/kmahala140-sudo/aura--fair-student-prioritization.git
cd aura--fair-student-prioritization
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train the prioritization model using the development dataset:

```bash
python train.py \
  --data data/student-mat.csv \
  --model_out models/fair_student_model.pkl \
  --output_dir outputs/
```

Run training before inference if the trained model artifact is not already available.

## Inference

Evaluate a held-out dataset and generate the ranking and selected-student outputs:

```bash
python inference.py \
  --input data/eval_test_sample.csv \
  --output_probabilities outputs/probability_and_ranking.csv \
  --output_selected outputs/selected_students.csv \
  --model models/fair_student_model.pkl
```

The inference script supports configurable input paths, output paths, model paths, ID-column selection, and support-budget fractions.

### Output Files

| File | Description |
|---|---|
| `probability_and_ranking.csv` | Student ID, estimated support probability, and priority rank |
| `selected_students.csv` | Ranked students and their support-selection status |
| `fairness_table.md` | Group-wise fairness and performance metrics |

Use only authorized datasets and protect all student-identifying information.

## Run Tests

Execute the automated test suite:

```bash
python -m pytest tests/
```

## Evaluation Metrics

AURA evaluates its performance using the benchmark's scoring criteria:

- **Recall at 20% capacity:** How many students identified as needing support are captured within the selection budget.
- **Worst-group recall:** The lowest recall among eligible groups.
- **Fairness gap:** The disparity in recall between groups under the benchmark definition.
- **Calibration:** The accuracy of predicted probabilities, assessed using the Brier score.
- **Reproducibility:** Whether the results can be reproduced under the prescribed execution conditions.

### Reported Development Results

The project's development report records the following results:

| Metric | Reported result |
|---|---:|
| Overall recall at 20% capacity | 60.77% |
| Fairness gap | 4.97% |
| Brier score | 0.0374 |
| Calibration utility (1 − Brier) | 0.9626 |
| Development score including reproducibility bonus | 77.61 / 100 |

These are reported development-benchmark results, not a guarantee of performance on new datasets. Independent evaluation is required before deployment.

## Responsible AI and Limitations

AURA is intended to support students—not stigmatize them.

- Predictions must not be used to deny opportunities, assign punitive labels, or make disciplinary decisions.
- Educators and counselors must retain authority over final support decisions.
- Student records must be protected and shared only with authorized personnel.
- Group fairness metrics should be reviewed alongside sample sizes and uncertainty.
- Historical grading bias, missing data, proxy variables, and changes in student circumstances can affect predictions.
- Model performance and fairness should be reassessed regularly.
- Students should have appropriate opportunities for human review and correction.

The system's rankings are advisory. They do not replace professional judgment or direct conversations with students.

See `RESPONSIBLE_USE_AND_LIMITATIONS.md` for further discussion of intended use, model limitations, and governance.

## Future Improvements

- More detailed explanations of individual predictions.
- Confidence intervals for subgroup fairness metrics.
- Expanded monitoring across grading periods.
- Stronger privacy controls and role-based access.
- Improved educator review and feedback workflows.
- Validation on additional datasets and educational settings.

## Project Purpose

AURA explores how responsible machine learning can help institutions allocate limited educational support more equitably. Its goal is not simply to predict academic outcomes, but to help educators make better-informed decisions about where additional learning resources may be most beneficial.

**AURA — Equitable Student Prioritization. Support, not stigma.**
