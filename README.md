# Cardiovascular disease prediction

A Python classifier using the `cardio_train.csv` dataset referenced by
[Vitalii Mokin's Kaggle notebook](https://www.kaggle.com/code/vbmokin/20-models-for-cardiovascular-disease-prediction).
This is an independent implementation using scikit-learn's histogram gradient boosting,
not a reproduction of the notebook's 20-model comparison.

For education and experimentation. It classifies the recorded `cardio` label;
it does not estimate a validated future disease risk or provide a diagnosis.

## Run in this workspace (Linux / Codespaces)

The environment, dataset, and trained model have already been created locally.

```bash
source .venv/bin/activate
python cardio.py train
python cardio.py predict --input examples/patients.csv
```

Training prints evaluation metrics and saves `artifacts/cardio_model.joblib`
and `artifacts/metrics.json`. Prediction prints results and saves
`artifacts/predictions.csv`. A probability of at least 0.5 produces class 1.

## Set up on another computer

Use Python 3.11 or newer, from this project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
`requirements-lock.txt` records the exact environment used here (Python 3.14);
`requirements.txt` allows compatible versions for other Python versions.

Download and unzip the [original Kaggle dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset),
then place `cardio_train.csv` inside `data/`. Kaggle may require sign-in.
The local run used this [public dataset mirror](https://github.com/garrysjh/cardioML),
which attributes the CSV to the same Kaggle dataset, because the Kaggle download endpoint failed.
To obtain that same mirror:

```bash
python -c "from pathlib import Path; from urllib.request import urlretrieve; Path('data').mkdir(exist_ok=True); urlretrieve('https://raw.githubusercontent.com/garrysjh/cardioML/main/cardio_train.csv', 'data/cardio_train.csv')"
python cardio.py train --data data/cardio_train.csv
python cardio.py predict --input examples/patients.csv
```

Data and generated artifacts are excluded from Git.

## Predict on your own rows

Create a CSV with the following columns (comma or semicolon separated).
See `examples/patients.csv` for two illustrative inputs.

| Column | Meaning / units |
| --- | --- |
| age | Age **in days**, approximately years × 365.25 |
| gender | Dataset code: 1 = women, 2 = men |
| height | Centimeters |
| weight | Kilograms |
| ap_hi | Systolic blood pressure, mmHg |
| ap_lo | Diastolic blood pressure, mmHg |
| cholesterol | 1 = normal, 2 = above normal, 3 = well above normal |
| gluc | Glucose: same codes as cholesterol |
| smoke | 0 = no, 1 = yes |
| alco | Alcohol intake: 0 = no, 1 = yes |
| active | Physically active: 0 = no, 1 = yes |

The training CSV also needs `cardio` (0 = disease absent, 1 = disease present).
`id` is ignored. Prediction input does not need `cardio` or `id`.
Blank feature values are filled with medians learned from training data.
Each input column must have at least one nonblank value.

```bash
python cardio.py predict --input my_patients.csv --output artifacts/my_predictions.csv
```

Only load trusted model files: joblib model loading can execute code.

## Evaluation and data handling

- Removes identical feature/label rows before splitting, ignoring `id`.
- Converts nonpositive continuous values to missing. Converts pressures to missing
  when systolic is above 300, diastolic above 200, or systolic below diastolic.
  These are documented data quality rules, not clinical reference ranges.
- Uses a fixed, stratified 80/20 training/test split (seed 42).
- Fits median imputation and the classifier using only training rows.
- Reports accuracy, ROC AUC, precision, recall, F1, and a confusion matrix
  with rows = actual class and columns = predicted class, in order [0, 1].
- Saves the evaluated model trained on the 80% training partition.

This is a single held-out evaluation from one dataset. It has no external clinical
validation, probability calibration assessment, or subgroup evaluation. A score of
0.7 is a model output, not a verified 70% clinical disease risk.

Local run on the mirror: 55,980 training rows and 13,996 test rows after
removing duplicate feature/label rows. Held-out accuracy: **73.17%**;
ROC AUC: **0.8000**; disease-class recall: **68.89%**. Results may vary with
library versions or a different copy of the dataset.
