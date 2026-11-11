"""Train and run a cardiovascular disease classifier. Age is measured in days."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

FEATURES = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active']


def read_features(path):
    frame = pd.read_csv(path, sep=None, engine='python')
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    x = frame[FEATURES].apply(pd.to_numeric, errors='raise')
    if np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError('Features must not contain infinity.')
    # Blank values are imputed using training data only.
    for column, allowed in {'gender': [1, 2], 'cholesterol': [1, 2, 3],
                            'gluc': [1, 2, 3], 'smoke': [0, 1],
                            'alco': [0, 1], 'active': [0, 1]}.items():
        if not x[column].dropna().isin(allowed).all():
            raise ValueError(f'{column} must use codes {allowed}')
    for column in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
        if (x[column].dropna() <= 0).any():
            raise ValueError(f'{column} must be positive; age is in days.')
    if (x.ap_hi < x.ap_lo).any():
        raise ValueError('Systolic pressure ap_hi must be >= diastolic ap_lo.')
    if x.isna().all().any():
        raise ValueError('Each feature needs at least one non-missing value.')
    return frame, x


def train(args):
    # The original dataset contains impossible pressures. Treat invalid continuous
    # measurements as missing rather than discarding patients based on their label.
    raw = pd.read_csv(args.data, sep=None, engine='python')
    for col in ['age', 'height', 'weight', 'ap_hi', 'ap_lo']:
        if col in raw:
            raw[col] = pd.to_numeric(raw[col], errors='raise')
            raw.loc[raw[col] <= 0, col] = np.nan
    if {'ap_hi', 'ap_lo'} <= set(raw.columns):
        bad = (raw.ap_hi > 300) | (raw.ap_lo > 200) | (raw.ap_hi < raw.ap_lo)
        raw.loc[bad, ['ap_hi', 'ap_lo']] = np.nan
    missing = set(FEATURES + ['cardio']) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    raw = raw.drop_duplicates(subset=FEATURES + ['cardio'])
    x = raw[FEATURES].apply(pd.to_numeric, errors='raise')
    y = raw['cardio']
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError('cardio must contain both classes 0 and 1, without blanks.')
    if np.isinf(x.to_numpy(dtype=float)).any() or x.isna().all().any():
        raise ValueError('Features contain infinity or an entirely empty column.')
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline(SimpleImputer(strategy='median'),
                          HistGradientBoostingClassifier(max_iter=200,
                              learning_rate=0.08, max_leaf_nodes=15,
                              l2_regularization=1.0, random_state=42))
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        'training_rows': len(x_train), 'test_rows': len(x_test),
        'accuracy': accuracy_score(y_test, predictions),
        'roc_auc': roc_auc_score(y_test, probabilities),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'classification_report': classification_report(y_test, predictions, output_dict=True),
        'random_seed': 42, 'threshold': 0.5,
    }
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    print(f'Saved model to {args.model}')


def predict(args):
    frame, x = read_features(args.input)
    # Load only trusted local model files: joblib uses pickle internally.
    model = joblib.load(args.model)
    frame['cardio_probability'] = model.predict_proba(x)[:, 1]
    frame['cardio_prediction'] = (frame.cardio_probability >= 0.5).astype(int)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print(f'Saved predictions to {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--data', default='data/cardio_train.csv')
    training.add_argument('--model', default='artifacts/cardio_model.joblib')
    training.add_argument('--metrics', default='artifacts/metrics.json')
    training.set_defaults(run=train)
    prediction = commands.add_parser('predict')
    prediction.add_argument('--input', default='examples/patients.csv')
    prediction.add_argument('--model', default='artifacts/cardio_model.joblib')
    prediction.add_argument('--output', default='artifacts/predictions.csv')
    prediction.set_defaults(run=predict)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
