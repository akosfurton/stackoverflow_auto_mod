import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from micdrop.utils.constants import NUMERIC_COLS, CATEGORICAL_COLS, Y_VAR
from micdrop.utils.evaluate_model import run_evaluate_model


def _create_train_validation_set(df):
    # No need to shuffle (0s and 1s appear randomly distributed in dataset, not all 0s then all 1s)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_x = train_df.drop(Y_VAR, axis=1)
    train_y = train_df[Y_VAR]

    val_x = val_df.drop(Y_VAR, axis=1)
    val_y = val_df[Y_VAR]

    return train_x, train_y, val_x, val_y


def fit_rf_model(x_train, y_train, save_external=False, base_folder=None, run_id=None):

    # Use balanced weights to counter the imbalanced data set
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(x_train, y_train.values.ravel())

    if save_external:
        save_path = f"{base_folder}/models/random_forest/{run_id}_random_forest.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(clf, save_path)

    return clf


def evaluate_model(clf, x_val, y_val):
    df = pd.DataFrame()
    df["label"] = y_val.values.ravel()
    df["pred_prob"] = clf.predict_proba(x_val)[:, 1]
    df["pred"] = np.where(df["pred_prob"] >= 0.5, 1, 0)

    run_evaluate_model(df)


def run_fit_evaluate_model(base_folder, run_id, save_external=True):
    df = pd.read_parquet(f"{base_folder}/data/processed/cleaned.parquet")

    data_dummy = pd.get_dummies(df[[Y_VAR] + CATEGORICAL_COLS + NUMERIC_COLS], dummy_na=True)

    if save_external:
        categorical_cols_dict = {x: list(df[x].unique()) for x in df[CATEGORICAL_COLS].columns}
        save_path = f"{base_folder}/models/random_forest/{run_id}_categorical_cols_dict.json"
        with open(save_path, "w") as fp:
            json.dump(categorical_cols_dict, fp)

    x_train, y_train, val_x, val_y = _create_train_validation_set(data_dummy)

    clf = fit_rf_model(
        x_train,
        y_train,
        save_external=save_external,
        base_folder=base_folder,
        run_id=run_id,
    )

    evaluate_model(clf, val_x, val_y)

    return clf, x_train, y_train
