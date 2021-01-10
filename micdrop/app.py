import json
import os

import pandas as pd
from pandas.api.types import CategoricalDtype

from flask import Flask, render_template, request, make_response
from joblib import load
from werkzeug.utils import secure_filename

from micdrop.src.preprocessing.run_preprocessing import run_preprocessing
from micdrop.utils.constants import CATEGORICAL_COLS, NUMERIC_COLS, EXPECTED_COLUMNS, Y_VAR
from micdrop.utils.git_utils import get_git_root

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    file_obj = request.files['file']
    post = pd.read_csv(file_obj, encoding="utf-8", parse_dates=["click_date"])
    df = post.copy()
    df.columns = [x.lower() for x in df.columns]
    if Y_VAR in df.columns:
        df = df.drop(Y_VAR, axis=1)

    input_cols = sorted(list(df.columns))
    if input_cols != sorted(list(EXPECTED_COLUMNS)):
        raise AssertionError(
            f"List of input columns does not match expected list."
            f"Input columns are: {input_cols}"
            f"Expected columns are: {sorted(list(EXPECTED_COLUMNS))}"
        )

    df = run_preprocessing("", df=df, save_external=False)
    # Convert into categorical data type before pd.get_dummies()
    # This way, categories that are missing in the prediction dataset are still kept as columns of all 0s.
    for col_nm in CATEGORICAL_COLS:
        df[col_nm] = df[col_nm].astype(CategoricalDtype(categories=latest_categorical_cols_dict[col_nm]))

    x_pred = pd.get_dummies(df[CATEGORICAL_COLS + NUMERIC_COLS], dummy_na=True)
    post["predicted_likelihood"] = clf.predict_proba(x_pred)[:, 1]

    response = make_response(post.to_csv(index=False))
    out_name = secure_filename(file_obj.filename).rsplit('.', 1)[0]
    response.headers["Content-Disposition"] = f"attachment; filename={out_name}_predictions.csv"
    return response


if __name__ == "__main__":

    root_dir = get_git_root(os.getcwd())
    path = f"{root_dir}/models/random_forest"
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    latest_model_path = max(
        [x for x in paths if "random_forest.pkl" in x], key=os.path.getctime,
    )
    latest_cat_cols_dict_path = max(
        [x for x in paths if "categorical_cols_dict.json" in x], key=os.path.getctime,
    )
    with open(latest_model_path, "rb") as saved_classifier:
        clf = load(saved_classifier)

    with open(latest_cat_cols_dict_path, "rb") as c:
        latest_categorical_cols_dict = json.load(c)

    app.run(debug=True)
