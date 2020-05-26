import os

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from joblib import load
from scipy.sparse import hstack

from okcupid_stackoverflow.src.preprocessing.run_preprocessing import run_preprocessing
from okcupid_stackoverflow.utils.constants import NOT_METADATA_COLS
from okcupid_stackoverflow.utils.git_utils import get_git_root

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    post = {
        "title": request.form["post_title"],
        "body": request.form["post_body"],
    }
    data = run_preprocessing("", df=pd.DataFrame([post]), save_external=False)
    x_pred = vectorizer.transform(data["cleaned_body"])

    metadata_cols = [x for x in data.columns if x not in NOT_METADATA_COLS]
    x_pred = hstack((x_pred, np.array(data[metadata_cols])))
    my_prediction = clf.predict_proba(x_pred)[:, 1][0]

    return render_template("result.html", prediction=my_prediction)


if __name__ == "__main__":


    root_dir = get_git_root(os.getcwd())
    path = f"{root_dir}/models/tf_idf"
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    latest_model_path = max(
        [x for x in paths if "_logistic" in x],
        key=os.path.getctime,
    )
    with open(latest_model_path, "rb") as saved_classifier:
        clf = load(saved_classifier)

    latest_vectorizer_path = max(
        [x for x in paths if "_vectorizer" in x],
        key=os.path.getctime,
    )
    with open(latest_vectorizer_path, "rb") as saved_vectorizer:
        vectorizer = load(saved_vectorizer)

    app.run(debug=True)