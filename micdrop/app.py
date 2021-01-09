import os

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from joblib import load
from scipy.sparse import hstack

from micdrop.src.preprocessing.run_preprocessing import run_preprocessing
from micdrop.utils.constants import NOT_METADATA_COLS
from micdrop.utils.git_utils import get_git_root

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    # TODO: Change this to a JSON of a row in the input dataframe
    post = {
        "title": request.form["post_title"],
        "body": request.form["post_body"],
    }
    data = run_preprocessing("", df=pd.DataFrame([post]), save_external=False)
    x_pred = vectorizer.transform(data["cleaned_body"])

    metadata_cols = [x for x in data.columns if x not in NOT_METADATA_COLS]
    x_pred = hstack((x_pred, np.array(data[metadata_cols])))
    my_prediction = clf.predict_proba(x_pred)[:, 1][0]

    return render_template(
        "result.html",
        prediction=my_prediction,
        post_title=request.form["post_title"],
        post_body=request.form["post_body"],
    )


if __name__ == "__main__":

    root_dir = get_git_root(os.getcwd())
    path = f"{root_dir}/models/tf_idf"
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    latest_model_path = max(
        [x for x in paths if "_logistic" in x], key=os.path.getctime,
    )
    with open(latest_model_path, "rb") as saved_classifier:
        clf = load(saved_classifier)

    app.run(debug=True)
