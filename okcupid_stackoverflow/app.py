import os
import glob
import pandas as pd
from flask import Flask, render_template, request
from joblib import load
from okcupid_stackoverflow.src.preprocessing.run_preprocessing import run_preprocessing
from okcupid_stackoverflow.utils.constants import NOT_METADATA_COLS
from okcupid_stackoverflow.utils.git_utils import get_git_root

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    root_dir = get_git_root(os.getcwd())
    latest_model_path = max(
        [x for x in glob.glob(f"{root_dir}/models/tf_idf") if "logistic" in x],
        key=os.path.getctime,
    )
    latest_vectorizer_path = max(
        [x for x in glob.glob(f"{root_dir}/models/tf_idf") if "vectorizer" in x],
        key=os.path.getctime,
    )

    with open(latest_vectorizer_path, "rb") as saved_vectorizer:
        vectorizer = load(saved_vectorizer)
    with open(latest_model_path, "rb") as saved_classifier:
        clf = load(saved_classifier)

    post = {
        "post_title": request.form["post_title"],
        "post_body": request.form["post_body"],
    }
    data = pd.DataFrame.from_dict(post)
    data = run_preprocessing("", df=data, save_external=False)
    x_pred = vectorizer.transform(data["cleaned_body"])

    metadata_cols = [x for x in data.columns if x not in NOT_METADATA_COLS]
    x_pred = pd.concat([x_pred, data[metadata_cols]], axis=1)
    my_prediction = clf.predict_proba(x_pred)[:, 1]

    return render_template("result.html", prediction=my_prediction)


if __name__ == "__main__":
    app.run(debug=True)
