import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from okcupid_stackoverflow.utils.evaluate_model.evaluate_model import run_evaluate_model


def _create_train_validation_set(df):
    # No need to shuffle (0s and 1s appear randomly distributed in dataset, not all 0s then all 1s)
    # No need to stratify (both classes are ~50%, should be approx maintained with random split)
    # Stratify is most useful with infrequent labels which may disappear from either train or test
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_x = train_df["cleaned_body"]
    train_y = train_df["label"]

    val_x = val_df["cleaned_body"]
    val_y = val_df["label"]

    return train_x, train_y, val_x, val_y


def create_vocab_word_count(df, col_nm, max_features=None):

    docs = df[col_nm].tolist()

    # Stop words have already been removed as part of preprocessing
    cv = CountVectorizer(max_df=0.85, stop_words="english")
    if max_features is not None:
        cv.max_features = max_features

    word_count_vector = cv.fit_transform(docs)

    return cv, word_count_vector


def fit_idf(word_count_vector):
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

    tfidf_transformer.fit(word_count_vector)

    return tfidf_transformer


def fit_tf_idf_vector(df):

    train_x, train_y, val_x, val_y = _create_train_validation_set(df)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 4),
        max_features=10000,
    )

    vectorizer.fit(train_x)
    x_train = vectorizer.transform(train_x)
    x_val = vectorizer.transform(val_x)

    return x_train, train_y, x_val, val_y


def fit_logistic_reg_on_tf_idf(x_train, y_train, save_external=False):

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(x_train, y_train)

    if save_external:
        joblib.dump(clf, "tf_idf_logistic.pkl")

    return clf


def evaluate_logistic_reg_on_tf_idf(clf, x_val, y_val):
    df = pd.DataFrame()
    df["label"] = y_val
    df["pred_prob"] = clf.predict_proba(x_val)[:, 1]
    df["pred"] = np.where(df["pred_prob"] >= 0.5, 1, 0)

    run_evaluate_model(df)


def predict_idf(doc, tfidf_transformer, cv):
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

    def _sort_coo(coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    sorted_items = _sort_coo(tf_idf_vector.tocoo())

    keywords = _extract_top_n_from_vector(cv.get_feature_names(), sorted_items, 10)

    return keywords


def _extract_top_n_from_vector(feature_names, sorted_items, top_n=10):

    sorted_items = sorted_items[:top_n]
    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results
