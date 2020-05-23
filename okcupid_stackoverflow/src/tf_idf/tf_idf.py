from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


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
