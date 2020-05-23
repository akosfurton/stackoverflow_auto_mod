import re
import string
import unicodedata

import contractions
import en_core_web_sm
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from pandas.core.common import flatten
from textstat import textstat
import swifter


def load_raw_data(data_path):
    df = pd.read_csv(data_path, usecols=["Title", "Body", "label"], encoding="utf-8")
    df.columns = [x.lower() for x in df.columns]
    return df


def load_stopwords(language="english"):
    stopword_list = stopwords.words(language)
    stopword_list.remove("no")
    stopword_list.remove("not")
    stopword_list.append("hi")
    stopword_list.append("please")

    return stopword_list


def strip_html_tags(text):
    tag_regex = re.compile(r"<[^>]+>")
    text = tag_regex.sub("", text)

    html_escape_table = {
        "&amp;": "and",
        "&quot;": '"',
        "&apos;": "'",
        "&gt;": ">",
        "&lt;": "<",
    }

    for html_esc in html_escape_table:
        text = text.replace(html_esc, html_escape_table[html_esc])

    return text


def strip_urls(text):
    # from geeksforgeeks.org/python-check-url-string
    url_regex = re.compile(
        r"(?i)\b((?:http?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)"
        r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()"
        r"<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    )
    text = url_regex.sub("replacedurl", text)

    return text


def convert_accented_characters(text):
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return text


def expand_contractions(text):
    return contractions.fix(text).lower()


def remove_punctuation(text):
    # Remove punctuation
    text = re.sub("[^A-Za-z0-9 ]+", "", text)
    text = " ".join(text.split())

    return text


def lemmatize_text(text, nlp):

    text = nlp(text)
    text = " ".join(
        [word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in text]
    )
    return text


def remove_stopwords(text):
    filtered_tokens = [token for token in text.split() if token not in load_stopwords()]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text


def remove_multiple_spaces(text):
    text = text.replace(r"\n", "")
    text = re.sub(r"\s+", " ", text)

    return text


def calc_num_sentences(text):
    return textstat.sentence_count(text)


def calc_num_words(text):
    return len(text.split())


def calc_num_chars(text):
    return textstat.char_count(text, ignore_spaces=True)


def calc_num_code_blocks(text):
    return text.count("<code>")


def calc_num_punctuation_chars(text):
    return len([x for x in text if x in set(string.punctuation)])


def calc_num_words_in_code_blocks(text):
    code_text = list(flatten(re.findall(r"<code>(.*?)</code>", text)))

    n_words = 0
    for snippet in code_text:
        n_words += len(snippet.split())

    return n_words


def calc_word_len_mean(text):
    words = text.split()
    word_lengths = np.array([len(x) for x in words])

    return np.mean(word_lengths)


def calc_word_len_median(text):
    words = text.split()
    word_lengths = np.array([len(x) for x in words])

    return np.median(word_lengths)


def calc_word_len_max(text):
    words = text.split()
    word_lengths = np.array([len(x) for x in words])

    return np.max(word_lengths)


def normalize_text(doc, deep_clean=False, nlp=None):
    # FOR BERT, Don't need to remove punctuation, don't need to remove stop-words
    if not deep_clean:
        doc = doc.lower()

        doc = strip_html_tags(doc)
        doc = strip_urls(doc)
        doc = convert_accented_characters(doc)
        doc = expand_contractions(doc)
        doc = remove_multiple_spaces(doc)

        # Tried to use the normalise library, but too slow
        # Would use it for converting to/from numbers, percents, abbreviations

        # Depending on text corpus, would want to process emojis as well
        # However, SO is not very emoji friendly

    else:
        assert doc == doc.lower(), "text has not been cleaned yet."
        doc = remove_punctuation(doc)
        doc = lemmatize_text(doc, nlp=nlp)
        doc = remove_stopwords(doc)

    return doc


def run_preprocessing():
    df = load_raw_data("../data/raw/interview_dataset.csv")

    # The removal of HTML tags will also remove the code block delimiter
    df["num_code_blocks"] = df["body"].apply(calc_num_code_blocks)
    df["num_words_code_blocks"] = df["body"].apply(calc_num_words_in_code_blocks)

    df["light_cleaned_title"] = (
        df["title"].swifter.allow_dask_on_strings().apply(normalize_text)
    )
    df["light_cleaned_body"] = (
        df["body"].swifter.allow_dask_on_strings().apply(normalize_text)
    )

    df["light_cleaned_text"] = (
        df["light_cleaned_title"] + " " + df["light_cleaned_body"]
    )

    # Calculate meta-data features
    df["num_sentences_body"] = df["light_cleaned_body"].apply(calc_num_sentences)
    df["num_words_title"] = df["light_cleaned_title"].apply(calc_num_words)
    df["num_words_body"] = df["light_cleaned_body"].apply(calc_num_words)
    df["num_chars_title"] = df["light_cleaned_title"].apply(calc_num_chars)
    df["num_chars_body"] = df["light_cleaned_body"].apply(calc_num_chars)
    df["num_punctuation"] = df["light_cleaned_body"].apply(calc_num_punctuation_chars)
    df["word_len_mean"] = df["light_cleaned_body"].apply(calc_word_len_mean)
    df["word_len_median"] = df["light_cleaned_body"].apply(calc_word_len_median)
    df["word_len_max"] = df["light_cleaned_body"].apply(calc_word_len_max)

    nlp = en_core_web_sm.load()
    # Normalize Text
    df["cleaned_title"] = (
        df["light_cleaned_title"]
        .swifter.allow_dask_on_strings()
        .apply(normalize_text, deep_clean=True, nlp=nlp)
    )
    df["cleaned_body"] = (
        df["light_cleaned_body"]
        .swifter.allow_dask_on_strings()
        .apply(normalize_text, deep_clean=True, nlp=nlp)
    )

    df["cleaned_text"] = df["cleaned_title"] + " " + df["cleaned_body"]
    df["num_words_body_cleaned"] = df["cleaned_body"].apply(calc_num_words)
    df["pct_words_meaning"] = df["num_words_body_cleaned"] / df["num_words_body"]

    df.to_parquet("../data/processed/cleaned.parquet")
