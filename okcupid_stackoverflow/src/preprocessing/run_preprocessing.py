import re
import string
import unicodedata

import pandas as pd
from nltk.corpus import stopwords
from pandas.core.common import flatten
from textstat import textstat

from okcupid_stackoverflow.utils.contractions import CONTRACTION_MAP


def load_raw_data(data_path):
    return pd.read_csv(data_path, usecols=["Title", "Body", "label"], encoding="utf-8")


def load_stopwords(language="english"):
    stopword_list = stopwords.words(language)
    stopword_list.remove("no")
    stopword_list.remove("not")

    return stopword_list


def strip_html_tags(text):
    tag_regex = re.compile(r"<[^>]+>")
    return tag_regex.sub("", text)


def convert_accented_characters(text):
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return text


def expand_contractions(text, contraction_mapping):
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_special_characters(text, remove_digits):
    # Remove punctuation, optionally remove digits
    if remove_digits:
        text = re.sub(r"(\d|\W)+", "", text)
    else:
        text = re.sub(r"[^a-zA-z0-9\s]", "", text)
    return text


def lemmatize_text(text):
    text = nlp(text)
    text = " ".join(
        [word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in text]
    )
    return text


def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in load_stopwords()]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text


def remove_multiple_spaces(text):
    # TODO: Also remove new lines
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


def normalize_text(doc, deep_clean=False, remove_digits=False):
    # FOR BERT, Don't need to remove punctuation, don't need to remove stop-words
    doc = doc.lower()

    doc = strip_html_tags(doc)
    doc = convert_accented_characters(doc)
    doc = expand_contractions(doc, CONTRACTION_MAP)
    doc = remove_multiple_spaces(doc)

    if deep_clean:
        doc = remove_special_characters(doc, remove_digits=remove_digits)
        doc = lemmatize_text(doc)
        doc = remove_stopwords(doc)

    return doc


def run_preprocessing():
    df = load_raw_data("data/raw/interview_dataset.csv")

    df["light_cleaned_title"] = df["title"].apply(normalize_text)
    df["light_cleaned_body"] = df["body"].apply(normalize_text)

    # Calculate pre-normalized features
    df["num_sentences_body"] = df["body"].apply(calc_num_sentences)
    df["num_words_title"] = df["title"].apply(calc_num_words)
    df["num_words_body"] = df["body"].apply(calc_num_words)
    df["num_chars_title"] = df["title"].apply(calc_num_chars)
    df["num_chars_body"] = df["body"].apply(calc_num_chars)
    df["num_code_blocks"] = df["body"].apply(calc_num_code_blocks)
    df["num_words_code_blocks"] = df["body"].apply(calc_num_words_in_code_blocks)
    df["num_punctuation"] = df["body"].apply(calc_num_punctuation_chars)

    # Normalize Text
    df["cleaned_title"] = df["title"].apply(normalize_text, deep_clean=True)
    df["cleaned_body"] = df["body"].apply(normalize_text, deep_clean=True)

    # Calculate TF_IDF features

    df.to_parquet("data/processed/cleaned.parquet")

