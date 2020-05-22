import pandas as pd
from okcupid_stackoverflow.utils.contractions import CONTRACTION_MAP
from nltk.corpus import stopwords
import re


def load_raw_data(data_path):
    return pd.read_csv(data_path, usecols=["Title", "Body", "label"], encoding="utf-8")


def load_stopwords(language="english"):
    stopword_list = stopwords.words(language)
    stopword_list.remove("no")
    stopword_list.remove("not")

    return stopword_list


def strip_html_tags(text):
    TAG_REGEX = re.compile(r'<[^>]+>')
    return TAG_REGEX.sub('', text)


def remove_accented_chars(text):
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
    if remove_digits:
        text = re.sub("(\\d|\\W)+", "", text)
    else:
        text = re.sub("[^a-zA-z0-9\s]", "", text)
    return text


def lemmatize_text(text):
    text = nlp(text)
    text = " ".join(
        [word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in text]
    )
    return text


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = " ".join([ps.stem(word) for word in text.split()])
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopword_list
        ]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text

def remove_multiple_spaces(text):
    text = re.sub(r'\s+', ' ', text)

    return text

def normalize_text(doc):
    # FOR BERT, Don't need to remove punctuation, don't need to remove stop-words
    doc = doc.lower()

    doc = strip_html_tags(doc)
    doc = remove_accented_chars(doc)
    doc = expand_contractions(doc, CONTRACTION_MAP)
    doc = re.sub(r"[\r|\n|\r\n]+", " ", doc)
    doc = lemmatize_text(doc)
    special_char_pattern = re.compile(r"([{.(-)!}])")
    doc = special_char_pattern.sub(" \\1 ", doc)
    doc = remove_special_characters(doc, remove_digits=remove_digits)

    doc = remove_multiple_spaces(doc)

    doc = remove_stopwords(doc, is_lower_case=text_lower_case)

    return doc


def run_preprocessing():
    # Calculate pre-normalized features

    # Normalize Text
    df['cleaned_title'] = df['title'].apply(normalize_text)
    df['cleaned_body'] = df['body'].apply(normalize_text)

    # Calculate TF_IDF features

    return
