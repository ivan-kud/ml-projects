import os

import fasttext
from nltk import regexp_tokenize
import numpy as np
import pandas as pd
import pymorphy2
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


class Tokenizer:
    """Interface of lemma tokenizer with sklearn"""

    def __init__(self, stop_words, regexp):
        self.stop_words = set(stop_words)
        self.regexp = regexp
        self.morph = pymorphy2.MorphAnalyzer()

    def __call__(self, doc):
        # Lemmatize
        tokens = [self.morph.parse(t)[0].normal_form
                  for t in regexp_tokenize(doc, self.regexp)]

        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]

        # Add 'empty' if len is zero
        if len(tokens) == 0:
            tokens = ['empty']

        # Transform to string
        tokens = ' '.join(tokens)

        return tokens


def get_tokenized_x(x, fname, stopwords_fname, regexp, saving=True):
    """Returns tokenized x from file or compute it"""

    # If it already exists
    if os.path.isfile(fname):
        # Load tokenized x
        tokenized_x = pd.read_csv(fname, names=['description'])
    else:
        # Read stop words
        with open(stopwords_fname) as f:
            stop_words = f.read().splitlines()

        # Initialize tokenizer
        tokenizer = Tokenizer(stop_words=stop_words, regexp=regexp)

        # Tokenize x
        tokenized_x = x.apply(tokenizer)

        if saving:
            # Save tokenized x
            tokenized_x.to_csv(fname, index=False, header=False)

    return tokenized_x


def get_vectorizer(params, fname, saving=True):
    """Returns vectorizer from file or compute it"""

    # If it already exists
    if os.path.isfile(fname):
        # Load vectorizer
        vectorizer = fasttext.load_model(fname)
    else:
        # Fit vectorizer
        vectorizer = fasttext.train_unsupervised(**params)

        if saving:
            # Save vectorizer
            vectorizer.save_model(fname)

    return vectorizer


def get_vectorized_x(tokenized_x, fname, vectorizer, saving=True):
    """Returns vectorized x from file or compute it"""

    # If it already exists
    if os.path.isfile(fname):
        # Load vectorized data
        vectorized_x = np.load(fname)
    else:
        # Vectorize data
        vectorized_x = np.stack(
            tokenized_x.apply(vectorizer.get_sentence_vector).values
        )

        if saving:
            # Save vectorized data
            np.save(fname, vectorized_x)

    return vectorized_x


def get_classifier(params, fname, saving=True):
    """Returns classifier from file or train it"""

    # If it already exists
    if os.path.isfile(fname):
        # Load classifier
        classifier = fasttext.load_model(fname)
    else:
        # Fit classifier
        classifier = fasttext.train_supervised(**params)

        if saving:
            # Save classifier
            classifier.save_model(fname)

    return classifier


def get_score(y_true, y_score, category):
    """Returns ROC-AUC score"""

    assert y_true.shape == y_score.shape

    # Encode labels
    le = LabelEncoder()
    encoded_category = le.fit_transform(category)
    unique_category = np.unique(encoded_category)

    # Compute ROC AUC for all categories
    roc_auc = {}
    roc_auc_w = {}
    for categ in unique_category:
        mask = encoded_category == categ
        score = roc_auc_score(y_true[mask], y_score[mask])
        label = le.inverse_transform([categ])[0]
        roc_auc[label] = score
        roc_auc_w[label] = score * len(y_true[mask]) / len(y_true)

    # Compute macro and micro scores
    macro_score = sum(roc_auc.values()) / len(unique_category)
    micro_score = sum(roc_auc_w.values())

    return macro_score, micro_score, roc_auc
