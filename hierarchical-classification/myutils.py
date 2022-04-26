import os
import time

import joblib
import hiclass
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode
import networkx as nx
from nltk import regexp_tokenize
import numpy as np
import pymorphy2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


WORKING_PATH = './working/'
TARGET = 'category_id'  # Target feature name
TRAIN = 'TRAIN'  # Binary feature to separate train and test data
PATH_COLS = ['node_1', 'node_2', 'node_3', 'node_4', 'node_5']


class Tokenizer:
    """Interface of lemma tokenizer with sklearn"""

    def __init__(self, stop_words, regexp):
        self.stop_words = set(stop_words)
        self.regexp = regexp
        self.morhp = pymorphy2.MorphAnalyzer()

    def __call__(self, doc):
        # Lemmatize
        tokens = [self.morhp.parse(t)[0].normal_form
                  for t in regexp_tokenize(doc, self.regexp)]

        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]

        return tokens


def get_vectorizer(x, fname, regexp, saving=True):
    """Returns vectorizer from file if it exists.
    Otherwise, this function initializes and fits vectorizer.
    """
    fname = WORKING_PATH + fname

    # If it already exists
    if os.path.isfile(fname):
        # Load vectorizer
        vectorizer = joblib.load(fname)
    else:
        # Read stop words
        with open(WORKING_PATH + 'stopwords-ru.txt') as f:
            stop_words = f.read().splitlines()

        # Initialize tokenizer
        tokenizer = Tokenizer(stop_words=stop_words, regexp=regexp)

        # Initialize vectorizer
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, min_df=2)

        # Fit vectorizer
        vectorizer.fit(x)

        if saving:
            # Save vectorizer
            joblib.dump(vectorizer, fname)

    return vectorizer


def get_title_vectors(x, fname, vectorizer, saving=True):
    """Returns vectorized 'title' column"""

    fname = WORKING_PATH + fname

    # If it already exists
    if os.path.isfile(fname):
        # Load 'title' vectors
        title_vect = joblib.load(fname)
    else:
        # Vectorize 'title' column
        title_vect = vectorizer.transform(x)

        if saving:
            # Save 'title' vectors
            joblib.dump(title_vect, fname)

    return title_vect


def get_prediction(X, fname, classifier, saving=True):
    """Returns prediction"""

    fname = WORKING_PATH + fname

    # If it already exists
    if os.path.isfile(fname):
        # Load prediction
        pred = joblib.load(fname)
    else:
        # Predict
        pred = classifier.predict(X).astype(int)

        if saving:
            # Save prediction
            joblib.dump(pred, fname)

    return pred


def get_classifier(X, y, fname, model_type, n_estimators, random_state,
                   saving=True):
    """Returns LCPN classifier from file if it exists.
    Otherwise, this function initializes and fits classifier.
    """
    fname = WORKING_PATH + fname

    # If it already exists
    if os.path.isfile(fname):
        # Load classifier
        clf = joblib.load(fname)
    else:
        # Init classifier
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features='sqrt',
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=random_state,
        )

        if model_type == 'lcn':
            clf = LocalClassifierPerNode(local_classifier=rf, verbose=0)
        elif model_type == 'lcpn':
            clf = LocalClassifierPerParentNode(local_classifier=rf, verbose=0)
        else:
            raise ValueError

        # Train classifier
        clf.fit(X, y)

        if saving:
            # Save classifier
            joblib.dump(clf, fname)

    return clf


def split_scale_df(df, validation_size, random_state, stratify=False):
    """Splits and scale DataFrame"""

    # Split data: Х - features, у - target variable
    X = df.loc[df[TRAIN]].drop([TRAIN, TARGET] + PATH_COLS, axis=1)
    X_test = df.loc[~df[TRAIN]].drop([TRAIN, TARGET] + PATH_COLS, axis=1)
    y = df.loc[df[TRAIN], PATH_COLS].astype(int).values
    indices = df.loc[df[TRAIN]].index

    # Split data on train and validation parts
    strat = df.loc[df[TRAIN], TARGET] if stratify else None
    (X_train, X_valid, y_train, y_valid,
     indices_train, indices_valid) = train_test_split(
        X, y, indices, test_size=validation_size, stratify=strat,
        shuffle=True, random_state=random_state
    )

    return (X, X_train, X_valid, X_test, y, y_train, y_valid, indices_train,
            indices_valid)


def get_path_map(graph, root):
    """Returns a graph path map for all leaves"""

    leaves = [node for node in graph.nodes if graph.out_degree(node) == 0]
    depth = hiclass.data.find_max_depth(graph, root=root) - 1
    placeholder = hiclass.data.PLACEHOLDER_LABEL_NUMERIC

    path_map = {}
    for leaf in leaves:
        for p in nx.all_simple_paths(graph, source=root, target=leaf):
            path = np.full(depth, placeholder, 'int')
            path[:len(p) - 1] = p[1:]
            path_map[leaf] = path
    return path_map


def get_path_map_by_depth(graph, root):
    """Returns a graph path map for all leaves and for each graph depth"""

    path_maps = []
    depth = hiclass.data.find_max_depth(graph, root=root) - 1
    for i in range(depth):
        G_depth = nx.dfs_tree(graph, source=root, depth_limit=i + 1)
        path_maps.append(get_path_map(G_depth, root))
    return path_maps


def get_score(path_map, y_true, y_pred):
    """Returns weighted hF score and dictionary
    with other scores for each leaf
    """
    assert y_true.shape == y_pred.shape

    whF, results = 0, {}
    for leaf, path in tqdm(path_map.items()):
        # Form 2D array of repeated leaf path
        path_2D = np.repeat([path], len(y_true), axis=0)

        # Compute masks of P, T and their intersection PT
        P_mask = np.logical_and(path_2D == y_pred, path_2D != -1)
        T_mask = np.logical_and(path_2D == y_true, path_2D != -1)
        PT_mask = np.logical_and(P_mask, T_mask)

        # Compute P, T and PT
        P = P_mask.sum()
        T = T_mask.sum()
        PT = PT_mask.sum()

        # Compute hP, hR and hF
        hP = PT / P
        hR = PT / T
        hF = (2 * hP * hR) / (hP + hR)

        # Accumulate weighted hF
        leaf_count = (y_true == leaf).sum()
        whF += leaf_count * hF / len(y_true)

        # Store results for the leaf
        results[leaf] = {}
        results[leaf]['P'] = P
        results[leaf]['T'] = T
        results[leaf]['PT'] = PT
        results[leaf]['hP'] = hP
        results[leaf]['hR'] = hR
        results[leaf]['hF'] = hF
        results[leaf]['count'] = leaf_count

    time.sleep(0.1)  # For clear printing
    return whF, results


def get_score_by_depth(path_maps, y_true, y_pred):
    """Returns weighted hF score and dictionary
    with other scores for each leaf and for each graph depth
    """
    whF_depth, results_depth = [], []
    for i in range(len(path_maps)):
        whF, results = get_score(path_maps[i], y_true[:, :i + 1],
                                 y_pred[:, :i + 1])
        whF_depth.append(whF)
        results_depth.append(results)
    return whF_depth, results_depth


def get_leaves(y):
    """Returns leaves form array of predicted paths"""

    leaf_indices = (y != -1).sum(axis=1) - 1
    return y[np.arange(y.shape[0]), leaf_indices]
