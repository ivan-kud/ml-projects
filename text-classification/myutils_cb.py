from itertools import groupby
import os
import re

from nltk import regexp_tokenize
import numpy as np
import pandas as pd
import pymorphy2
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

units_en = [
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
units_ru = [
    "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь",
    "восемь", "девять", "десять", "одиннадцать", "двенадцать",
    "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
    "семнадцать", "восемнадцать", "девятнадцать",
]
tens_en = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty",
    "seventy", "eighty", "ninety",
]
tens_ru = [
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
    "семьдесят", "восемьдесят", "девяносто",
]
hundreds_ru = [
    "", "сто", "двести", "триста", "четыреста", "пятьсот",
    "шестьсот", "семьсот", "восемьсот", "девятьсот",
]
NUMWORDS = {}
for idx, word in enumerate(units_en):
    NUMWORDS[word] = str(idx)
for idx, word in enumerate(units_ru):
    NUMWORDS[word] = str(idx)
for idx, word in enumerate(tens_en):
    NUMWORDS[word] = str(idx * 10)
for idx, word in enumerate(tens_ru):
    NUMWORDS[word] = str(idx * 10)
for idx, word in enumerate(hundreds_ru):
    NUMWORDS[word] = str(idx * 100)
NUMWORDS['сот'] = '100'

PHONE_DETECT = 4

PHONE_RE_1 = re.compile(r'([78]?)([3489]\d{2})\d{7}')
PHONE_RE_2 = re.compile(r'([78][3489]\d{2})\d{7}')
PHONE_RE_3 = re.compile(r'([3489]\d{2})\d{7}')
NUMBER_RE = re.compile(r'(\d+)')
ENGLISH_RE = re.compile(r'[a-z]')


class Tokenizer:
    def __init__(self, stop_words: list, regexp: str, mode):
        self.stop_words = set(stop_words)
        self.regexp = regexp
        self.mode = mode
        self.morph = pymorphy2.MorphAnalyzer()
        self.non_lemmatize_words = {'тел', 'сорок'}

    def tokenize(self, doc: str):
        # Convert string to lowercase
        doc = doc.lower()

        # Tokenize by regular expression
        tokens = [t for t in regexp_tokenize(doc, self.regexp)]

        # Add 'продавать' if len is zero
        if len(tokens) == 0:
            tokens = ['продавать']

        # Split tokens onto digits and letters
        if self.mode[1]:
            tokens = split_digits(tokens, ru_only=False)
        else:
            tokens = split_digits(tokens, ru_only=True)

        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]

        # Concatenate 'восемь сот', 'девять сот' and so on into one token
        t = [tokens[0]]
        for i in range(1, len(tokens)):
            if tokens[i] == 'сот' and tokens[i-1] in ["пять", "шесть", "семь",
                                                      "восемь", "девять"]:
                t[-1] = tokens[i-1] + tokens[i]
            else:
                t.append(tokens[i])
        tokens = t

        # Lemmatize tokens with russian letters
        tokens = [self.morph.parse(t)[0].normal_form
                  if not ENGLISH_RE.search(t)
                  and len(t) > 2
                  and t not in self.non_lemmatize_words
                  else t for t in tokens]

        # Convert tokens of word numbers to tokens of digit numbers
        if self.mode[0]:
            tokens = convert_word_numbers(tokens)

        # Replace numbers to number of its digits
        if self.mode[1]:
            tokens = merge_numbers(tokens)
            tokens = [str(len(t)) if t.isdigit() else t for t in tokens]

        # Transform to string
        tokens = ' '.join(tokens)

        return tokens


def get_tokenized_x(x, fname, stopwords_fname, regexp, mode, saving=True):
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
        tokenizer = Tokenizer(stop_words=stop_words, regexp=regexp, mode=mode)

        # Tokenize x
        tokenized_x = x.apply(tokenizer.tokenize)

        if saving:
            # Save tokenized x
            tokenized_x.to_csv(fname, index=False, header=False)

    return tokenized_x


def get_phone_feature(x, fname, saving=True):
    """Returns phone feature"""

    # If it already exists
    if os.path.isfile(fname):
        # Load phone feature
        phone = pd.read_csv(fname, names=['phone'])
    else:
        # Calculate phone feature
        phone = x.apply(phone_detect)

        if saving:
            # Save phone feature
            phone.to_csv(fname, index=False, header=False)

    return phone


def split_digits(tokens, ru_only):
    """Splits tokens onto non digit tokens and digit tokens.

    Example: '945abc404пуля' splits onto '945', 'abc', 404', 'пуля'.
    If ru_only == True than it splits tokens that consist of only russian
    letters and digits.
    """
    t = []
    for token in tokens:
        if ru_only and ENGLISH_RE.search(token):
            t.append(token)
            continue
        lst = NUMBER_RE.split(token)
        if not lst[0]:
            lst = lst[1:]
        if not lst[-1]:
            lst = lst[:-1]
        t += lst

    return t
    # t = []
    # for token in tokens:
    #     number = re.sub(r'\D', '', token)
    #     if number:
    #         t.append(number)
    #     non_number = re.sub(r'\d', '', token)
    #     if non_number:
    #         t.append(non_number)


def convert_word_numbers(tokens):
    """Convert tokens of word numbers to tokens of digit numbers"""

    t = []
    last_was_numword = False
    for token in tokens:
        if token in NUMWORDS.keys():
            if last_was_numword:
                zeros = len(t[-1]) - len(t[-1].rstrip('0'))
                if (zeros >= len(NUMWORDS[token])
                        and t[-1] not in ['0', '10']
                        and NUMWORDS[token] not in ['0']):
                    t[-1] = str(int(t[-1]) + int(NUMWORDS[token]))
                else:
                    t.append(NUMWORDS[token])
            else:
                t.append(NUMWORDS[token])
            last_was_numword = True
        else:
            t.append(token)
            last_was_numword = False
    return t


def merge_neighbor_numbers(tokens: list):
    return [sub for k, g in groupby(tokens, str.isdigit)
            for sub in ([''.join(g)] if k else g)]


def merge_numbers(tokens):
    if len(tokens) < 2:
        return tokens
    tokens = tokens.copy()

    # Merge neighbor tokens and through one token with length up to 4
    num_through_token_max_len = 4
    t = []
    for i in range(len(tokens) - 2):
        if tokens[i].isdigit():
            if tokens[i + 1].isdigit():
                tokens[i + 1] = ''.join([tokens[i], tokens[i + 1]])
                continue
            elif tokens[i + 2].isdigit() \
                    and (len(tokens[i + 1]) <= num_through_token_max_len):
                tokens[i + 2] = ''.join([tokens[i], tokens[i + 2]])
                continue
        t.append(tokens[i])
    if tokens[-2].isdigit() and tokens[-1].isdigit():
        tokens[-1] = ''.join([tokens[-2], tokens[-1]])
    else:
        t.append(tokens[-2])
    t.append(tokens[-1])

    return t


def burn_tokens(tokens: list):
    """Burn token after each number"""

    t = []
    burn_next = False
    for token in tokens:
        if token.isdigit():
            burn_next = True
            t.append(token)
        elif burn_next:
            burn_next = False
        else:
            t.append(token)
    return t
    # t = []
    # j = merge_through + 1
    # for i in range(len(tokens) - j):
    #     if tokens[i].isdigit() and tokens[i+j].isdigit():
    #         tokens[i+j] = ''.join([tokens[i], tokens[i+j]])
    #     else:
    #         t.append(tokens[i])
    # for i in range(-j, 0):
    #     t.append(tokens[i])


def phone_detect(doc: str):
    """Detect phone number in string"""

    searched = 0
    tokens = doc.split()
    tokens = split_digits(tokens, ru_only=False)
    tokens = convert_word_numbers(tokens)

    # Find phone number
    m = phone_match(tokens)
    if m > PHONE_DETECT:
        return m
    elif m > searched:
        searched = m

    # Find phone number
    tokens = merge_neighbor_numbers(tokens)
    m = phone_match(tokens)
    if m > PHONE_DETECT:
        return m
    elif m > searched:
        searched = m

    for _ in range(2):
        # Merge numbers through i tokens
        tokens = burn_tokens(tokens)

        # Find phone number
        tokens = merge_neighbor_numbers(tokens)
        m = phone_match(tokens)
        if m > PHONE_DETECT:
            return m
        elif m > searched:
            searched = m

    return searched


def phone_match(tokens: list):
    """Match phone number by re"""

    for t in tokens:
        if m := PHONE_RE_1.fullmatch(t):
            if m.group(2)[0] == '9':
                if m.group(1):  # mobile with first 7 or 8
                    return PHONE_DETECT + 2
                else:  # mobile without first 7 or 8
                    return PHONE_DETECT + 1
            else:
                if m.group(1):  # home with first 7 or 8
                    return PHONE_DETECT + 4
                else:  # home without first 7 or 8
                    return PHONE_DETECT + 3
    doc = ' '.join(tokens)
    if m := PHONE_RE_2.search(doc):
        if m.group(1)[1] == '9':  # mobile with first 7 or 8
            return 2
        else:  # home with first 7 or 8
            return 4
    if m := PHONE_RE_3.search(doc):
        if m.group(1)[0] == '9':
            if m.group(1):  # mobile without first 7 or 8
                return 1
            else:  # home without first 7 or 8
                return 3
    return 0


def get_max_number(doc):
    max_number = 0
    for t in doc.split():
        if t.isdigit() and int(t) > max_number:
            max_number = int(t)
    return max_number


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
