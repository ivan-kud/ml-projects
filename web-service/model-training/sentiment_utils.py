import os

import requests
import nltk
import tqdm


nltk.download('stopwords')


class DatasetError(Exception):
    pass


class Tokenizer:
    def __init__(self, corpus=None):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
        self.stemmer = nltk.stem.porter.PorterStemmer()

        self.corpus = corpus
        self.corpus_index = 0

    def __call__(self, text: str, return_str: bool = False) -> list[str] | str:
        # To lower case
        tokens = text.strip(' "').lower()

        # Tokenize
        tokens = self.tokenizer.tokenize(tokens)

        tokens_temp = []
        for token in tokens:
            # Remove stop words
            if token in self.stop_words:
                continue

            # Replace usernames and links by placeholders
            token = '@user' if token.startswith('@') and len(token) > 1 else token
            token = 'http' if token.startswith('http') else token

            # Stemming
            token = self.stemmer.stem(token, to_lowercase=False)

            tokens_temp.append(token)
        tokens = tokens_temp

        # Add a word if len is zero
        if len(tokens) == 0:
            tokens = ['word']

        return ' '.join(tokens) if return_str else tokens

    def __iter__(self):
        return self

    def __next__(self):
        try:
            text = self.corpus[self.corpus_index]
        except IndexError:
            raise StopIteration
        self.corpus_index += 1

        return self(text)

    def __getitem__(self, index):
        return self(self.corpus[index])


def preprocess_text(text: str | list[str]) -> str | list[str]:
    """Replace usernames and links by placeholders"""
    def preprocess(text):
        tokens = text.split(' ')
        for token in text.split(' '):
            token = '@user' if token.startswith('@') and len(
                token) > 1 else token
            token = 'http' if token.startswith('http') else token
            tokens.append(token)
        return ' '.join(tokens)

    if isinstance(text, list):
        return [preprocess(t) for t in text]
    else:
        return preprocess(text)


def dataset_query(api_url):
    response = requests.request('GET', api_url)
    return response.json()


def download_dataset(dataset_name, dataset_conf, dataset_path):
    query = 'is-valid'
    api_url = f'https://datasets-server.huggingface.co/{query}?dataset={dataset_name}'
    api_response = dataset_query(api_url)
    if api_response['valid']:
        query = 'parquet'
        api_url = f'https://datasets-server.huggingface.co/{query}?dataset={dataset_name}'
        api_response = dataset_query(api_url)
        for config in api_response['parquet_files']:
            if config['config'] == dataset_conf:
                response = requests.get(config['url'], stream=True)
                file_name = config['url'].split('/')[-1]
                file_path = dataset_path + file_name
                if not os.path.isfile(file_path):
                    with open(file_path, 'wb') as handle:
                        for data in tqdm.tqdm(response.iter_content()):
                            handle.write(data)
    else:
        raise DatasetError(f'Dataset "{dataset_name}" is not valid.')
