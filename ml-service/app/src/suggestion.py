from fastapi import Request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import RobertaTokenizerFast, RobertaModel

from . import InputError, _timer


MAX_TEXT_LENGTH = 1000
MAX_SUGGESTION_NUM = 5
DATA_PATH = './ml-models/suggestion-data/'
MODEL_CHECKPOINT = 'roberta-base'
MODEL_INTERNAL_DIM = 768


def read_terms() -> list[str]:
    with open(DATA_PATH + 'terms.txt') as f:
        terms = f.read()
    terms = terms.lower().split('\n')
    terms = [' ' + term for term in terms]

    return terms


def preprocess_text(text: str) -> str:
    # Check text length
    stripped_text = text.strip()
    if len(stripped_text) < 1:
        raise InputError('Write a text please.')
    if len(stripped_text) > MAX_TEXT_LENGTH:
        text = stripped_text[:MAX_TEXT_LENGTH]

    return text


def get_hidden_state(tokenizer, model, text: str) -> tuple[np.ndarray, np.ndarray]:
    # Get offset mapping
    model_input = tokenizer(
        text,
        return_tensors='pt',
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
    )
    offset_mapping = model_input['offset_mapping'].detach().numpy().squeeze()

    # Get last hidden state
    with torch.no_grad():
        model_output = model(
            input_ids=model_input['input_ids'],
            attention_mask=model_input['attention_mask'],
        )
    last_hidden_state = model_output[
        'last_hidden_state'].detach().numpy().squeeze()

    # Mask to get no special tokens
    mask = model_input['special_tokens_mask'].detach().numpy().squeeze() == 0
    last_hidden_state = last_hidden_state[mask, :]
    offset_mapping = offset_mapping[mask, :]

    return last_hidden_state, offset_mapping


def get_term_embeddings(terms: list[str]) -> np.ndarray:
    # Load model
    (tokenizer, model), _ = load_model()

    # Compute mean pooling embedding
    term_embeddings = np.empty((len(terms), MODEL_INTERNAL_DIM))
    for i, term in enumerate(terms):
        last_hidden_state, _ = get_hidden_state(tokenizer, model, term)
        term_embeddings[i] = (np.sum(last_hidden_state, axis=0)
                              / last_hidden_state.shape[0])

    return term_embeddings


@_timer
def get_text_embeddings(text: str) -> tuple[np.ndarray, np.ndarray]:
    # Load model
    (tokenizer, model), _ = load_model()

    # get last_hidden_state and offset_mapping
    last_hidden_state, offset_mapping = get_hidden_state(tokenizer, model, text)

    # Compute mean pooling embeddings for all contexts in sample text
    length = get_number_of_contexts(last_hidden_state.shape[0])
    context_embedding = np.empty((length, MODEL_INTERNAL_DIM))
    context_mapping = np.empty((length, 2), int)
    index = 0
    for i in range(last_hidden_state.shape[0]):
        for j in range(i, i + 8):
            if j < last_hidden_state.shape[0]:
                slice_ = last_hidden_state[i:j + 1, :]
                context_embedding[index] = (np.sum(slice_, axis=0)
                                            / slice_.shape[0])
                context_mapping[index] = np.array(
                    (offset_mapping[i, 0], offset_mapping[j, 1])
                )
                index += 1

    return context_embedding, context_mapping


def get_number_of_contexts(token_number: int) -> int:
    # Compute number of contexts
    number_of_contexts = 0
    for i in range(token_number):
        for j in range(i, i + 8):
            if j >= token_number:
                continue
            number_of_contexts += 1

    return number_of_contexts


@_timer
def load_model():
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    model = RobertaModel.from_pretrained(MODEL_CHECKPOINT)

    return tokenizer, model


def get_similarity_scores(context_embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Compute similarity and sort
    if context_embedding.shape[0]:
        similarity = cosine_similarity(context_embedding, term_embeddings)
    else:
        similarity = np.array([[]])
    sorted_flat_indices = np.flip(np.argsort(similarity, axis=None))

    return similarity, sorted_flat_indices


def get_suggestions(text: str, context_embedding: np.ndarray,
                    context_mapping: np.ndarray) -> (
        tuple[list[str], list[str], list[float], list[list]]
):
    # Get similarity scores
    similarity, sorted_similarity_indices = get_similarity_scores(context_embedding)

    # Get suggestions
    spans = []
    original_phrases = []
    replacements = []
    scores = []
    for index in sorted_similarity_indices:
        row = index // len(terms)
        col = index % len(terms)

        # Get current suggestion in the order
        text_span = context_mapping[row].tolist()
        original_phrase = text[text_span[0]:text_span[1]]
        replacement = terms[col][1:]
        score = similarity[row, col].item()

        # Check if it is a new span
        new_span = True
        for span in spans:
            if ((span[0] <= text_span[0] < span[1])
                    or (span[0] < text_span[1] <= span[1])
                    or (text_span[0] < span[0] and span[1] < text_span[1])):
                new_span = False

        # Add suggestion if it is a new span
        if new_span:
            spans.append(text_span)
            original_phrases.append(original_phrase)
            replacements.append(replacement)
            scores.append(score)

            if len(spans) >= MAX_SUGGESTION_NUM:
                break

    return original_phrases, replacements, scores, spans


def get_response(text: str) -> dict[str, str | Request]:
    try:
        preprocessed_text = preprocess_text(text)
        (context_embedding, context_mapping), inference_time = get_text_embeddings(preprocessed_text)
        original_phrases, replacements, scores, _ = get_suggestions(
            preprocessed_text, context_embedding, context_mapping
        )
    except InputError as err:
        return {'output1': str(err)}
    except Exception as err:
        err_msg = type(err).__name__ + ': ' + str(err)
        print(f'File "{__name__}",', err_msg)
        return {'output1': err_msg}

    # Form the info string
    info = f'Model load and inference time: {int(inference_time * 1000.0)} ms.'

    # Form result dict
    result = {'text': preprocessed_text, 'info': info}
    for i in range(len(scores)):
        result.update({
            f'output{i * 3 + 1}': f'Original phrase: {original_phrases[i]}',
            f'output{i * 3 + 2}': f'Suggestion:      {replacements[i]}',
            f'output{i * 3 + 3}': f'Score:           {scores[i]:.4f}',
        })

    return result


terms = read_terms()
term_embeddings = get_term_embeddings(terms)
