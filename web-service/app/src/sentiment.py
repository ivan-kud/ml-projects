from fastapi import Request
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import InputError, _timer


MAX_TEXT_LENGTH = 300
MODEL_CHECKPOINT = 'cardiffnlp/twitter-roberta-base-sentiment-latest'


def preprocess_text(text: str) -> str:
    # Check text length
    stripped_text = text.strip()
    if len(stripped_text) < 1:
        raise InputError('Write a review please.')
    if len(stripped_text) > MAX_TEXT_LENGTH:
        text = stripped_text[:MAX_TEXT_LENGTH]

    # Replace usernames and links by placeholders
    token_list = []
    for token in text.split(' '):
        token = '@user' if token.startswith('@') and len(token) > 1 else token
        token = 'http' if token.startswith('http') else token
        token_list.append(token)

    return ' '.join(token_list)


@_timer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT)

    return tokenizer, model


@_timer
def predict(model, model_input) -> tuple[float, str]:
    model.eval()
    with torch.no_grad():
        logits = model(**model_input).logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    proba = probabilities.max().item()
    class_id = probabilities.argmax().item()
    label = model.config.id2label[class_id]

    return proba, label


def get_response(text: str) -> dict[str, str | Request]:
    try:
        preprocessed_text = preprocess_text(text)
        (tokenizer, model), model_load_time = load_model()
        model_input = tokenizer(preprocessed_text, return_tensors='pt')
        (proba, label), inference_time = predict(model, model_input)
    except InputError as err:
        return {'output1': str(err)}
    except Exception as err:
        err_msg = type(err).__name__ + ': ' + str(err)
        print(f'File "{__name__}",', err_msg)
        return {'output1': err_msg}

    # Form the info string
    info = (f'Model load time: {int(model_load_time * 1000.0)} ms. '
            + f'Inference time: {int(inference_time * 1000.0)} ms.')

    return {
        'text': text,
        'output1': 'Label: ' + label,
        'output2': 'Confidence: ' + f'{100*proba:.2f} %',
        'info': info,
    }
