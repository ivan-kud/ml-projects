from enum import Enum

import cv2 as cv
from datasets import load_from_disk
from fastapi import UploadFile, Request
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                          BlipImageProcessor, BlipModel)

from . import pil2opencv, opencv2base64, InputError, _timer


DATASET_PATH = './ml-models/search-data/'
CLIP_CHECKPOINT = 'openai/clip-vit-base-patch32'
BLIP_CHECKPOINT = 'Salesforce/blip-image-captioning-base'
IMAGE_MAX_SIZE = 5000


class ModelName(str, Enum):
    clip = 'clip'
    blip = 'blip'
    color = 'color'


@_timer
def load_dataset():
    dataset = load_from_disk(DATASET_PATH + 'dataset')

    # Restore indices
    dataset.load_faiss_index('clip_embedding',
                             DATASET_PATH + 'clip_index.faiss')
    dataset.load_faiss_index('blip_embedding',
                             DATASET_PATH + 'blip_index.faiss')
    dataset.load_faiss_index('color_embedding',
                             DATASET_PATH + 'color_index.faiss')
    return dataset


@_timer
def load_model(model_name: ModelName):
    if model_name is ModelName.clip:
        # Load CLIP model
        preprocessor = CLIPImageProcessor.from_pretrained(CLIP_CHECKPOINT)
        model = CLIPVisionModelWithProjection.from_pretrained(CLIP_CHECKPOINT)
    elif model_name is ModelName.blip:
        # Load BLIP model
        preprocessor = BlipImageProcessor.from_pretrained(BLIP_CHECKPOINT)
        model = BlipModel.from_pretrained(BLIP_CHECKPOINT)
    else:
        preprocessor = None
        model = None

    return preprocessor, model


def get_embedding(model_name: ModelName,
                  pil_image: Image) -> tuple[np.ndarray, float, float]:
    # Check image size
    width, height = pil_image.size
    if height > IMAGE_MAX_SIZE or width > IMAGE_MAX_SIZE:
        raise InputError(
            f'Image width and height must be less than {IMAGE_MAX_SIZE}px.'
            + ' Choose another file.'
        )

    if model_name in [ModelName.clip, ModelName.blip]:
        # Load preprocessor and model
        (preprocessor, model), model_load_time = load_model(model_name)

        # Preprocess image
        model_input = preprocessor(pil_image, return_tensors='pt')

        # Get image embedding
        embedding, inference_time = get_clip_embedding(model_name, model,
                                                       model_input)
    else:
        # Get image embedding
        embedding, inference_time = get_color_embedding(pil_image)

        model_load_time = 0

    return embedding, model_load_time, inference_time


@_timer
def get_clip_embedding(model_name: ModelName,
                       model, model_input) -> np.ndarray:
    # Model inference
    model.eval()
    with torch.no_grad():
        if model_name is ModelName.clip:
            embedding = model(**model_input).image_embeds.numpy()
        elif model_name is ModelName.blip:
            embedding = model.get_image_features(**model_input).numpy()

    # Normalize embedding to use cosine similarity later
    embedding = np.squeeze(embedding)
    embedding /= np.linalg.norm(embedding)

    return embedding


@_timer
def get_color_embedding(pil_image: Image) -> np.ndarray:
    # Resize image
    image = cv.resize(pil2opencv(pil_image), (2, 2), cv.INTER_AREA)

    # Convert image back to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Compute embedding (flatten and scale)
    embedding = (image.flatten() / 127.5) - 1.0

    return embedding


@_timer
def search_similar(dataset, model_name,
                   embedding) -> tuple[np.ndarray, list[Image]]:
    scores, images = dataset.get_nearest_examples(
        model_name + '_embedding', embedding, k=5)

    return scores, images['image']


def many2one_image(query_image: Image, scores: np.ndarray,
                   images: list[Image]) -> Image:
    cols, rows = 3, 2
    w, h = 200, 200
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate([query_image] + images):
        image = image.resize((w, h))
        draw = ImageDraw.Draw(image)
        text = 'Query image' if i == 0 else str(scores[i-1])
        draw.text((0, 0), text, (255, 255, 255))
        grid.paste(image, box=(i % cols * w, i // cols * h))

    return grid


def get_response(model_name: ModelName,
                 file: UploadFile) -> dict[str, str | Request]:
    try:
        pil_image = Image.open(file.file)
        dataset, dataset_load_time = load_dataset()
        embedding, model_load_time, inference_time = get_embedding(
            model_name, pil_image)
        (scores, similar_images), search_time = search_similar(
            dataset, model_name, embedding)
        result_image = many2one_image(pil_image, scores, similar_images)
        image_base64 = opencv2base64(pil2opencv(result_image))
    except InputError as err:
        return {'info': str(err)}
    except Exception as err:
        err_msg = type(err).__name__ + ': ' + str(err)
        print(f'File "{__name__}",', err_msg)
        return {'info': err_msg}

    # Form the info strings
    unit = ' ms'
    left_just = 25
    right_just = 6
    time1 = ('Dataset load time:'.ljust(left_just)
             + str(int(dataset_load_time * 1000.0)).rjust(right_just) + unit)
    time2 = ('Model load time:'.ljust(left_just)
             + str(int(model_load_time * 1000.0)).rjust(right_just) + unit)
    time3 = ('Model inference time:'.ljust(left_just)
             + str(int(inference_time * 1000.0)).rjust(right_just) + unit)
    time4 = ('Search time:'.ljust(left_just)
             + str(int(search_time * 1000.0)).rjust(right_just) + unit)

    return {
        'model': model_name,
        'image': image_base64,
        'info': 'Done!',
        'time1': time1, 'time2': time2, 'time3': time3, 'time4': time4,
    }
