import random

import cv2 as cv
from fastapi import UploadFile, Request
import numpy as np

from . import file2opencv, opencv2base64, InputError, _timer


MODEL_PATH = './ml-models/mask-rcnn-coco/'
IMAGE_MAX_SIZE = 10000
IMAGE_MAX_SIZE_MODEL = 1024
with open(MODEL_PATH + 'labels.txt', 'r') as f:
    LABELS = f.read().split('\n')
COLORS = np.random.randint(0, 255, (len(LABELS), 3))


def draw_masks(image, boxes, masks, conf_threshold=0.5, mask_threshold=0.3):
    number_of_objects = boxes.shape[2]
    for i in range(number_of_objects):
        box = boxes[0, 0, i]
        conf = box[2]
        if conf > conf_threshold:
            class_id = int(box[1])
            left, top, right, bottom = get_box_edges(image, box)
            color = get_color(class_id)

            # Extract and resize the mask for the object
            mask = masks[i, class_id]
            mask = cv.resize(mask, (right - left + 1, bottom - top + 1))
            mask = (mask > mask_threshold)

            # Overlay mask
            overlay = 0.6
            roi = image[top:bottom+1, left:right+1][mask]
            image[top:bottom+1, left:right+1][mask] = (
                    overlay * color + (1 - overlay) * roi).astype(np.uint8)

            # Draw contours
            contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(
                image[top:bottom+1, left:right+1], contours, -1,
                (int(color[0]), int(color[1]), int(color[2])), 2,
            )


def draw_boxes(image, boxes, conf_threshold=0.5):
    number_of_objects = boxes.shape[2]
    for i in range(number_of_objects):
        box = boxes[0, 0, i]
        conf = box[2]
        if conf > conf_threshold:
            left, top, right, bottom = get_box_edges(image, box)

            # Draw a bounding box
            cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)


def draw_confs(image, boxes, conf_threshold=0.5):
    number_of_objects = boxes.shape[2]
    for i in range(number_of_objects):
        box = boxes[0, 0, i]
        conf = box[2]
        if conf > conf_threshold:
            class_id = int(box[1])
            left, top, right, bottom = get_box_edges(image, box)

            # Display the label at the top of the bounding box
            label = f'{LABELS[class_id]}: {int(100 * conf)} %'
            cv.putText(image, label, (left+2, top+20), cv.FONT_HERSHEY_SIMPLEX,
                       0.75, (255, 255, 255), 2)


def get_box_edges(image, box):
    height = image.shape[0]
    width = image.shape[1]

    # Extract the box edges
    left = int(width * box[3])
    top = int(height * box[4])
    right = int(width * box[5])
    bottom = int(height * box[6])

    # Limit the box edges
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(0, min(right, width - 1))
    bottom = max(0, min(bottom, height - 1))

    return left, top, right, bottom


def get_color(class_id, color_per_class=False):
    if color_per_class:
        color = COLORS[class_id % len(COLORS)]
    else:
        color_index = random.randint(0, len(COLORS) - 1)
        color = COLORS[color_index]

    return color


def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Check for maximum image size
    height, width = image.shape[:2]
    if height > IMAGE_MAX_SIZE or width > IMAGE_MAX_SIZE:
        raise InputError(
            f'Image width and height must be less than {IMAGE_MAX_SIZE}px.'
            + ' Choose another file.'
        )

    # Resize image for better Mask R-CNN performance
    if max(width, height) > IMAGE_MAX_SIZE_MODEL:
        aspect_ratio = width / height
        if aspect_ratio > 1.0:
            shape = (IMAGE_MAX_SIZE_MODEL, int(IMAGE_MAX_SIZE_MODEL
                                               / aspect_ratio))
        else:
            shape = (int(IMAGE_MAX_SIZE_MODEL * aspect_ratio),
                     IMAGE_MAX_SIZE_MODEL)
        image = cv.resize(image, shape, interpolation=cv.INTER_AREA)

    return image


@_timer
def load_model() -> cv.dnn.Net:
    model = cv.dnn.readNetFromTensorflow(
        MODEL_PATH + 'frozen_inference_graph.pb',
        MODEL_PATH + 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt',
    )
    model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    return model


def predict(model: cv.dnn.Net, image: np.ndarray):
    blob = cv.dnn.blobFromImage(image, swapRB=True, crop=False)
    model.setInput(blob)
    boxes, masks = model.forward(['detection_out_final',
                                  'detection_masks'])
    return boxes, masks


def get_response(file: UploadFile) -> dict[str, str | Request]:
    try:
        image = file2opencv(file)
        image = preprocess_image(image)
        model, load_time = load_model()
        boxes, masks = predict(model, image)
        draw_masks(image, boxes, masks)
        draw_boxes(image, boxes)
        draw_confs(image, boxes)
        image_base64 = opencv2base64(image)
    except InputError as err:
        return {'info': str(err)}
    except Exception as err:
        err_msg = type(err).__name__ + ': ' + str(err)
        print(f'File "{__name__}",', err_msg)
        return {'info': err_msg}

    # Form the info string
    t, _ = model.getPerfProfile()
    info = (f'Done! Model load time: {int(load_time * 1000.0)} ms. '
            + f'Inference time: {int(t * 1000.0 / cv.getTickFrequency())} ms.')

    return {'image': image_base64, 'info': info}
