import base64
import tempfile
import time

import cv2 as cv
from fastapi import UploadFile
import numpy as np
from PIL import Image


class InputError(Exception):
    pass


def _timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_return = func(*args, **kwargs)
        end_time = time.time()
        return func_return, (end_time - start_time)
    return wrapper


def file2opencv(file: UploadFile) -> np.ndarray:
    """Convert UploadFile to OpenCV image"""
    # Convert UploadFile to numpy array
    file_array = np.fromfile(file.file, np.uint8)

    # Check for empty file
    if len(file_array) == 0:
        raise InputError('Choose a file.')

    # Convert numpy array to OpenCV image
    image = cv.imdecode(file_array, cv.IMREAD_COLOR)
    if image is None:  # check operation status
        raise InputError('Image must be in JPEG or PNG format.'
                         + ' Choose another file.')
    return image


def opencv2base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    # Encode OpenCV image to numpy array of JPEG image format
    status, file_array = cv.imencode('.jpg', image)
    if not status:
        raise ValueError("OpenCV can't encode image to JPEG format.")

    # Convert numpy array to temporary file and encode to base64 format string
    with tempfile.SpooledTemporaryFile() as fp:
        file_array.tofile(fp)
        fp.seek(0)
        bytes_array = base64.b64encode(fp.read())
    image_base64 = 'data:image/jpeg;base64,'
    image_base64 += bytes_array.decode()

    return image_base64


def pil2opencv(pil_image: Image) -> np.ndarray:
    opencv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
    return opencv_image
