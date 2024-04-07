import base64
import binascii
from enum import Enum
import io

from fastapi import Request
import PIL
import PIL.ImageOps
from PIL import Image
import torch
from torch import nn
from torchvision.transforms.functional import to_tensor


IMG_WIDTH, IMG_HEIGHT = 28, 28
CLASSES = 10
CHANNELS = 1
DATA_MEAN, DATA_STD = 0.13, 0.31
DATA_SUBZERO = (0 - DATA_MEAN) / DATA_STD
MODEL_PATH = './ml-models/'


class LogRegModel(nn.Module):
    """Logistic Regression model"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(IMG_HEIGHT * IMG_WIDTH, CLASSES)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class Dense3Model(nn.Module):
    """Dense model with 3 fully connected layers"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(IMG_HEIGHT * IMG_WIDTH, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, CLASSES)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


class Conv3Model(nn.Module):
    """Convolutional model with 2 conv and 1 FC layers"""
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # Conv 1 layer
            nn.ConstantPad2d(1, DATA_SUBZERO),
            nn.Conv2d(CHANNELS, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),

            # Conv 2 layer
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 7 * 7, CLASSES)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class Conv5Model(nn.Module):
    """Convolutional model with 4 conv and 1 FC layers"""
    def __init__(self, ch=(8, 16, 32, 64)):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # Conv 1 layer
            nn.ConstantPad2d(1, DATA_SUBZERO),
            nn.Conv2d(CHANNELS, ch[0], 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv 2 layer
            nn.Conv2d(ch[0], ch[1], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv 3 layer
            nn.Conv2d(ch[1], ch[2], 3, padding=1),
            nn.ReLU(),
            nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.MaxPool2d(2),

            # Conv 4 layer
            nn.Conv2d(ch[2], ch[3], 4),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(ch[3], CLASSES)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class ModelName(str, Enum):
    logreg = 'logreg'
    dense3 = 'dense3'
    conv3 = 'conv3'
    conv5 = 'conv5'
    all = 'all'


def preprocess_image(img: str) -> torch.Tensor:
    # Convert base64 encoded PNG image to byte array
    image_base64 = img.split(';base64,')[-1]
    try:
        image_bytes = base64.b64decode(image_base64)
    except binascii.Error:
        raise binascii.Error('String of base64 image is incorrectly padded')

    # Open by PIL, convert to grayscale, invert
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert('L')
    pil_image = PIL.ImageOps.invert(pil_image)

    # Transform to Tensor, normalize, add dimension
    image_tensor = to_tensor(pil_image)
    image_tensor = (image_tensor - DATA_MEAN) / DATA_STD
    image_tensor = torch.unsqueeze(image_tensor, dim=0)

    return image_tensor


def load_model(model_name: ModelName) -> nn.Module:
    if model_name is ModelName.logreg:
        model = LogRegModel()
    elif model_name is ModelName.dense3:
        model = Dense3Model()
    elif model_name is ModelName.conv3:
        model = Conv3Model()
    elif model_name is ModelName.conv5:
        model = Conv5Model((32, 32, 32, 64))
    else:
        raise ValueError(f'Model name "{str(model_name)}" is undefined.')
    path = MODEL_PATH + 'digit_' + model_name + '.pt'
    model.load_state_dict(torch.load(path))

    return model


def predict(model: nn.Module, image: torch.Tensor) -> tuple[float, int]:
    model.eval()
    with torch.no_grad():
        probabilities = nn.Softmax(dim=1)(model(image))[0]
        proba = probabilities.max().item()
        label = probabilities.argmax().item()

    return proba, label


def get_response(model_name: ModelName,
                 image: str) -> dict[str, str | Request]:
    # Preprocess image to use it as model input
    try:
        image_tensor = preprocess_image(image)
    except Exception as err:
        err_msg = type(err).__name__ + ': ' + str(err)
        print(f'File "{__name__}",', err_msg)
        return {'output1': err_msg}

    # Load model and predict
    result = {}
    for name in ModelName:
        if name is not ModelName.all and (model_name is name
                                          or model_name is ModelName.all):
            try:
                model = load_model(name)
                result[name] = predict(model, image_tensor)
            except Exception as err:
                err_msg = type(err).__name__ + ': ' + str(err)
                print(f'File "{__name__}",', err_msg)
                return {'output1': err_msg}

    # Form result strings
    if len(result) == 1:
        output1 = f'{result[model_name][1]}'
        output2 = f'{100*result[model_name][0]:.2f} %'
    else:
        output1 = '; '.join([f'{name} - {value[1]}      '
                            for name, value in result.items()])
        output2 = '; '.join([f'{name} - {100*value[0]:.2f} %'
                            for name, value in result.items()])

    return {
        'model': model_name,
        'image': image,
        'output1': 'Label: ' + output1,
        'output2': 'Confidence: ' + output2,
    }
