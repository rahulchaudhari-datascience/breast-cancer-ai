import cv2
import numpy as np


def read_image(path: str):

    image = cv2.imread(path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


def resize_image(
    image,
    size=(512, 512)
):

    return cv2.resize(
        image,
        size
    )


def normalize_image(image):

    image = image.astype(
        np.float32
    )

    image /= 255.0

    return image


def apply_clahe(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced



