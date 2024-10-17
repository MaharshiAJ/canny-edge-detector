from cv2 import imread, IMREAD_GRAYSCALE, resize, imshow, waitKey
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Loads an image in grayscale

    Args:
        path: Path to image

    Returns: The image loaded as a numpy array

    """
    return imread(path, IMREAD_GRAYSCALE)


def resize_image(image: np.ndarray) -> np.ndarray:
    """Resizes an image to 256px by 256px

    Args:
        image: The image represented by a numpy array

    Returns: The resized image

    """
    return resize(image, dsize=(256, 256))


def pad_image(image: np.ndarray, width: int, value: int = 0) -> np.ndarray:
    """Pads the edges of an image

    Args:
        image: The image represented by a numpy array
        width: How many rows/columns to add to each edge of the image
        value: What value to pad the edges with, default = 0

    Returns: The padded image

    """
    return np.pad(image, width, mode="constant", constant_values=value)


def show_image(image: np.ndarray) -> None:
    """Displays an image in a window

    Args:
        image: The image represented by a numpy array
    """
    imshow("", image)
    waitKey(0)
