import numpy as np

# Sobel Operators for finding the gradient of an image
HORIZONTAL_GRADIENT_KERNEL = np.array(([1, 0, -1], [2, 0, -2], [1, 0, -1]))
VERTICAL_GRADIENT_KERNEL = np.array(([1, 2, 1], [0, 0, 0], [-1, -2, -1]))


def get_window(
    image: np.ndarray, row: int, column: int, window_size: int
) -> np.ndarray:
    """Get a window of an image centered around a point.

    Args:
        image: The image represented by a numpy array
        row: The x index of the center point
        column: The y index of the center point
        window_size: The size of the window

    Returns: A square matrix of size (window_size, window_size)

    """
    return image[
        row - (window_size) // 2 : row + (window_size // 2) + 1,
        column - (window_size // 2) : column + (window_size // 2) + 1,
    ]


def gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """Generates a Gaussian Filter Kernel.

    Args:
        kernel_size: The desired size of the kernel
        sigma: The desired standard deviation

    Returns: A square matrix representing the gaussian kernel of size kernel_size and of standard deviation sigma.

    """
    kernel = np.zeros((kernel_size, kernel_size))
    m = kernel_size // 2
    n = kernel_size // 2

    for row in range(-m, m + 1):
        for col in range(-n, n + 1):
            left = 1 / (2 * np.pi + (sigma * 2))
            right = np.exp(-(row**2 + col**2) / (2 * (sigma**2)))
            kernel[row + m, col + n] = left * right

    return kernel / np.sum(kernel)


def convolve(window: np.ndarray, filter: np.ndarray) -> float:
    """Computes the convolution between a window and a filter.
    Args:
        window: A window taken from an image.
        filter: The filter that the window will be convolved with.

    Returns: The resulting number after convolution is performed.

    """
    output = 0.0
    kernel_size = filter.shape[0]

    for row in range(kernel_size):
        sum = 0.0
        for col in range(kernel_size):
            sum += window[row, col] * filter[row, col]

        output += sum

    return output
