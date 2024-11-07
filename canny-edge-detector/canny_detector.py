import filter
import numpy as np


def filter_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Filters an image.

    Args:
        image: The image represented as a numpy array. If the image is not padded, the edges will not be properly handled.
        kernel: The filter kernel that the image will be filtered with.

    Returns: The filtered image.

    """
    num_rows = image.shape[0]
    num_cols = image.shape[1]

    filtered = np.zeros_like(image)
    kernel_size = kernel.shape[0]
    half_kernel_size = kernel_size // 2
    i, j = half_kernel_size, half_kernel_size

    for row in range(half_kernel_size, num_rows - half_kernel_size):
        for col in range(half_kernel_size, num_cols - half_kernel_size):
            filtered[i, j] = filter.convolve(
                filter.get_window(image, row, col, kernel_size), kernel
            )
            j += 1
        i += 1
        j = 0

    # Normalizing the filtered image
    filtered_max = filtered.max()
    filtered_min = filtered.min()

    filtered = (filtered - filtered_min) / (filtered_max - filtered_min) * 255
    return filtered.astype(np.uint8)


def compute_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the gradient of an image using Sobel Operators.

    Args:
        image: The image represented as a numpy array.

    Returns: A tuple containing the gradient of the image as the first value and the direction of edges as the second value.

    """
    horizontal = filter.HORIZONTAL_GRADIENT_KERNEL
    vertical = filter.VERTICAL_GRADIENT_KERNEL

    G_x = filter_image(image, horizontal)
    G_y = filter_image(image, vertical)

    return np.sqrt(np.square(G_x) + np.square(G_y)).astype(np.uint8), np.rad2deg(
        np.arctan2(G_y, G_x)
    )


def round_angles(theta: np.ndarray) -> np.ndarray:
    """Rounds angles to the closest quadrantal angles or special angles (0, 45, 90, 135)

    Args:
        theta: Numpy array containing the direction of edges (in degrees).

    Returns: A numpy array containing the direction of edges rounded to the closest quadrantal or special angles (0, 45, 90, 135).

    """
    target = np.array([0, 45, 90, 135])
    rounded = np.array(
        [target[np.argmin(np.abs(target - angle))] for angle in theta.flatten()]
    )
    return rounded.reshape(theta.shape)


def cut_off_supression(gradient: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Suppresses pixels in the gradient that are along weak edges.

    Args:
        gradient: Numpy array containing the gradient of the image
        theta: Numpy array containing the direction of edges rounded to the closest quadrantal or special angles.

    Returns: The gradient image with suppressed weak edges.

    """
    result = gradient
    num_rows = gradient.shape[0]
    num_cols = gradient.shape[1]

    for row in range(1, num_rows - 1):
        for col in range(1, num_cols - 1):
            current = gradient[row, col]
            angle = theta[row, col]

            if angle == 0:
                east = gradient[row, col + 1]
                west = gradient[row, col - 1]

                if current <= east and current <= west:
                    result[row, col] = 0
            elif angle == 45:
                northeast = gradient[row - 1, col + 1]
                southwest = gradient[row + 1, col - 1]

                if current <= northeast and current <= southwest:
                    result[row, col] = 0
            elif angle == 90:
                north = gradient[row - 1, col]
                south = gradient[row + 1, col]

                if current <= north and current <= south:
                    result[row, col] = 0
            elif angle == 135:
                northwest = gradient[row - 1, col - 1]
                southeast = gradient[row + 1, col + 1]

                if current <= northwest and current <= southeast:
                    result[row, col] = 0

    return result


def double_theshold(gradient: np.ndarray) -> np.ndarray:
    """Thresholds the image gradient using two thresholds that are based on the strongest pixel intensity within the gradient.
    The high threshold is 20% of the highest intensity pixel and the low threshold is 10% of the highest intensity pixel.

    Args:
        gradient: Numpy array containing the gradient of the image


    Returns: The thresholded image as a numpy array.

    """
    thresholded = np.zeros_like(gradient)
    max_pixel = gradient.max()
    high = max_pixel * 0.2
    low = max_pixel * 0.1

    for row in range(1, gradient.shape[0] - 1):
        for col in range(1, gradient.shape[1] - 1):
            if gradient[row, col] > high:
                thresholded[row, col] = 255
            elif gradient[row, col] <= high and gradient[row, col] > low:
                thresholded[row, col] = gradient[row, col]

    return thresholded


def hysteresis(thresholded_gradient: np.ndarray) -> np.ndarray:
    """Locates weak edges within the gradient and looks at its 8 neighbors to determine if should be a strong edge. If not the edge is suppressed.

    Args:
        thresholded_gradient: The gradient of the image after undergoing cut-off suppression and double thresholding

    Returns: A numpy array containing the image gradient showing the edges of the image.

    """
    result = np.copy(thresholded_gradient)

    for row in range(1, result.shape[0] - 1):
        for col in range(1, result.shape[1] - 1):
            current = result[row, col]

            if current == 0 or current == 255:
                continue

            neighborhood = get_neighborhood_3x3(result, row, col)

            if 255 in neighborhood:
                result[row, col] = 255
            else:
                result[row, col] = 0

    return result


def get_neighborhood_3x3(arr: np.ndarray, x: int, y: int) -> np.ndarray:
    """Gets a 3x3 neighborhood based on a location within an array

    Args:
        arr: Numpy array containing the data.
        x: The x coordinate of the location.
        y: The y coordinate of the location.

    Returns: A 3x3 numpy array containing the neighborhood of the location within arr specified by the location [x, y].

    """
    return arr[x - 1 : x + 2, y - 1 : y + 2]
