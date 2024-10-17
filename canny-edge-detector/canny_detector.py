import numpy as np
import filter


def filter_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Filters an image.

    Args:
        image: The image represented as a numpy array. If the image is not padded, the edges will not be properly handled.
        kernel: The filter kernel that the image will be filtered with.

    Returns: The filtered image.

    """
    num_rows = image.shape[0]
    num_cols = image.shape[1]

    filtered = np.zeros((num_rows, num_cols))
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


# def compute_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
# pass
