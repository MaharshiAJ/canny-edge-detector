import canny_detector
import preprocess
import filter

img_path = "Test Images/lizard.jpg"

image = preprocess.load_image(img_path)
resized_image = preprocess.resize_image(image)
padded_image = preprocess.pad_image(image, 5 // 2)
gaussian = filter.gaussian_kernel(5, 1.4)
filtered_image = canny_detector.filter_image(padded_image, gaussian)
preprocess.show_image(filtered_image)
