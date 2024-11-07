import canny_detector
import filter
import preprocess

img_path = "Test Images/lizard.jpg"

image = preprocess.load_image(img_path)
resized_image = preprocess.resize_image(image)
padded_image = preprocess.pad_image(image, 5 // 2)
gaussian = filter.gaussian_kernel(5, 1.4)
filtered_image = canny_detector.filter_image(padded_image, gaussian)
gradient, theta = canny_detector.compute_gradient(filtered_image)
rounded_theta = canny_detector.round_angles(theta)
cut_off = canny_detector.cut_off_supression(gradient, rounded_theta)
double_thresh = canny_detector.double_theshold(cut_off)
edges = canny_detector.hysteresis(double_thresh)
preprocess.show_image(edges)
