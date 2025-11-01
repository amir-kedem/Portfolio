# Shir Greenblat, 308570209
# Amir Kedem, 066560475

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Method to create a gaussian kernel in the frequency domain
def create_gaussian_kernel(shape, sigma):
	h, w = shape
	x = np.linspace(-w//2, w//2 - 1, w) # center in x direction
	y = np.linspace(-h//2, h//2 - 1, h) # center in y direction
	x, y = np.meshgrid(x, y) # create the kernel
	kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2)) # calculate Gaussian blur function

	# Normalize the kernel
	kernel = kernel / kernel.sum()

	return kernel

# Method to apply a Guassian blur in the frequency domain
def apply_gaussian_blur(image, sigma=1):
	# FFT Transform for the image
	fft_image = np.fft.fft2(image)
	fft_image = np.fft.fftshift(fft_image)

	# Create Gaussian kernel in the frequency domain
	kernel = create_gaussian_kernel(image.shape, sigma)

	# Apply Gaussian kernel
	fft_blurred = fft_image * kernel

	# Return to Spatial Domain
	fft_blurred = np.fft.ifftshift(fft_blurred)
	blurred_image = np.fft.ifft2(fft_blurred)

	return np.abs(blurred_image)


def get_laplacian_pyramid(image, levels):

	current_image = image
	laplacian_pyramid = [] # empty array for the pyramid

	# Run a loop to build number of level or until the image is to small to continue scaling
	for _ in range(levels):
		# Blur the image with optimal gaussian kernel
		blurred_image = apply_gaussian_blur(current_image)

		# Calculate layer which is the difference between image and blurred image
		laplacian_layer = current_image - blurred_image
		laplacian_pyramid.append(laplacian_layer)

		# Reduce the image by taking every second pixel as seen in the lecture
		reduced_image = blurred_image[::2, ::2]

		# Stop if the image is too small to reduce further
		if reduced_image.shape[0] < 2 or reduced_image.shape[1] < 2:
			break

		# Prepare for next iteration
		current_image = reduced_image

	return laplacian_pyramid


def scale_up(image, resize_shape):
	# Convert dtype to float in order to preserve data
	image = image.astype(np.float32)

	# FFT Transform for the image
	f_transform = np.fft.fft2(image)
	f_shifted = np.fft.fftshift(f_transform)

	# Define scale down ratio
	h, w = f_shifted.shape[:2]
	new_h, new_w = resize_shape

	# Create zero padded array in size after scaling
	scaled_f_shifted = np.zeros((new_h, new_w), dtype=complex)

	# Calculate the center position and copy the original image to the center of the padded array
	start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
	scaled_f_shifted[start_h:start_h + h, start_w:start_w + w] = f_shifted

	# Convert back to the spatial domain
	f_ishifted = np.fft.ifftshift(scaled_f_shifted)
	scaled_image = np.fft.ifft2(f_ishifted)

	# Restore the average gray value
	scaled_image = np.abs(scaled_image)
	brightness_factor = 4 # we always double the size of the image
	scaled_image = np.clip(scaled_image * brightness_factor, 0, 255)

	return np.real(scaled_image)


def restore_from_pyramid(pyramidList):
	# Start from the smalled image - top of the pyramid
	current_level = pyramidList[-1]

	# Run in reverse order, second layer from the top and going down to the bottom layer
	for level in range(len(pyramidList) - 2, -1, -1):
		# Scale up the current image to the size of the next layer.
		upscale_image = scale_up(current_level, pyramidList[level].shape[:2])

		# Add the scaled image to the next laplacian layer to receive its respective original image
		current_level = cv2.add(upscale_image, pyramidList[level])

	# The last image is the original
	return current_level


def validate_operation(img):
	levels = 5
	pyr = get_laplacian_pyramid(img, levels)
	img_restored = restore_from_pyramid(pyr)

	plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
	plt.imshow(img_restored, cmap='gray')

	plt.show()
	
def blend_pyramids(pyr_apple, pyr_orange):
	blended = [] # array for the blended pyramid
	pyramid_size = len(pyr_apple)

	for level in range(pyramid_size):
		curr_level = level
		blend_size = pyramid_size - curr_level # the size of the blending area increases as we go down the pyramid
		y, x = pyr_apple[level].shape
		mask = np.zeros((y, x), dtype=np.float32) # prepare a blending mask
		mask[:, x // 2:] = 1  # left half to 0, right half to 1
		# Define a gradual blend which is a function of how much we approach the intersecting line
		for i in range(2 * (blend_size + 1)):
			mask[:, x // 2 - i] = 0.9 - 0.9 * i / (2 * (blend_size + 1))
		blended.append(pyr_apple[level]*mask + pyr_orange[level]*(1-mask)) # combine the two pyramids based on the mask

	return blended


def blend_pyramids_exponentail(pyr_apple, pyr_orange):
	blended = [] # array for the blended pyramid
	pyramid_size = len(pyr_apple)

	for level in range(pyramid_size):
		curr_level = level
		blend_size = pyramid_size - curr_level # the size of the blending area increases as we go down the pyramid
		y, x = pyr_apple[level].shape
		mask = np.zeros((y, x), dtype=np.float32) # prepare a blending mask
		mask[:, x // 2:] = 1  # left half to 0, right half to 1
		# Define a gradual blend which is a function of how much we approach the intersecting line
		for i in range(2 * (blend_size + 1)):
			mask[:, x // 2 - i] = (0.9 - 0.9 * i / (2 * (blend_size + 1)))**2
		blended.append(pyr_apple[level]*mask + pyr_orange[level]*(1-mask)) # combine the two pyramids based on the mask

	return blended

def blend_pyramids_width_control(pyr_apple, pyr_orange, levels, width):

	blended_pyramid = []

	for i in range(levels):
		rows, cols = pyr_apple[i].shape
		mask = np.zeros((rows, cols))

		# Feathering: create a linear transition for the mask
		transition_width = int(cols * width)  # width controls the width of the transition
		mask[:, :cols // 2 - transition_width // 2] = 1

		# Ensure the transition zone fits exactly into the mask slice
		transition_zone_start = cols // 2 - transition_width // 2
		transition_zone_end = transition_zone_start + transition_width
		transition_zone = np.linspace(1, 0, transition_zone_end - transition_zone_start)

		mask[:, transition_zone_start:transition_zone_end] = transition_zone

		# The rest of the mask to the right remains zero
		blended_layer = pyr_orange[i] * mask + pyr_apple[i] * (1 - mask)
		blended_pyramid.append(blended_layer)

	return blended_pyramid

######## start of code ########

apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

validate_operation(apple)
validate_operation(orange)

levels = 5

pyr_apple = get_laplacian_pyramid(apple, levels)
pyr_orange = get_laplacian_pyramid(orange, levels)

# Your code goes here

# Blend the two pyramids
pyr_result = blend_pyramids(pyr_apple, pyr_orange)

# Restore the image from the blended pyramid
final = restore_from_pyramid(pyr_result)

# Plot the result
plt.imshow(final, cmap='gray')
plt.show()

# Save the image
cv2.imwrite("result.jpg", final)

# Alternate solutions
pyr_result_exponential = blend_pyramids_exponentail(pyr_apple, pyr_orange)
cv2.imwrite("result_exponential.jpg", restore_from_pyramid(pyr_result_exponential))

pyr_result_small_transition = blend_pyramids_width_control(pyr_apple, pyr_orange, len(pyr_orange), 0.05)
cv2.imwrite("result_small_transition.jpg", restore_from_pyramid(pyr_result_small_transition))

pyr_result_medium_transition = blend_pyramids_width_control(pyr_apple, pyr_orange, len(pyr_orange), 0.1)
cv2.imwrite("result_medium_transition.jpg", restore_from_pyramid(pyr_result_medium_transition))

pyr_result_wide_transition = blend_pyramids_width_control(pyr_apple, pyr_orange, len(pyr_orange), 0.15)
cv2.imwrite("result_wide_transition.jpg", restore_from_pyramid(pyr_result_wide_transition))
