# Shir Greenblat, 308570209
# Amir Kedem, 066560475

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import warnings
warnings.filterwarnings("ignore")

def scale_down(image, resize_ratio):
	# FFT Transform for the image
	f_transform = fft2(image)
	f_shifted = fftshift(f_transform)

	# Define scale down ratio (assumption it is between 0 and 1)
	h, w = f_shifted.shape[:2]
	new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)
	cropped = f_shifted[(h - new_h) // 2:(h + new_h) // 2, (w - new_w) // 2:(w + new_w) // 2]

	# Return to Spatial Domain
	f_ishifted = ifftshift(cropped)
	down_scaled_image = ifft2(f_ishifted)

	# Restore the average gray value
	down_scaled_image = np.abs(down_scaled_image)
	brightness_factor = resize_ratio**2 # assumption resize_ratio is between 0 and 1
	down_scaled_image = np.clip(down_scaled_image * brightness_factor, 0, 255)

	return np.real(down_scaled_image)


def scale_up(image, resize_ratio):
	# Convert dtype to float in order to preserve data
	image = image.astype(np.float32)

	# FFT Transform for the image
	f_transform = np.fft.fft2(image)
	f_shifted = np.fft.fftshift(f_transform)

	# Define scale up ratio
	h, w = f_shifted.shape[:2]
	new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)

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
	brightness_factor = resize_ratio**2 # assumption resize_ratio is at least 1
	scaled_image = np.clip(scaled_image * brightness_factor, 0, 255)

	return np.real(scaled_image)


def ncc_2d(image, pattern):
	# Ensure the image and pattern are in float for precision
	image = image.astype(np.float32)
	pattern = pattern.astype(np.float32)

	# Get the mean of the pattern
	pattern_mean = pattern.mean()

	# Subtract the mean from the pattern
	pattern_zero_mean = pattern - pattern_mean

	# Calculate the denominator for NCC
	pattern_denom = np.sqrt((pattern_zero_mean ** 2).sum())

	# Create sliding windows of the same shape as the pattern
	windows = np.lib.stride_tricks.sliding_window_view(image, pattern.shape)

	# Calculate the mean of each window
	window_means = windows.mean(axis=(-2, -1))

	# Subtract the mean from each window
	windows_zero_mean = windows - window_means[..., np.newaxis, np.newaxis]

	# Calculate NCC numerator and denominator for each window
	numerator = np.sum(windows_zero_mean * pattern_zero_mean, axis=(-2, -1))
	denominator = np.sqrt(np.sum(windows_zero_mean ** 2, axis=(-2, -1))) * pattern_denom

	# Avoid division by zero
	denominator = np.where(denominator == 0, 1, denominator)

	# Calculate NCC
	ncc = numerator / denominator

	return ncc


def display(image, pattern):
	
	plt.subplot(2, 3, 1)
	plt.title('Image')
	plt.imshow(image, cmap='gray')
		
	plt.subplot(2, 3, 3)
	plt.title('Pattern')
	plt.imshow(pattern, cmap='gray', aspect='equal')
	
	ncc = ncc_2d(image, pattern)
	
	plt.subplot(2, 3, 5)
	plt.title('Normalized Cross-Correlation Heatmap')
	plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto') 
	
	cbar = plt.colorbar()
	cbar.set_label('NCC Values')
		
	plt.show()

def draw_matches(image, matches):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	for match, size in matches:
		pattern_size = (size[0], size[1])
		y, x = match
		top_left = (int(x - pattern_size[1]/2), int(y - pattern_size[0]/2))
		bottom_right = (int(x + pattern_size[1]/2), int(y + pattern_size[0]/2))
		cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)
	
	plt.imshow(image, cmap='gray')
	plt.show()
	
	cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)


def check_pattern(scale_and_threshold, image, pattern):
	# Array to return all matches found
	matches = []

	# Run on all scale and threshold combinations
	for scale, thresh in scale_and_threshold:
		pattern_scaled = scale_down(pattern, scale) # scale down the pattern
		ncc = ncc_2d(image, pattern_scaled) # calculate NCC
		display(image, pattern_scaled) # display heatmap of correlation
		y_indices, x_indices = np.where(ncc > thresh) # keep all windows which match defined threshold
		current_matches = np.vstack((y_indices, x_indices)).T
		current_matches[:, 0] += pattern_scaled.shape[0] // 2 # calculate mid point
		current_matches[:, 1] += pattern_scaled.shape[1] // 2 # calculate mid point
		for match in current_matches:
			matches.append((match, pattern_scaled.shape)) # keep all matches and its respected pattern size

	return matches


# Method to check if two points are within 10 pixels distance
def are_close(point1, point2):
	return np.abs(point1[0] - point2[0]) <= 10 and np.abs(point1[1] - point2[1]) <= 10

# Method to remove duplicate windows, which are defined either as:
# 1. Center coordinate closer than 10 pixels away
# 2. Same center coordinate which was in more than one pattern size
def remove_duplicates(matches):
	unique = [] # array for unique values
	seen_points = [] # array to keep all

	for item in matches:
		point = item[0]  # get the x,y coordinates from the numpy array
		# check if the center point is closer than 10 pixels, or if i've already saved the same center point of the face
		if not any(are_close(point, seen) for seen in seen_points):
			unique.append(item)
			seen_points.append(point)

	return unique

########## start of code #############

CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# DEMO #############
display(image, pattern)


############# Students #############

# Define scale and threshold combinations
scale_and_threshold = [(0.6, 0.5), (0.5, 0.55), (0.45, 0.6), (0.4, 0.6)]

# Keep array of all matches + the pattern size they were found
matches = check_pattern(scale_and_threshold, image, pattern)

# Sort all found matches by pattern size (bigger is first) and x,y coordinates (by y coordinates in ascending order)
matches.sort(key=lambda item: (-np.sqrt(item[1][0]**2 + item[1][1]**2), item[0][0]))

# Remove duplicate faces found
unique = remove_duplicates(matches)

# Draw all faces matched
draw_matches(image, unique)


############# Crew #############

CURR_IMAGE = "thecrew"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

# Define scale and threshold combinations
scale_and_threshold = [(0.4, 0.4), (0.3, 0.4), (0.25, 0.45), (0.2, 0.51)]

# Keep array of all matches + the pattern size they were found
matches = check_pattern(scale_and_threshold, image[:76, :], pattern)

# Sort all found matches by pattern size (bigger is first) and x,y coordinates (by y coordinates in ascending order)
matches.sort(key=lambda item: (-np.sqrt(item[1][0]**2 + item[1][1]**2), item[0][0]))

# Remove duplicate faces found
unique = remove_duplicates(matches)

# Draw all faces matched
draw_matches(image, unique)
