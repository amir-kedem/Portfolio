# Shir Greenblat, 308570209
# Amir Kedem, 066560475

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parametric_y(t):
	return 0.5 * np.cos(4 * t) + 2 * np.cos(3 * t) + 4 * np.cos(2 * t) - 13 * np.cos(t)

def parametric_x(t):
	return 14.5 * np.sin(t)**3

def find_hough_shape(image, edge_image, r_min, r_max, bin_threshold):
	img_height, img_width = image.shape[:2]

	# Note that cos and sin work with radians
	thetas = np.arange(0, 360, step=2)
	rs = np.concatenate((r_min, r_max)) # this allows to have multiple radius range

	# Calculate Cos(theta) and Sin(theta), it will be required later
	cos_thetas = parametric_x(np.deg2rad(thetas))
	sin_thetas = parametric_y(np.deg2rad(thetas))

	# Quantize thetas and radii
	shape_candidates = []
	for r in rs:
		for theta in thetas:
			shape_candidates.append((r, theta))


	# Hough Accumulator. We are using defaultdict instead of standard dict as this will initialize for keys which are not already present in the dictionary instead of throwing an exception.
	accumulator = defaultdict(int)

	edge_points = np.argwhere(edge_image > 0)

	for y, x in edge_points:
		for r in rs:
			for idx, (cos_t, sin_t) in enumerate(zip(cos_thetas, sin_thetas)):
				y_center = int(x - r * cos_t)
				x_center = int(y - r * sin_t)
				if 0 <= x_center < img_width and 0 <= y_center < img_height:
					accumulator[(y_center, x_center, r)] += 1

	# Output image with detected lines drawn
	output_img = image.copy()
	# Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
	out_shapes = []

	num_thetas = len(thetas) * len(rs)

	max = 0 # to print the max threshold and give us an idea what value to choose

	# Sort the accumulator based on the votes for the candidate circles 
	for candidate_shape, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
		x, y, r = candidate_shape
		current_vote_percentage = votes / num_thetas
		# save the max value to know how to set the threshold
		if current_vote_percentage > max:
			max = current_vote_percentage
		if current_vote_percentage > bin_threshold:
			out_shapes.append((x, y, r, current_vote_percentage))

	print(max)

	# DO NOT EDIT
	pixel_threshold = 10
	postprocess_shapes = []
	for x, y, r, v in out_shapes:
		# Exclude shapes that are too close of each other
		# Remove nearby duplicate circles based on postprocess_shapes
		if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold > pixel_threshold and abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_shapes):
			postprocess_shapes.append((x, y, r, v))
	out_shapes = postprocess_shapes


	# Draw shortlisted hearts on the output image
	for x, y, r, v in out_shapes:
		t = np.deg2rad(thetas)
	
		# We found heart at (x,y) with radius r. 
		# Generate all the x1 and y1 points of that heart.
		y1 = ((0.5 * np.cos(4 * t) + 2 * np.cos(3 * t) + 4 * np.cos(2 * t) - 13 * np.cos(t)) * r + y).astype(int)
		x1 = ((np.sin(t) ** 3) * r * 14.5 + x).astype(int)

		colors = [
			(255, 0, 0),  # 'b' (blue)
			(0, 255, 0),  # 'g' (green)
			(255, 255, 0),  # 'c' (cyan)
			(255, 0, 255),  # 'm' (magenta)
			(0, 255, 255),  # 'y' (yellow)
			(0, 0, 0),  # 'k' (black)
		]
		
		color_chars = ['b', 'g', 'c', 'm', 'y', 'k']
		
		id = np.random.randint(len(colors))
		color1 = colors[id]
		color2 = color_chars[id]

		for i in range(len(x1) - 1):
			cv2.line(output_img, (x1[i], y1[i]), (x1[i + 1], y1[i + 1]), color1, 2)

		plt.plot(x1,y1, markersize=1.5, color=color2)
		output_img = cv2.circle(output_img, (x,y), 1, color1, -1)
		print(x, y, r, v)

	return output_img


IMAGE_NAME = "hard"

image = cv2.imread(f'{IMAGE_NAME}.jpg')
edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

min_edge_threshold, max_edge_threshold = 100, 200
edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)

#6-8 for simple
#2-6 for med
#3-4 and 11-12 for hard
r_min = np.arange(3, 4, 0.5)
r_max = np.arange(11, 12, 0.5)

#0.036 for simple
#0.033 for med
#0.065 for hard without false positives, 0.056 with one false positive
bin_threshold = 0.056

if edge_image is not None:
	
	print ("Attempting to detect Hough hearts...")
	results_img = find_hough_shape(image, edge_image, r_min, r_max, bin_threshold)
	
	if results_img is not None:
		plt.imshow(cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB))
		plt.show()
		cv2.imwrite(f'{IMAGE_NAME}_detected.jpg', results_img)
	else:
		print ("Error in input image!")

	print ("Hough hearts detection complete!")
