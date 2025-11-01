# Shir Grinblat, 308570209
# Amir Kedem, 066560475

import cv2
import numpy as np
from scipy import ndimage

# load the image
image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

# kernels that will be used for filtering
# kernel for sharpening
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# kernel for gibbs artifacts
gibbs_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# shift kernel
shift_kernel = [[1],[0]]
# mean kernel for each row
mean_kernel = np.ones((1, image.shape[1]))
mean_kernel /= float(image.shape[1])
# identity kernel
identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

# kernel for motion blur
kernel_size = 15
kernel_v = np.zeros((kernel_size, kernel_size))
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
kernel_v /= kernel_size #normalization

# run on the image and implement 9 different filters
for i in range(1, 10):
    match i:
        case 1:
            filter = ndimage.convolve(image, mean_kernel, mode='wrap', cval=0)
            cv2.imwrite(f'image_{i}fixed.jpg', filter)
        case 2:
            filter = cv2.GaussianBlur(image, (9, 9), 300, 300, borderType=cv2.BORDER_WRAP)
            cv2.imwrite(f'image_{i}fixed.jpg', filter)
        case 3:
            filter = cv2.medianBlur(image,9)
            cv2.imwrite(f'image_{i}fixed.jpg', filter)
        case 4:
            filter = ndimage.convolve(image, kernel_v, mode='wrap', cval=0)
            cv2.imwrite(f'image_{i}fixed.jpg', filter)
        case 5:
            filter = cv2.GaussianBlur(image, (9, 9), 300, 300, borderType=cv2.BORDER_WRAP)
            # convert values to handle negatives
            image_signed = image.astype(np.int16)
            filter_signed = filter.astype(np.int16)
            filter = image_signed - filter_signed # reduce the image to get the B sharp
            filter = filter + 128 # correct to be able to present the image
            filter = np.clip(filter, 0, 255).astype(np.uint8) # verify we have 0-255 range
            cv2.imwrite(f'image_{i}fixed.jpg', filter)
        case 6:
            filter = cv2.bilateralFilter(image, 5, 100, 100)
            filter = cv2.filter2D(filter, -1, gibbs_kernel)
            cv2.imwrite(f'image_{i}fixed.jpg', filter)
        case 7:
            filter = image.copy()
            for j in range(1, image.shape[0] // 2):
                filter = ndimage.convolve(filter, shift_kernel, mode='wrap', cval=0)
            cv2.imwrite(f'image_{i}fixed.jpg', filter)
        case 8:
            filter = ndimage.convolve(image, identity_kernel, mode='wrap', cval=0)
            cv2.imwrite(f'image_{i}fixed.jpg', filter)
        case 9:
            filter = cv2.filter2D(image, -1, sharpen_kernel)
            cv2.imwrite(f'image_{i}fixed.jpg', filter)
