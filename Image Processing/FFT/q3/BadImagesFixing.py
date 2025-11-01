# Shir Grinblat, 308570209
# Amir Kedem, 066560475

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt


def cleanImageMedian(im, radius):
    median_image = im.copy()

    height, width = im.shape[:2]
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            window = im[i - radius: i + radius + 1, j - radius: j + radius + 1]
            median_image[i, j] = np.median(window)

    return median_image

def clean_baby(im):
    # choose corners of each image
    input_pts1 = np.float32([[77, 162], [146, 115], [244, 160], [132, 244]])  # bottom right
    input_pts2 = np.float32([[181, 4], [249, 71], [176, 120], [121, 50]])  # top right
    input_pts3 = np.float32([[5, 19], [110, 19], [110, 130], [5, 130]])  # straight baby - left
    # target point of the output image
    output_pts = np.float32([[0, 0], [255, 0], [255, 255], [0, 255]])  # entire frame

    # warp first image
    projective_matrix1 = cv2.getPerspectiveTransform(input_pts1, output_pts)
    img_proj1 = cv2.warpPerspective(im, projective_matrix1, (255, 255))
    # warp second image
    projective_matrix2 = cv2.getPerspectiveTransform(input_pts2, output_pts)
    img_proj2 = cv2.warpPerspective(im, projective_matrix2, (255, 255))
    # warp third image
    projective_matrix3 = cv2.getPerspectiveTransform(input_pts3, output_pts)
    img_proj3 = cv2.warpPerspective(im, projective_matrix3, (255, 255))
    # apply median blur before combining all images
    img_proj1 = cv2.medianBlur(img_proj1, 5)
    img_proj2 = cv2.medianBlur(img_proj2, 5)
    img_proj3 = cv2.medianBlur(img_proj3, 5)
    # take the median of each pixel
    new_image = np.median([img_proj1, img_proj2, img_proj3], axis=0)
    # apply filters to correct the final image
    new_image = cv2.medianBlur(new_image.astype(np.uint8), 5)
    new_image = cv2.bilateralFilter(new_image.astype(np.uint8), 7, 20, 150, borderType=cv2.BORDER_WRAP)

    return new_image

def clean_windmill(im):
    # transform to the frequency level + shift DC to center
    image_fourier = np.fft.fftshift(np.fft.fft2(im))
    # calculate the coefficient of each frequency
    magnitude = np.abs(image_fourier)

    peaks = [] # array to hold peak values which are noise
    threshold = 20 # define threshold to locate peaks
    # run on half of the image and check for high magnitude indexes
    for i in range(0, im.shape[0]//2):
        for j in range(0 , im.shape[1]//2):
            # if the coefficient is higher than three neighbors then it may be noise, assuming that further away
            # from the DC the coefficients should reduce
            if ((magnitude[i,j]>threshold*magnitude[i+1,j+1]) & (magnitude[i,j]>threshold*magnitude[i,j+1])
                    & (magnitude[i,j]>threshold*magnitude[i+1,j])):
                peaks.append([i,j])

    # iterate on all peak values and adjust the image
    for peak in peaks:
        i = peak[0]
        j = peak[1]
        # need to correct both sides
        inverse_i = im.shape[0]-i
        inverse_j = im.shape[1]-j

        # Calculate the average of specific high-frequency components to mitigate noise or unwanted details
        avg = (image_fourier[i-1, j] + image_fourier[i, j-1] + image_fourier[i+1, j] + image_fourier[
            i, j+1]) / 4

        # Replace specific high-frequency components with their averages to smooth out these frequencies
        image_fourier[i, j] = avg
        image_fourier[inverse_i, inverse_j] = avg

    # Apply inverse Fourier transform to convert the modified frequency domain back to the spatial domain
    cleaned_image = abs(np.fft.ifft2(np.fft.ifftshift(image_fourier)))

    return cleaned_image

def clean_watermelon(im):
    # define the kernel
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # apply cv2 function for convolution with the kernel
    cleaned_image = cv2.filter2D(im, -1, sharpen_kernel)

    return cleaned_image


def clean_umbrella(im):
    # Compute the Fourier transform of the input image
    image_fourier = np.fft.fft2(im)

    # using paint we've checked the shifting of the second image
    x_shift, y_shift = 79, 4
    # Initialize a delta filter with the shift
    kernel = np.zeros(im.shape)
    kernel[0, 0] = 0.5
    kernel[y_shift, x_shift] = 0.5

    # Compute the Fourier transform of the filter
    delta_fourier = np.fft.fft2(kernel)

    # Delta parameter to avoid division by zero
    delta = 0.0000001

    # Estimate the original image frequencies
    f_cleaned_image = (np.conj(delta_fourier) /
                                (np.conj(delta_fourier) * delta_fourier + delta)) * image_fourier

    # Compute the inverse Fourier transform to get back to the spatial domain and take the absolute value to form the cleaned image
    cleaned_image = abs(np.fft.ifft2(f_cleaned_image))

    return cleaned_image

def clean_USAflag(im):
    # extract the stars from the image
    roi = im[0:89, 0:141]
    # define a kernel that will be used to calculate the median
    kernel_width = 15
    # pad width for the image to not lose information when filtering with the kernel
    pad_width = kernel_width//2

    fixed_im = np.zeros_like(im) # new image to save the result
    im_padded = np.pad(im, ((0,0), (pad_width, pad_width)), 'reflect')  # add padding to the original image

    for i in range(im.shape[0]):
        for j in range(pad_width, im.shape[1]+pad_width):
            # Extract the neighborhood
            window = im_padded[i, (j):(j + kernel_width)]
            # Compute the median and set the pixel value
            fixed_im[i, j-pad_width] = np.median(window)

    # re-attach the stars
    fixed_im[0:89, 0:141] = roi

    return fixed_im

def clean_house(im):
    image_fft = np.fft.fft2(im)

    # Create a 2D FFT of a 10-pixel wide mask with constant value 0.10
    mask_fft = np.fft.fft2(np.ones((1, 10)) * 0.10, im.shape)

    # Divide the blurred image with the mask
    fixed_im = image_fft / mask_fft

    # Perform inverse FFT and take absolute value to get the clean image
    fixed_im = np.abs(np.fft.ifft2(fixed_im))

    return fixed_im


def clean_bears(im):
    fixed_im = cv2.convertScaleAbs(im, alpha=4.25, beta=-123.25)
    return fixed_im



