# Shir Grinblat, 308570209
# Amir Kedem, 066560475

import cv2
import numpy as np

# input: radius (integer), std deviation used for gaussian spatial distance (float)
# output: gaussian spatial mask in float64 format
def create_spatial_mask(radius, stdSpatial):
    # create a 2D array of distance between the center
    x, y = np.meshgrid(np.linspace(-radius, radius, num=(radius*2+1), dtype='float64'),
                        np.linspace(-radius, radius, num=(radius*2+1), dtype='float64'))
    # calculate L2 distance
    d = np.sqrt(x**2 + y**2)
    # calculate the mask of gaussian filter by spatial distance
    mask = np.exp(-(d / (2*stdSpatial**2)))

    return mask.astype('float64')

# input: the window to be corrected (array), std of intensity, the intensity value of the center pixel
# output: masked window (array)
def create_intensity_mask(window, stdIntensity, intensity):
    intensity_mask = np.exp(-((window-intensity)**2)/(2*stdIntensity**2))
    return intensity_mask

# input: the input image to be corrected (array), radius of the kernel, std of intensity and of spatial
# output: corrected image (array unit8)
def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    corrected_im = np.zeros(shape=im.shape, dtype='float64') # make a new image of the same shape since we need to copy results
    im = im.astype(np.float64) # modify type so we could multiply without losing information
    im_padded = np.pad(im, radius, 'reflect') # add padding to the original image
    spatial_mask = create_spatial_mask(radius, stdSpatial) # create spatial mask, will remain the same on all calculations

    # iterate on the entire image
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            window = im_padded[i:i+2*radius+1, j:j+2*radius+1] # define the current window the kernel is reviewing
            intense_mask = create_intensity_mask(window, stdIntensity, im[i, j]) # calculate intensity mask
            guassian_mask = np.multiply(spatial_mask, intense_mask) # convolution of both masks
            # set the result as new pixel value of the corrected image and normalize the value
            corrected_im[i, j] = np.sum(np.multiply(window, guassian_mask)/np.sum(guassian_mask))
    return corrected_im.astype('uint8') # modify back to uint8

# start of program
lst = ['balls.jpg', 'NoisyGrayImage.png', 'taj.jpg']

# Loop over each file in the directory
for index, image_path in enumerate(lst):
    # Check if it's a file and ends with .png or .jpg
    match index:
        case 0:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            clear_image_b = clean_Gaussian_noise_bilateral(image, 3, 5, 15)
            cv2.imwrite('balls_fixed.jpg', clear_image_b)
        case 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            clear_image_b = clean_Gaussian_noise_bilateral(image, 8, 10, 100)
            cv2.imwrite('NoisyGrayImage_fixed.jpg', clear_image_b)
        case 2:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            clear_image_b = clean_Gaussian_noise_bilateral(image, 8, 150, 30)
            cv2.imwrite('taj_fixed.jpg', clear_image_b)
