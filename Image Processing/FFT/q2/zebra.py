# Shir Grinblat, 308570209
# Amir Kedem, 066560475

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Your code goes here

# Original image
plt.figure(figsize=(10,10))
plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

# Transfrom to fourier and present the specturm
image_fft = np.fft.fft2(image)
image_fft_shifted = np.fft.fftshift(image_fft)
image_fft_shifted_plot = np.log(1+np.absolute(image_fft_shifted)) # take magnitude in abs, cannot show complex

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(image_fft_shifted_plot, cmap='gray')

# Implement zero padding to make the image twice the size
image_zero_pad = np.pad(image_fft_shifted, ((image.shape[0]//2, image.shape[0]//2), (image.shape[1]//2, image.shape[1]//2)), 'constant', constant_values=0)
image_zero_pad_plot = np.log(1 + np.abs(image_zero_pad))

# Plot the new frequency domain
plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(image_zero_pad_plot, cmap='gray')

# Restore
image_scaled_zero_pad = np.abs(ifft2(ifftshift(image_zero_pad)))
brightness_factor = 4
image_scaled_zero_pad = np.clip(image_scaled_zero_pad * 4, 0, 255)

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(image_scaled_zero_pad, cmap='gray')

four_copy_fft = np.fft.fft2(image)
four_copy_fft = np.fft.fftshift(four_copy_fft)
newH = image.shape[0]*2
newW = image.shape[1]*2
four_copy_image = np.zeros((newH, newW), dtype=complex)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        four_copy_image[i*2, j*2] = four_copy_fft[i, j]

four_copy_fft_plot = np.log(1+np.absolute(four_copy_image))

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(four_copy_fft_plot, cmap='gray')

zebra_scaled = np.abs(ifft2(ifftshift(four_copy_image)))

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(zebra_scaled, cmap='gray')
plt.savefig('zebra_scaled.png')
plt.show()