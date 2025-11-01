# Shir Grinblat, 308570209
# Amir Kedem, 066560475

import numpy as np
import cv2


# section a

# load the broken image
broken = cv2.imread('broken.jpg', cv2.IMREAD_GRAYSCALE)

# use median filter to remove salt and pepper noise
median = cv2.medianBlur(broken, 3)

# appy bilateral filter
bilateral = cv2.bilateralFilter(median, 15, 25, 90, borderType=cv2.BORDER_WRAP)

# show both images side by side
compare = np.concatenate((broken, bilateral), axis=1)
cv2.imshow('image', compare)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('broken_fixed.jpg', bilateral)

# section b

# load all noisy images
data = np.load('noised_images.npy')

# run on all pixels and choose the median for each
medianImage = np.median(data, axis=0).astype(np.uint8)

# show corrected image
compare = np.concatenate((broken, medianImage), axis=1)
cv2.imshow('compare', compare)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('noisy_images_fixed.jpg', medianImage)
