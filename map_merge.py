import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# final_map = np.load('./final_maps/final_map_2.npy')
# # plt.imshow(final_map)
# # plt.show()
#
# img = cv.imread('Dataset0.png')
# img_float32 = np.float32(img)
# gray = cv.cvtColor(img_float32,cv.COLOR_BGR2GRAY)
# sift = cv.SIFT_create()
# image8bit = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
# kp = sift.detect(image8bit,None)
# img = cv.drawKeypoints(image8bit,kp,img)
# cv.imwrite('sift_keypoints.jpg',img)

#Method 4: register_translation from skimage.feature (already covered in previous video)
from skimage import io
from image_registration import cross_correlation_shifts

image = io.imread("Dataset0.png")
offset_image = io.imread("Dataset2.png")
# offset image translated by (-17.45, 18.75) in y and x


from skimage.feature import register_translation
shifted, error, diffphase = register_translation(image, offset_image, 100)
xoff = -shifted[1]
yoff = -shifted[0]


print("Offset image was translated by: 18.75, -17.45")
print("Pixels shifted by: ", xoff, yoff)


from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()

# print(5)