import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

final_map = np.load('./final_maps/final_map_2.npy')
# plt.imshow(final_map)
# plt.show()

img = cv.imread('home.jpg')
gray = cv.cvtColor(final_map,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img = cv.drawKeypoints(gray,kp,final_map)
cv.imwrite('sift_keypoints.jpg',final_map)


# print(5)