import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

final_map0 = np.load('./final_maps/final_map_0.npy')
final_map2 = np.load('./final_maps/final_map_2.npy')
plt.axis('off')

cv.imwrite("final_map0.png", final_map0)
cv.imwrite("final_map2.png", final_map2)
# plt.imshow(final_map0)
# plt.savefig("final_map0.png", bbox_inches='tight')
# plt.imshow(final_map2)
# plt.savefig("final_map2.png", bbox_inches='tight')
# plt.show()