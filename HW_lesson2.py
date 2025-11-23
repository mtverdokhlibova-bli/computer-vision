import cv2
import numpy as np
# image = cv2.imread('images/photo.jpg')
# print(image.shape)
# image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
# image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 100, 180)
# carnal = np.ones((2, 2), np.uint8)
# image = cv2.dilate(image, carnal, iterations=1)
# image = cv2.erode(image, carnal, iterations=1)

image = cv2.imread('images/photo1.jpg')
print(image.shape)
image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 100, 180)
carnal = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, carnal, iterations=1)
image = cv2.erode(image, carnal, iterations=1)


cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()