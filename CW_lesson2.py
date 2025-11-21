import cv2
import numpy as np # as створення псевдоніму шоб менше писати
image = cv2.imread('images/1.jpg')
print(image.shape)
#image = cv2.resize(image, (800, 800))
#image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
image = cv2.Canny(image, 100, 100)
carnal = np.ones((5, 5), np.uint8)
#image = cv2.dilate(image, carnal, iterations=1) #розширює світлі облвсті на фото
#image = cv2.erode(image, carnal, iterations=1) #розширює темні зони
cv2.imwrite('1.jpg', image)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()