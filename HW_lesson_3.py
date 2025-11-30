import cv2
import numpy as np

img = cv2.imread('images/2.jpg')
print(img.shape)
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
cv2.rectangle(img, (230, 180), (310, 260), (0, 220, 0), 2)
cv2.rectangle(img, (120, 195), (180, 245), (0, 220, 0), 2)
cv2.rectangle(img, (340, 50), (560, 300), (0, 220, 0), 2)
cv2.putText(img, "deer 1", (110, 270), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 1)
cv2.putText(img, "deer 2", (230, 290), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 1)
cv2.putText(img, "Tverdokhlibova Masha", (320, 330), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 1)

cv2.imshow('photochka', img)
cv2.waitKey(0)
cv2.destroyAllWindows()