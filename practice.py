import cv2
import numpy as np

img = cv2.imread('images/candy2.jpg')
img_copy = img.copy()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([139, 116, 94])
upper_red = np.array([179, 255, 255])

lower_purple = np.array([97, 0, 0])
upper_purple = np.array([150, 255, 255])

lower_orange = np.array([0, 127, 16])
upper_orange = np.array([19, 255, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
mask = cv2.bitwise_or(mask_red, mask_purple)
final_mask = cv2.bitwise_or(mask, mask_orange)
result = cv2.bitwise_and(img, img, mask=final_mask)

contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
        if cv2.contourArea(cnt) > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            candy = 'candy'
            text = x, y - 10 if y - 10 > 20 else y + 20
            cv2.putText(img_copy, candy, text, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)


cv2.imshow('image', result)
cv2.imshow('image2', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()