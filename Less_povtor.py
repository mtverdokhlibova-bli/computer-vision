import cv2

img = cv2.imread('images/1234.jpg')
scale = 4
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
img_copy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 2)
img = cv2.equalizeHist(img)
img_edges = cv2.Canny(img, 50, 50)
contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rectangle = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 70:
        rectangle += 1
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(f'Кількість знайдених магнітів: {rectangle}')

cv2.imshow('image', img)
cv2.imshow('image2', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()