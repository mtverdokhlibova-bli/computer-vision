# import os
# import cv2
# import time
# from ultralytics import YOLO
#
# cap = cv2.VideoCapture('video/video.mp4')
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
# # model = YOLO("yolov8n-cls.pt")
# # CONF_THRESHOLD = 0.4
# #
# # parking_place = 0
# # parking_CLASS_ID = 0
# #
# # result = model(frame, conf = CONF_THRESHOLD, verbose=False)
# #
# # for r in result:
# #     boxes = r.boxes
# #     if boxes is None:
# #         continue
# #
# #     for box in boxes:
# #         cls = int(box.cls[0])
# #         conf = float(box.conf[0])
# #
# #         x1, y1, x2, y2 = map(int, box.xyxy[0])
# #
# #         if cls ==parking_CLASS_ID:
# #             parking_place += 1
# #
# #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
#
#
#
#
# cv2.imshow('frame',frame)
# cap.release()


import cv2

cap = cv2.VideoCapture('video/video.mp4')

if not cap.isOpened():
    print("Помилка: Не вдалося відкрити відео.")
    exit()


while True:
    ret, frame = cap.read()


    if not ret:
        break
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    cv2.imshow('Video Player', frame)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
