import os
import cv2
import numpy as np
import time

PATH = os.getcwd()
INIT_IMG = os.path.join(PATH, 'data/img/init.jpg')
NEXT_IMG = os.path.join(PATH, 'data/img/IMG_20200921_202713_425.jpg')

VIDEO_PATH = os.path.join(PATH, 'data/construction_360/004.mp4')
cap = cv2.VideoCapture(VIDEO_PATH)
ct = 0
time.sleep(1)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    ct += 1
    if ct % 25 == 0:
        frame = cv2.resize(frame, (1024, 512))
        print(ct)
        cv2.imshow('Frame', frame)
        cv2.imwrite(f'004_{ct}.jpg', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
