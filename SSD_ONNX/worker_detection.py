import os
import cv2
import numpy as np
from PIL import Image
import onnxruntime as rt
import matplotlib.pyplot as plt

def preprocess(img):
    input_shape = (1, 3, 1200, 1200)
    # img = Image.open(img_path)

    # img = img.resize((1200, 1200), Image.BILINEAR)
    img = cv2.resize(img,(1200,1200))
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data



PATH = os.getcwd()

VIDEO_PATH = os.path.join(PATH, 'CASE_1.mov')
cap = cv2.VideoCapture(VIDEO_PATH)
sess = rt.InferenceSession('model.onnx')
input_name = sess.get_inputs()[0].name
while True:
    ret, frame = cap.read()
    if frame is None:
        break
 
    input_data = preprocess(frame)
    pred_onx = sess.run(None, {input_name: input_data})
    img = frame
    rows, cols, channels = img.shape
    boxes = pred_onx[0][0]
    score = pred_onx[2][0]
    for i in range(100):
        if score[i]>.4:
            detection = boxes[i] 
            left = detection[0] * cols
            top = detection[1] * rows
            right = detection[2] * cols
            bottom = detection[3] * rows
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
        
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
