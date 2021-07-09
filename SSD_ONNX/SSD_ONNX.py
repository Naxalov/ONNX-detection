import numpy as np
from PIL import Image
import onnxruntime as rt
import cv2
import matplotlib.pyplot as plt

def preprocess(img_path):
    input_shape = (1, 3, 1200, 1200)
    img = Image.open(img_path)
    img = img.resize((1200, 1200), Image.BILINEAR)
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data



img = cv2.imread('worker.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

sess = rt.InferenceSession('model.onnx')
input_name = sess.get_inputs()[0].name
input_data = preprocess('worker.jpeg')
pred_onx = sess.run(None, {input_name: input_data})
print(pred_onx[0].shape)
print('NAME')
print(input_name)


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
	
plt.imshow(img)
plt.show()