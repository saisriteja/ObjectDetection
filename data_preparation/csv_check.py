import pandas as pd

data = pd.read_csv('/data/tmp_teja/datacv/final/ObjectDetection/clipNN_output/clipNN_pruned.csv')

import cv2

img  = cv2.imread(data['img_path'][0])

print(img.shape)

x_min = data['x_min'][0]
y_min = data['y_min'][0]
width = data['width'][0]
height = data['height'][0]

x_min = int(x_min)
y_min = int(y_min)
width = int(width)
height = int(height)

print(x_min, y_min, width, height)

# draw a box around the object
cv2.rectangle(img, (x_min, y_min), (x_min+width, y_min+height), (0, 255, 0), 2)
cv2.imshow('image', img)
cv2.waitKey(0)