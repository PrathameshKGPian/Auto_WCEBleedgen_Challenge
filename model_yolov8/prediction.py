from ultralytics import YOLO

import numpy as np


model = YOLO(r"C:\Users\HP\PycharmProjects\NN and DL\runs\classify\train3\weights\last.pt")  # load a custom model

results = model(r"C:\Users\HP\OneDrive\Desktop\dataset\WCEBleedGen\Auto-WCEBleedGen Challenge Test Dataset\Test Data")  # predict on an image

# names_dict = results[0].names
probs = results.probs.data.tolist()
# print(names_dict)
print(probs)
# y_pred = results.predict
# y_pred = np.argmax(y_pred, axis=1)
# print(y_pred[np.argmax(probs)])
