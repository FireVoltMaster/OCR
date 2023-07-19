import keras_ocr
import matplotlib.pyplot as plt
keras_pipeline = keras_ocr.pipeline.Pipeline()

img = [
    'C://Users//pc//Documents//My Received Files//2-5-KsjJg//1.png'
]

pred = keras_pipeline.recognize(img)

print(pred)

pred_img = pred[1]
for text, box in pred_img:
    print(text)

