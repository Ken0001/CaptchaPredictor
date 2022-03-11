import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import requests as req
from io import BytesIO
import time
import os

class CaptchaPredictor():
    ALPHA_LOWER = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 
                   'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 
                   's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']

    def data_preprocess(self, img_src, input_shape):
        if "http" in img_src:
            print("Requesting...")
            start = time.time()
            response = req.get(img_src)
            image = Image.open(BytesIO(response.content))
            end = time.time()
            print("Requesting time: ", end-start)
        else: 
            image = Image.open(img_src)
        im = np.array(image).astype(np.float32)
        im = cv2.resize(im, input_shape)
        im = im.swapaxes(0, 1)
        im = np.expand_dims(im, axis=0)
        image = np.array(im[:, :]) / 255.
        return image

    def inference(self, image):
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        ans = ""
        for i in output_data[0]:
            if i != 27: ans += self.ALPHA_LOWER[i-1]
            else: ans += "-"

        return ans

    def evaluate_with_dir(self, path):
        correct_sample = 0
        correct_word = 0
        total_sample = 0
        
        for file in os.listdir(path):
            total_sample += 1
            label = file.split('_')[0]
            img_path = os.path.join(path, file)
            img_data = self.data_preprocess(img_path, (self.input_shape[1], self.input_shape[2]))
            pred = self.inference(img_data)

            print(total_sample, "["+str(label)+"->"+str(pred)+"]", pred == label, file)

            # Caculate accuracy for image and word
            if pred == label: 
                correct_sample += 1
                correct_word += 4
            else: 
                for word in pred:
                    if word in label:
                        correct_word += 1
            
        return round(100*(correct_sample/total_sample), 4), round(100*(correct_word/(total_sample*4)), 4)