ALPHA_LOWER = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']

""" Inference with PB """
# from cap_ocr.predict_testing import *

# predict = Predict(project_name="BEST")
# predict.testing(image_dir=r"cap_ocr/data/hard", limit=None)

""" Inference with tflite """
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageOps


def load_image(filename, input_shape):
    pil_image = Image.open(filename)
    
    im = np.array(pil_image).astype(np.float32)
    im = cv2.resize(im, input_shape)
    im = im.swapaxes(0, 1)
    
    return np.array(im[:, :]) / 255.

def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    return interpreter, input_shape, input_details, output_details

def inference(interpreter, input_details, output_details, data):
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

path = "model.tflite"
interpreter, input_shape, input_details, output_details = load_model(path)
print(input_shape)
print(input_details)
print(output_details)

test_data = load_image("./cap_ocr/data/hard/iulm_000.jpg", (input_shape[1], input_shape[2]))
test_data = np.expand_dims(test_data, axis=0)
print(test_data.shape)

output = inference(interpreter, input_details, output_details, test_data)

print(output)

ans = ""

for i in output[0]:
    if i != 0: ans += ALPHA_LOWER[i-1]
    else: ans += "-"

print(ans)

""" Inference with class """

from predict_captcha import CaptchaPredictor

predictor = CaptchaPredictor("model.tflite")
src = "https://tixcraft.com/ticket/captcha?v=6229be6eac854"
test_data = predictor.data_preprocess(src, (input_shape[1], input_shape[2]))
answer = predictor.inference(test_data)
print(answer)