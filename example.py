""" 
    Inference with CaptchaPredictor
    Support image with file  url
"""
from predict_captcha import CaptchaPredictor

# Initialize model
predictor = CaptchaPredictor("model.tflite")

# Inference with singel image
src = "test.jpg"
test_data = predictor.data_preprocess(src, (predictor.input_shape[1], predictor.input_shape[2]))
answer = predictor.inference(test_data)
print("Inference with singel image")
print("Src: " + src + "\nPrediction: " + answer + "\n")

# Evaluate with dir
print("Evaluate with dir")
i_acc, w_acc = predictor.evaluate_with_dir("val_data")
print(f"Image Accuracy: {i_acc}%")
print(f"Word Accuracy:  {w_acc}%")
