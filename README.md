# CaptchaPredictor

## Data

## Model
1. Model(.pb) trained by using [CaptchaTrainer](https://github.com/kerlomz/captcha_trainer.git).
2. Convert .pd file to tensorflow lite.

## Usage

1. ```data_preprocess()``` Load image from path or url and convert to numpy array.
2. ```inference()``` Inference on single picture with numpy array format.
3. ```evaluate_with_dir()``` Evaluate model for a image dataset.
