# Overview

This project implements a Captcha recognition system using a trained Convolutional Neural Network (CNN) built with Keras. It uses image processing techniques with OpenCV and NumPy for character segmentation and preprocessing, and the model predicts alphanumeric captchas.

Assumptions: 
- the number of characters remains the same each time
- the font and spacing is the same each time
- the background and foreground colors and texture, remain largely the same
- there is no skew in the structure of the characters.
- the captcha generator, creates strictly 5-character captchas, and each of the characters is either an upper-case character (A-Z) or a numeral (0-9).

# Project Structure
├── captcha.py              # Captcha class and processing methods
├── Model
│   ├── captcha_model.json      # Model architecture
│   ├── captcha_model.h5        # Trained model weights
├── input
│   ├── input01.jpg             # Example captcha image
│   ├── input02.jpg             # Another example captcha image
├── output
│   ├── output01.jpg             # Ground Truth of the corresponding input image, e.g, input01.jpg
│   ├── output02.jpg             # Ground Truth of the corresponding input image, e.g, input02.jpg

# Requirements

Python 3.5+
Keras
TensorFlow
OpenCV
NumPy
Matplotlib

you may install the relavent package using the below command
python3 -m pip install --upgrade pip
pip install opencv-python
sudo apt-get update 
sudo apt-get install libgl1-mesa-glx
sudo apt-get install -y libgl1 libglib2.0-0
pip install keras
pip install tensorflow


# Usage for simplicity, you may directly run 
python captcha_inference_main.py Model/model.json Model/model.h5 input/input02.jpg
the following arguments are required: model_json, model_weights, image_path

# Key Methods

# preprocess_image(img)
Converts an image to a binary thresholded image for better character segmentation.

# get_character_boundaries(thresh_img)
Finds the horizontal and vertical boundaries of each character.

# segment_characters(thresh_img, boundaries, num_chars=5, size=28)
Segments individual characters from the binary image and normalizes them to a fixed size.

# process_and_predict_image(image, model, characters)
Processes an input image and predicts the captcha text.

# fix_zeros_and_os_with_size(roi, predicted_string)
Distinguishes between the characters '0' and 'O' based on the size and proportion of white pixels.

# Notes
Characters recognized: 0-9, A-Z
Ensure consistent image size and quality for accurate predictions.
