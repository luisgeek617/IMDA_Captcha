# this script aim to test all the provided cases, the input images are in the input folder, where the ground truth can be found in the output folder

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#from tensorflow.keras.models import Sequential, load_model
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#from tensorflow.keras.utils import to_categorical
#from sklearn.model_selection import train_test_split

INPUT_DIR = "input/"
OUTPUT_DIR = "output/"
IMG_HEIGHT, IMG_WIDTH = 30, 60
# Updated character set (only uppercase letters and digits)
characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Helper functions
def load_images_and_labels():
    images, labels = [], []
    for filename in os.listdir(INPUT_DIR):
        if filename.startswith("input") and filename.endswith(".jpg"):
            label_filename = filename.replace("input", "output").replace(".jpg", ".txt")
            if not os.path.exists(os.path.join(OUTPUT_DIR, label_filename)):
                continue
            img = cv2.imread(os.path.join(INPUT_DIR, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)

            with open(os.path.join(OUTPUT_DIR, label_filename), 'r') as f:
                label = f.read().strip()
            labels.append(list(label))
    return np.array(images), np.array(labels)

# Threshold the image to get a binary image
def preprocess_image(img):
    _, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh_img

def get_character_boundaries(thresh_img):
    # Get the horizontal projection (sum of pixel values along each column)
    projection_x = np.sum(thresh_img, axis=0)

    # Threshold the projection to find regions that likely correspond to characters
    min_char_width = 10  # Minimum character width
    min_space_width = 5  # Minimum space width
    boundaries = []
    current_start = None

    # Find the horizontal character boundaries (left-right edges)
    for i, value in enumerate(projection_x):
        if value > 0:  # Character region
            if current_start is None:
                current_start = i  # Found the start of a character
        else:  # Empty space region
            if current_start is not None:
                boundaries.append((current_start, i))  # Character ends at this position
                current_start = None

    # Handle edge case if the last character doesn't have space after it
    if current_start is not None:
        boundaries.append((current_start, len(projection_x)))

    # Now we need to find vertical boundaries for each character (top-bottom edges)
    char_boundaries = []
    for x_start, x_end in boundaries:
        # Sum the projection along the rows (vertical sum) to find where characters start and end vertically
        projection_y = np.sum(thresh_img[:, x_start:x_end], axis=1)

        # Threshold the projection to get the top and bottom of the character
        min_height = 10  # Minimum height of a character
        top, bottom = None, None

        for i, value in enumerate(projection_y):
            if value > 0 and top is None:
                top = i  # Found the top of the character
            if value == 0 and top is not None and bottom is None:
                bottom = i  # Found the bottom of the character
                break

        if top is None: top = 0  # Handle case if no character is found
        if bottom is None: bottom = thresh_img.shape[0]  # Handle case if no character is found

        char_boundaries.append((x_start, x_end, top, bottom))

    return char_boundaries

def segment_characters(thresh_img, boundaries, num_chars=5, size=28):
    char_segments = []

    for i in range(num_chars):
        # Get the bounding box for the character from the boundaries
        x_start, x_end, y_start, y_end = boundaries[i]

        # Crop the image to get only the region containing the character
        char_region = thresh_img[y_start:y_end, x_start:x_end]

        # Find connected components in this cropped region
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(char_region)

        # Exclude the background (label 0), which is not a character
        for label in range(1, num_labels):
            # Get the centroid of each connected component
            cx, cy = centroids[label]
            # Get the bounding box of the component
            x, y, w, h, area = stats[label]

            # Define the top-left and bottom-right coordinates of the 50x50 window centered around the centroid
            top_left_x = int(cx - size // 2)
            top_left_y = int(cy - size // 2)
            bottom_right_x = int(cx + size // 2)
            bottom_right_y = int(cy + size // 2)

            # Clip the bounding box to avoid out-of-bounds access
            top_left_x = max(0, top_left_x)
            top_left_y = max(0, top_left_y)
            bottom_right_x = min(char_region.shape[1], bottom_right_x)
            bottom_right_y = min(char_region.shape[0], bottom_right_y)

            # Extract the region around the centroid
            char_segment = char_region[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # If the segment is smaller than 50x50, pad it with zeros, ensuring the character is centered
            padded_segment = np.zeros((size, size), dtype=np.uint8)

            # Calculate how much padding to add to each side
            pad_top = max(0, (size - char_segment.shape[0]) // 2)
            pad_bottom = max(0, size - char_segment.shape[0] - pad_top)
            pad_left = max(0, (size - char_segment.shape[1]) // 2)
            pad_right = max(0, size - char_segment.shape[1] - pad_left)

            # Place the character at the center of the padded region
            padded_segment[pad_top:pad_top+char_segment.shape[0], pad_left:pad_left+char_segment.shape[1]] = char_segment
            char_segments.append(padded_segment)

    return [255 - x for x in char_segments]

def visualize_segments(char_segments):
    fig, axs = plt.subplots(1, len(char_segments), figsize=(15, 5))
    for i, char_segment in enumerate(char_segments):
        axs[i].imshow(char_segment, cmap='gray')
        axs[i].axis('off')
    plt.show()

# Load the images and labels
images, labels = load_images_and_labels()

img = images[5]
thresh_img = preprocess_image(img)
# Step 2: Get the character boundaries based on projection analysis
boundaries = get_character_boundaries(thresh_img)
# Step 3: Segment the image into 5 characters
char_segments = segment_characters(thresh_img, boundaries)
# Step 4: Visualize the segmented characters
visualize_segments(char_segments)

from keras.models import Sequential, model_from_json
from keras.saving import register_keras_serializable

# Register the Sequential class
@register_keras_serializable()
class RegisteredSequential(Sequential):
    pass

# Load model architecture from JSON
json_file = open('Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Pass custom objects to model_from_json
loaded_model = model_from_json(loaded_model_json, custom_objects={"Sequential": RegisteredSequential})

# Load model weights
loaded_model.load_weights('Model/model.h5')

model = loaded_model

print('Model successfully loaded')

import cv2
import numpy as np
from matplotlib import pyplot as plt



def process_and_predict_image(image, model):
    # If the image is grayscale, convert it to RGB
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)  # Repeat along the channel axis

    # Resize image for better space detection
    height, width, depth = image.shape
    image = cv2.resize(image, dsize=(width*5, height*4), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Dilation for space detection
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # Gaussian Blur
    gsblur = cv2.GaussianBlur(img_dilation, (5, 5), 0)

    # Find contours
    ctrs, _ = cv2.findContours(gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Initialize list for predictions
    predicted_chars = []
    for ctr in sorted_ctrs:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Extract Region of Interest (ROI)
        roi = image[y-10:y+h+10, x-10:x+w+10]
        roi = cv2.resize(roi, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Preprocess ROI for prediction
        roi = np.array(roi) / 255.0
        roi = 1 - roi
        roi = roi.reshape(1, 28, 28, 1)

        # Predict the character using the model
        prediction = model.predict(roi)
        pred = np.argmax(prediction, axis=-1)

        # If the predicted class is out of range, take the second highest prediction
        if pred[0] < len(characters):  # If the predicted class is valid
            predicted_chars.append(characters[pred[0]])
        else:
            # Get the second-highest score if out of range
            second_highest_index = np.argsort(prediction[0])[-2]
            predicted_chars.append(characters[second_highest_index])

    # Output the predicted string
    predicted_string = ''.join(predicted_chars)
    return predicted_string

import cv2
import numpy as np

# Function to check the size difference between 'O' and '0'
def fix_zeros_and_os_with_size(roi, predicted_string):
    # Count the number of white pixels in the region of interest (ROI)
    white_pixels = np.sum(roi == 0)
    total_pixels = roi.size  # Total number of pixels in the ROI

    # Calculate the proportion of white pixels
    white_pixel_ratio = white_pixels / total_pixels

    # If the ratio of white pixels is high, it's more likely to be an 'O'
    # You can adjust this threshold based on your observations
    if white_pixels <39:  # Adjust the threshold based on your dataset
        predicted_string = predicted_string.replace('O', '0')  # Change 'O' to '0'

    return predicted_string

# Example usage with any image from char_segments[i]
true_count = 0
false_count = 0
total_count = 0

for k in range(len(images)):
    img = images[k]
    thresh_img = preprocess_image(img)

    # Step 2: Get the character boundaries based on projection analysis
    boundaries = get_character_boundaries(thresh_img)

    # Step 3: Segment the image into 5 characters
    char_segments = segment_characters(thresh_img, boundaries)

    for i in range(5):
        image = char_segments[i]
        predicted_string = process_and_predict_image(image, model)

        # Post-process based on size and aspect ratio
        if (predicted_string == 'O') or (predicted_string == 'O'):
          predicted_string = fix_zeros_and_os_with_size(image, predicted_string)

        # Compare fixed predicted string with the label
        label = labels[k][i]  # Assuming labels[k][i] is the correct label for comparison
        if predicted_string != label:
            print(k,i)
            print(f"Predicted String: {predicted_string}")
            print(f"Actual Label: {label}")
            print(f"Match: {predicted_string == label}")

        # Increment the appropriate counter
        if predicted_string == label:
            true_count += 1
        else:
            false_count += 1
        total_count += 1

# Calculate percentages
true_percentage = (true_count / total_count) * 100 if total_count > 0 else 0
false_percentage = (false_count / total_count) * 100 if total_count > 0 else 0

# Display statistics
print(f"\nTotal Predictions: {total_count}")
print(f"Correct Predictions (True): {true_count} ({true_percentage:.2f}%)")
print(f"Incorrect Predictions (False): {false_count} ({false_percentage:.2f}%)")