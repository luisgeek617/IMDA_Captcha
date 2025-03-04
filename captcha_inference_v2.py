#python captcha_inference_v2.py Model/model.json Model/model.h5 input/input01.jpg

import argparse
import cv2
import numpy as np
from keras.models import Sequential, model_from_json
from keras.saving import register_keras_serializable
import matplotlib.pyplot as plt

# Register the custom Sequential class globally
@register_keras_serializable()
class RegisteredSequential(Sequential):
    pass

def load_captcha_model(model_json_path, model_weights_path):
    """
    Loads the trained captcha model.

    Args:
        model_json_path (str): Path to the JSON file containing model architecture.
        model_weights_path (str): Path to the H5 file containing model weights.

    Returns:
        keras.Model: Loaded captcha model.
    """
    # Load the model architecture from the JSON file
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    print('json loaded successfully')

    # Load the model from JSON and apply custom_objects for the registered Sequential class
    custom_objects = {"RegisteredSequential": RegisteredSequential, "Sequential": RegisteredSequential}
    loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)

    # Load the model weights
    loaded_model.load_weights(model_weights_path)

    print('Model successfully loaded')
    return loaded_model

class Captcha(object):
    def __init__(self, model_json_path, model_weights_path):
        """
        Initialize the Captcha class by loading the trained model.
        """
        # Load the model here in the __init__ method
        self.model = load_captcha_model(model_json_path, model_weights_path)
        self.characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # Threshold the image to get a binary image
    def preprocess_image(self, img):
        _, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh_img

    def get_character_boundaries(self, thresh_img):
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

    def segment_characters(self, thresh_img, boundaries, num_chars=5, size=28):
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

    def visualize_segments(self, char_segments):
        fig, axs = plt.subplots(1, len(char_segments), figsize=(15, 5))
        for i, char_segment in enumerate(char_segments):
            axs[i].imshow(char_segment, cmap='gray')
            axs[i].axis('off')
        plt.show()

    def process_and_predict_image(self, image, model, characters):
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

            # Preprocess the ROI
            roi = 255 - roi

            # Predict using the trained model
            pred = model.predict(roi.reshape(1, 28, 28, 1))

            # Get the predicted label
            predicted_char = characters[np.argmax(pred)]
            predicted_chars.append(predicted_char)

        return ''.join(predicted_chars)

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captcha Recognition")
    parser.add_argument('model_json', type=str, help="Path to the model JSON file.")
    parser.add_argument('model_weights', type=str, help="Path to the model weights file.")
    parser.add_argument('image_path', type=str, help="Path to the input captcha image.")
    args = parser.parse_args()

    # Load and process image
    image = cv2.imread(args.image_path)

    # Initialize Captcha class and process the image
    captcha_recognizer = Captcha(args.model_json, args.model_weights)
    captcha_result = captcha_recognizer.process_and_predict_image(image, captcha_recognizer.model, captcha_recognizer.characters)

    print(f"Predicted Captcha: {captcha_result}")
