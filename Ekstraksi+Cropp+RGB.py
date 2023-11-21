import cv2
import pandas as pd
import numpy as np
import os

def detect_and_crop_tomato(image, scale_factor=0.5):
    """Detect tomato in the image using color-based detection and crop it."""
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red, orange, and green colors
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_orange = np.array([11, 100, 100])
    upper_orange = np.array([25, 255, 255])

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Threshold the image to get regions corresponding to red, orange, and green colors
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Combine the masks to get a binary mask for tomato regions
    mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_orange, mask_green))

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour corresponds to the tomato
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Adjust the bounding box dimensions using the scale_factor
        x_offset = int(w * (1 - scale_factor) / 2)
        y_offset = int(h * (1 - scale_factor) / 2)

        x = max(0, x + x_offset)
        y = max(0, y + y_offset)
        w = min(image.shape[1] - x, int(w * scale_factor))
        h = min(image.shape[0] - y, int(h * scale_factor))

        tomato_image = image[y:y+h, x:x+w]
        return tomato_image, (x, y, w, h)
    else:
        return None, None

def process_image(image_path, output_folder='cropped_images'):
    image = cv2.imread(image_path)
    tomato_image, crop_coords = detect_and_crop_tomato(image)

    if tomato_image is not None:
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, tomato_image)

        average_color_per_row = np.average(tomato_image, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        average_color = np.uint8(average_color)
        average_rgb = average_color.tolist()

        # Determine label based on average RGB values
        if average_rgb[0] > average_rgb[1] and average_rgb[0] > average_rgb[2]:
            label = 'red'
        elif average_rgb[1] > average_rgb[0] and average_rgb[1] > average_rgb[2]:
            label = 'green'
        else:
            label = 'orange'

        return crop_coords, average_rgb, label
    else:
        return None, None

# ... (unchanged)

# Process each image
results = []
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    crop_coords, average_rgb, label_result = process_image(image_path, output_folder)

    if crop_coords is not None:
        results.append([image_file] + average_rgb + [label_result])

# Save the results to Excel
df = pd.DataFrame(results, columns=['Image', 'Red', 'Green', 'Blue', 'Label'])
df.to_excel('excel/average_tomato_rgb_labels.xlsx', index=False)
