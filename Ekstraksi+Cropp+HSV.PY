import cv2
import pandas as pd
import numpy as np
import os

def detect_and_crop_tomato(image, scale_factor=0.5):
    """Detect tomato in the image using color-based detection and crop it."""
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red, orange, and green colors in HSV
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

def process_image(image_path, label, output_folder='cropped_images'):
    """Process a single image to get average HSV and save cropped image."""
    image = cv2.imread(image_path)
    tomato_image, crop_coords = detect_and_crop_tomato(image)

    if tomato_image is not None:
        # Convert the cropped image to HSV
        hsv_tomato = cv2.cvtColor(tomato_image, cv2.COLOR_BGR2HSV)

        # Save the cropped image to the "cropped_images" folder
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, hsv_tomato)

        # Calculate average HSV values
        average_hsv_per_row = np.average(hsv_tomato, axis=0)
        average_hsv = np.average(average_hsv_per_row, axis=0)
        average_hsv = np.uint8(average_hsv)
        average_hsv = average_hsv.tolist()

        return crop_coords, average_hsv, label
    else:
        return None, None

# ... (fungsi detect_and_crop_tomato dan contoh penggunaan tetap sama)



# Define the folder path
folder_path = 'data_set_tomat/tomat_merah/'

# Ensure that the output folder exists
output_folder = 'cropped_images'
os.makedirs(output_folder, exist_ok=True)

# List all images in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Process each image
results = []
label = 'merah'
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    crop_coords, average_hsv, label_result = process_image(image_path, label, output_folder)

    if crop_coords is not None:
        results.append([image_file] + average_hsv + [label_result])

# Save the results to Excel
df = pd.DataFrame(results, columns=['Image', 'Hue', 'Saturation', 'Value', 'Label'])
df.to_excel('excel/average_tomato_hsv_merah.xlsx', index=False)
