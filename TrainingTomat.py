import cv2
import numpy as np
import os
from skimage.feature.texture import graycomatrix, graycoprops
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

# Function to extract texture features using GLCM
def extract_texture_features(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract HSV components
    hue, saturation, value = cv2.split(hsv)

    # Calculate GLCM (Gray-Level Co-occurrence Matrix) on the Value channel
    glcm = graycomatrix(value, [1], [0], symmetric=True, normed=True)
    
    # Calculate texture properties from GLCM
    dissimilarity = graycoprops(glcm, prop='dissimilarity')[0, 0]
    correlation = graycoprops(glcm, prop='correlation')[0, 0]
    homogeneity = graycoprops(glcm, prop='homogeneity')[0, 0]
    contrast = graycoprops(glcm, prop='contrast')[0, 0]
    asm = graycoprops(glcm, prop='ASM')[0, 0]
    energy = graycoprops(glcm, prop='energy')[0, 0]

    # Additional texture features
    # Additional texture features
    contours, _ = cv2.findContours(value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and len(contours[0]) >= 5:
        metric = cv2.arcLength(contours[0], True)
        (x, y), (MA, ma), angle = cv2.fitEllipse(contours[0])
        eccentricity = np.sqrt(1 - (ma / MA)**2)
    else:
        metric, (x, y), (MA, ma), angle, eccentricity = 0, (0, 0), (0, 0), 0, 0


    return [hue.mean(), saturation.mean(), value.mean(), dissimilarity, correlation, homogeneity, contrast, asm, energy, metric, eccentricity]

# List of folder paths for training data
train_base_folder = r"D:/Kuliah/Project/Python/deteksi_tomat/dateset_tomat"
train_subfolders = ["ada tomat", "tidak ada tomat"]

train_feature_data = []

# Collect training data
for folder_name in train_subfolders:
    folder_path = os.path.join(train_base_folder, folder_name)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".png", ".JPG", ".PNG")):
                image_path = os.path.join(folder_path, filename)
                texture_features = extract_texture_features(image_path)
                train_feature_data.append([filename, folder_name] + texture_features)

# Create a DataFrame for training data
train_data = pd.DataFrame(train_feature_data, columns=['Nama Image', 'Kategori', 'Hue', 'Saturation', 'Value', 'Dissimilarity', 'Correlation', 'Homogeneity', 'Contrast', 'ASM', 'Energy', 'Metric', 'Eccentricity'])

# Read data from Excel
data = pd.read_excel('ekstrasi-data-tomat.xlsx')

# Combine training data with existing data
combined_data = pd.concat([data, train_data], ignore_index=True)

# Sort the combined DataFrame by 'Nama Image'
combined_data = combined_data.sort_values(by='Nama Image')

# Normalize features
scaler = MinMaxScaler()
combined_data[['Hue', 'Saturation', 'Value', 'Dissimilarity', 'Correlation', 'Homogeneity', 'Contrast', 'ASM', 'Energy', 'Metric', 'Eccentricity']] = scaler.fit_transform(combined_data[['Hue', 'Saturation', 'Value', 'Dissimilarity', 'Correlation', 'Homogeneity', 'Contrast', 'ASM', 'Energy', 'Metric', 'Eccentricity']])

# Split data into features (X) and target labels (y)
X = combined_data[['Hue', 'Saturation', 'Value', 'Dissimilarity', 'Correlation', 'Homogeneity', 'Contrast', 'ASM', 'Energy', 'Metric', 'Eccentricity']]
y = combined_data['Kategori']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=45)

# Train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=45)
model.fit(X_train, y_train)

# Save the trained model to a file (optional)
joblib.dump(model, 'trained_model.joblib')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_result)

# Combine test features and predictions
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
test_results.reset_index(drop=True, inplace=True)

# Save the test results to Excel
test_results.to_excel('test_results.xlsx', index=False)
