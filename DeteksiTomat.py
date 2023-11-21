import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from skimage import color, feature, transform
import os

# Fungsi untuk mengekstrak fitur dari gambar
def extract_features(image_path):
    # Membaca gambar dan mengonversi ke skala abu-abu
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Menghitung histogram warna
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Menggabungkan fitur
    features = np.concatenate([gray.ravel(), hist.ravel()])
    
    return features

# Fungsi untuk membaca data dari folder dan mengekstrak fitur
def load_data_from_folder(folder_path):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            label = filename.split('_')[0].lower()  # Menggunakan label dari nama file

            # Mengekstrak fitur dari gambar
            features = extract_features(image_path)

            # Menambahkan data dan label ke list
            data.append(features)
            labels.append(label)

    return np.array(data), np.array(labels)

# Fungsi untuk melakukan label encoding pada target
def encode_labels(labels):
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    return encoded_labels, le.classes_

# Fungsi untuk melatih model KNN
def train_knn_model(X_train, y_train, k_neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn_model.fit(X_train, y_train)
    return knn_model

# Fungsi untuk melakukan prediksi menggunakan model KNN
def make_knn_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Fungsi untuk menampilkan hasil evaluasi model KNN
def evaluate_knn_model(y_true, y_pred, target_names):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=target_names))

# Path folder dengan data gambar training
training_folder_path = 'excel/average_tomato_rgb_hijau.xlsx'

# Membaca data dan mengekstrak fitur dari gambar training
X_train, y_train = load_data_from_folder(training_folder_path)

# Melakukan label encoding pada target
encoded_labels, target_classes = encode_labels(y_train)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_train, encoded_labels, test_size=0.2, random_state=42)

# Melatih model KNN
k_neighbors = 3  # Jumlah tetangga terdekat
knn_model = train_knn_model(X_train, y_train, k_neighbors)

# Membuat prediksi menggunakan model KNN
predictions = make_knn_predictions(knn_model, X_test)

# Menampilkan hasil evaluasi model KNN
evaluate_knn_model(y_test, predictions, target_classes)

# Input gambar yang akan dideteksi kematangannya
input_image_path = 'data_set_tomat/tomat_hijau/tomat_hijau.jpg'
input_features = extract_features(input_image_path)

# Melakukan prediksi kematangan tomat dari gambar input
predicted_label = knn_model.predict([input_features])[0]
predicted_class = target_classes[predicted_label]

print(f"\nPredicted Class: {predicted_class}")
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def extract_features(image_path):
    # Fungsi ini dapat diperluas sesuai dengan fitur-fitur yang ingin Anda gunakan
    image = cv2.imread(image_path)
    # Ekstrak fitur berdasarkan warna, tekstur, dll.
    # Contoh sederhana: menggunakan rata-rata nilai piksel di setiap saluran warna
    features = np.mean(image, axis=(0, 1))
    return features

def train_knn(train_data, labels):
    # Membagi data training menjadi data latih dan data validasi
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        train_data, labels, test_size=0.2, random_state=42
    )

    # Normalisasi fitur
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    valid_features = scaler.transform(valid_features)

    # Inisialisasi k-NN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    # Melatih model
    knn_classifier.fit(train_features, train_labels)

    # Prediksi pada data validasi
    valid_predictions = knn_classifier.predict(valid_features)

    # Evaluasi performa pada data validasi
    accuracy = accuracy_score(valid_labels, valid_predictions)
    print(f'Accuracy on validation data: {accuracy * 100:.2f}%')

    return knn_classifier, scaler

def main():
    # Path folder data training
    folder_path = 'img_tomat/'
    
    # Baca data training dari Excel
    excel_path = 'excel/average_tomato_rgb_merah.xlsx'
    df = pd.read_excel(excel_path)

    # Ekstrak fitur dari setiap gambar
    features_list = []
    labels = []

    for index, row in df.iterrows():
        image_path = os.path.join(folder_path, row['Image'])
        features = extract_features(image_path)
        features_list.append(features)
        labels.append(row['Label'])

    # Konversi list ke dalam array numpy
    features_array = np.array(features_list)
    labels_array = np.array(labels)

    # Latih k-NN classifier
    knn_classifier, scaler = train_knn(features_array, labels_array)

    # Input gambar untuk prediksi
    input_image_path = 'tomat.jpg'  # Ganti dengan path gambar input Anda
    input_features = extract_features(input_image_path)
    input_features = scaler.transform([input_features])  # Normalisasi fitur
    prediction = knn_classifier.predict(input_features)

    # Print hasil prediksi
    print(f'Predicted label: {prediction[0]}')

if __name__ == "__main__":
    main()
