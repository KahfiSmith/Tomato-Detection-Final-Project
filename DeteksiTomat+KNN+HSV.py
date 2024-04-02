import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.impute import SimpleImputer
import mysql.connector
import time

def insert_into_database(kelas_pred):
    # Ganti dengan informasi koneksi ke database MySQL Anda
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'tomatify'
    }

    try:
        # Membuat koneksi ke database
        conn = mysql.connector.connect(**db_config)

        # Membuat objek cursor untuk eksekusi query
        cursor = conn.cursor()

        # Menyisipkan nilai kelas_pred ke dalam tabel tertentu (ganti 'nama_tabel' dengan nama tabel yang benar)
        query = "INSERT INTO kematangan_tomat (kematangan) VALUES (%s)"
        values = (kelas_pred,)

        cursor.execute(query, values)

        # Commit perubahan ke database
        conn.commit()

        print("Data berhasil dimasukkan ke database!")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # Menutup koneksi dan cursor
        if conn.is_connected():
            cursor.close()
            conn.close()

# PATH PENYIMPANAN
path_tomat = 'captured_images/'
file = 'hasil_kematangan_tomat.xlsx'
file_name = path_tomat + 'tomat.jpg'
dataset = pd.read_excel(file)

fitur = dataset.iloc[:, 1:4].values
kelas = dataset.iloc[:, 4].values
tes_fitur = [[]]
tes_kelas = [[]]

src = cv2.imread(file_name, 1)
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV)

mask = cv2.dilate(mask.copy(), None, iterations=10)
mask = cv2.erode(mask.copy(), None, iterations=10)
b, g, r = cv2.split(src)
rgba = [b, g, r, mask]
dst = cv2.merge(rgba, 4)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
selected = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(selected)
cropped = dst[y:y+h, x:x+w]
mask = mask[y:y+h, x:x+w]
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

# HSV
hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))
clt = KMeans(n_clusters=3)
labels = clt.fit_predict(image)
label_counts = Counter(labels)
dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

tes_fitur[0].append(dom_color[0])
tes_fitur[0].append(dom_color[1])
tes_fitur[0].append(dom_color[2])

scaler = StandardScaler()
scaler.fit(fitur)

imputer = SimpleImputer(strategy='mean')

# Melakukan imputasi pada fitur-fitur yang memiliki nilai NaN
fitur = imputer.fit_transform(fitur)
tes_fitur = imputer.transform(tes_fitur)

while True:

    # Proses pemodelan setelah melakukan imputasi
    classifier = KNeighborsClassifier(n_neighbors=13)
    classifier.fit(fitur, kelas)

    # Melakukan prediksi kelas
    kelas_pred = classifier.predict(tes_fitur)
    print("Kelas Prediksi:", kelas_pred[0])

    # Insert kematangan tomat ke dalam database
    insert_into_database(kelas_pred[0])

    time.sleep(10)
