import cv2
import numpy as np
from PyQt5.QtGui import QImage, QColor
import pyrebase

# Inisialisasi konfigurasi Firebase
config = {
    "apiKey": "AIzaSyCeboZB9765ir8NKh8C5qnXtiNHjZwI05o",
    "authDomain": "tomatify-b02d2.firebaseapp.com",
    "databaseURL": "https://tomatify-b02d2-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "projectId": "tomatify-b02d2",
    "storageBucket": "tomatify-b02d2.appspot.com",
    "messagingSenderId": "263537612753",
    "appId": "1:263537612753:android:ff16ddcf74fc8dd34f72f0",
    "measurementId": "G-K1F5XRHQEH",
    "serviceAccount": "ServiceACC.json"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

# Download gambar dari Firebase
storage.download("tomat.jpg", "local_image.jpg")

# Baca gambar lokal menggunakan OpenCV
image = cv2.imread("tomat.jpg")

# Dapatkan tinggi dan lebar gambar
height, width = image.shape[:2]

# Buat gambar untuk menyimpan hasil SegmentCitraing
SegmentCitra_image = QImage(width, height, QImage.Format_ARGB32)

SegmentCitra_value = 127  # Kamu bisa mengubah ini sesuai kebutuhan

for y in range(height):
    for x in range(width):
        # Dapatkan warna asli pada posisi piksel (x, y)
        b, g, r = image[y, x]

        # Menghitung nilai grayscale menggunakan metode Luminance
        gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)

        # Menerapkan SegmentCitra
        if gray_value >= SegmentCitra_value:
            new_value = 255
        else:
            new_value = 0

        # Set warna SegmentCitra pada gambar SegmentCitra
        SegmentCitra_color = QColor(new_value, new_value, new_value)
        SegmentCitra_image.setPixelColor(x, y, SegmentCitra_color)

# Simpan gambar hasil SegmentCitraing
SegmentCitra_image.save("SegmentCitra_image.png")

# Upload gambar hasil SegmentCitraing ke Firebase
storage.child("SegmentCitra_image.png").put("SegmentCitra_image.png")



