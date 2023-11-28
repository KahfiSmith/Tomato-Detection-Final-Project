import cv2
import os
import pandas as pd
import numpy as np

def resize_image(image, skala_persen=50):
    lebar = int(image.shape[1] * skala_persen / 100)
    tinggi = int(image.shape[0] * skala_persen / 100)
    dimensi = (lebar, tinggi)
    return cv2.resize(image, dimensi, interpolation=cv2.INTER_AREA)

def ambil_rata_hsv(gambar):
    gambar_hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
    rata_hsv = np.mean(gambar_hsv, axis=(0, 1))
    rata_hsv = np.round(rata_hsv).astype(int)  # Bulatkan nilai dan ubah tipe data menjadi integer
    return rata_hsv

def label_kematangan(hsv):
    # Sesuaikan batas nilai HSV untuk menandai kematangan atau mentahan
    batas_h = 30  # Misalnya, anggap nilai di bawah 40 sebagai mentah dan di atas 40 sebagai matang

    if hsv[0] > batas_h:
        return 'Mentah'
    else:
        return 'Matang'

def simpan_gambar_hasil(gambar, path_output):
    cv2.imwrite(path_output, gambar)

def crop_tengah_tomat(gambar):
    tmp     = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    _,mask  = cv2.threshold(tmp,127,255,cv2.THRESH_BINARY_INV)

    mask    = cv2.dilate(mask.copy(), None, iterations=10)
    mask    = cv2.erode(mask.copy(), None, iterations=10)
    b, g, r = cv2.split(gambar)
    rgba    = [b, g, r, mask]
    dst     = cv2.merge(rgba, 4)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Tambahkan pengecekan apakah contours tidak kosong
    if contours:
        selected    = max(contours, key=cv2.contourArea)
        x, y, w, h  = cv2.boundingRect(selected)
        cropped     = dst[y:y+h, x:x+w]

        return cropped
    else:
        # Jika contours kosong, kembalikan gambar asli
        return gambar

def proses_gambar(folder_path):
    data = {'File': [], 'Rata_H': [], 'Rata_S': [], 'Rata_V': [], 'Label': []}

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path_file = os.path.join(folder_path, filename)

            # Baca dan ubah ukuran gambar
            gambar_asli = cv2.imread(path_file)
            gambar_diubah = resize_image(gambar_asli)

            # Lakukan cropping pada bagian tomat
            gambar_diubah = crop_tengah_tomat(gambar_diubah)

            # Dapatkan nilai rata-rata HSV
            rata_hsv = ambil_rata_hsv(gambar_diubah)

            # Dapatkan label kematangan
            label = label_kematangan(rata_hsv)

            # Simpan data ke dalam kamus
            data['File'].append(filename)
            data['Rata_H'].append(rata_hsv[0])
            data['Rata_S'].append(rata_hsv[1])
            data['Rata_V'].append(rata_hsv[2])
            data['Label'].append(label)

            # Simpan gambar hasil ekstraksi yang sudah dikonversi menjadi HSV
            path_output = os.path.join("hasil_ekstraksi", os.path.splitext(filename)[0] + ".jpg")
            simpan_gambar_hasil(cv2.cvtColor(gambar_diubah, cv2.COLOR_BGR2HSV), path_output)

    # Buat DataFrame Pandas
    df = pd.DataFrame(data)

    # Simpan DataFrame ke dalam file Excel
    output_excel_path = 'hasil_kematangan_tomat.xlsx'
    df.to_excel(output_excel_path, index=False)
    print(f"Hasil disimpan ke {output_excel_path}")

if __name__ == "__main__":
    folder_path = "data_set_tomat"  # Ganti dengan path folder lokal Anda
    os.makedirs("hasil_ekstraksi", exist_ok=True)  # Membuat folder jika belum ada
    proses_gambar(folder_path)
