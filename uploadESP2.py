import pyrebase
import base64
from io import BytesIO

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
db = firebase.database()
storage = firebase.storage()

# Ambil data dari Realtime Database
data = db.child('esp32-cam/-NixppaJt8nMDflhUci1/photo').get().val()

# Menghapus header (jika ada)
if data.startswith('data:image/png;base64,'):
    data = data.split('data:image/png;base64,')[1]

# Bersihkan string (hilangkan spasi dan baris baru jika ada)
data = data.replace(' ', '').replace('\n', '')

# Tambahkan padding jika diperlukan
padding = len(data) % 4
if padding != 0:
    data += '='* (4 - padding)

try:
    image_data = base64.b64decode(data)
    image_stream = BytesIO(image_data)  # Membuat stream dari data gambar

    # Upload gambar hasil decoding ke Firebase Storage
    storage.child("SegmentCitra_image.png").put(image_stream)  # Menggunakan stream untuk upload

    print("Gambar dalam format Base64 telah diunduh dari Realtime Database dan diunggah kembali ke Firebase Storage.")
except base64.binascii.Error as error:
    print("Dekode Base64 gagal: ", error)
except Exception as error:
    print("Error mengunggah gambar: ", error)  # Mencetak error apapun yang terjadi saat upload