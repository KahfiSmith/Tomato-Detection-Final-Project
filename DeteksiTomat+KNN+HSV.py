import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops, regionprops_table
from collections import Counter
from sklearn.impute import SimpleImputer

path_sawi = 'D:/Kuliah/!Sem5/dataset/belalang/images.cv_45fm2pf20cbzes97d8qtsb/data/train/belalang/Tidak Ada Belalang Terdeteksi/'
path_belalang = 'D:/Kuliah/!Sem5/dataset/belalang/images.cv_45fm2pf20cbzes97d8qtsb/data/train/belalang/Terdeteksi Belalang/'
path_belalang_campur = 'D:/Kuliah/!Sem5/dataset/belalang/images.cv_45fm2pf20cbzes97d8qtsb/data/train/belalang/Train Terdeteksi Belalang/'
path_gambar = 'D:/gambar/'
path_test = 'D:/Kuliah/!Sem5/dataset/belalang/images.cv_45fm2pf20cbzes97d8qtsb/data/test/grasshopper/'

file        = 'trainBelalangBaru2.xlsx'
file_name   = path_gambar + 'anime.png'
dataset     = pd.read_excel(file)
# glcm_properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

fitur       = dataset.iloc[:,+1:-1].values
kelas       = dataset.iloc[:,30].values
tes_fitur   = []
tes_fitur.append([])
tes_kelas   = []
tes_kelas.append([])
# tes_kelas[0].append('menuh')
# print(len(fitur))
# Feature extraction for data testing----------------------------------------------
# Preprocessing
src     = cv2.imread(file_name, 1)
tmp     = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_,mask  = cv2.threshold(tmp,127,255,cv2.THRESH_BINARY_INV)

mask    = cv2.dilate(mask.copy(),None,iterations=10)
mask    = cv2.erode(mask.copy(),None,iterations=10)
b, g, r = cv2.split(src)
rgba    = [b,g,r, mask]
dst     = cv2.merge(rgba,4)



contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
selected    = max(contours,key=cv2.contourArea)
x,y,w,h     = cv2.boundingRect(selected)
cropped     = dst[y:y+h,x:x+w]
mask        = mask[y:y+h,x:x+w]
gray        = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)



# HSV
hsv_image   = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
image       = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))
clt         = KMeans(n_clusters = 3)
labels      = clt.fit_predict(image)
label_counts= Counter(labels)
dom_color   = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

cv2.imshow('hsv',hsv_image)
cv2.imshow('image',image)
print(clt)


tes_fitur[0].append(dom_color[0])
tes_fitur[0].append(dom_color[1])
tes_fitur[0].append(dom_color[2])

# GLCM
# glcm = graycomatrix(gray, 
#                     distances=[5], 
#                     angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
#                     levels=256,
#                     symmetric=True, 
#                     normed=True)
# feature = []
# glcm_props = [propery for name in glcm_properties for propery in graycoprops(glcm, name)[0]]
# for item in glcm_props:            
#     tes_fitur[0].append(item)

# Shape
label_img    = label(mask)
props        = regionprops(label_img)
eccentricity = getattr(props[0], 'eccentricity')
area         = getattr(props[0], 'area')
perimeter    = getattr(props[0], 'perimeter')
metric       = (4*np.pi*area)/(perimeter*perimeter)
tes_fitur[0].append(metric)
tes_fitur[0].append(eccentricity)
# --------------------------------------------------
scaler      = StandardScaler()
scaler.fit(fitur)

imputer = SimpleImputer(strategy='mean')

# Melakukan imputasi pada fitur-fitur yang memiliki nilai NaN
fitur = imputer.fit_transform(fitur)
tes_fitur = imputer.transform(tes_fitur)

# Proses pemodelan setelah melakukan imputasi
classifier = KNeighborsClassifier(n_neighbors=13)
classifier.fit(fitur, kelas)

kelas_pred = classifier.predict(tes_fitur)
print(kelas_pred)

# Menampilkan gambar yang telah diproses
plt.figure(figsize=(8, 4))

# Menampilkan gambar asli yang diproses
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.title('Processed Image')
plt.axis('off')

# Menampilkan nilai kelas_pred
plt.subplot(1, 2, 2)
plt.text(0.5, 0.5, f'Predicted Class: {kelas_pred[0]}', ha='center', va='center', fontsize=12)
plt.axis('off')

plt.show()