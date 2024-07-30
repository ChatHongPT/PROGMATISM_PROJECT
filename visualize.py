import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 음성 데이터 폴더
data_folder = 'data'

# 각 단어 폴더
words = ['안녕하세요', '감사합니다', '네', '아니요']

# 데이터와 레이블을 저장할 리스트
X = []
y = []

# 음성 데이터를 읽고 MFCCs 추출
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# 각 단어 폴더에서 음성 파일 읽고 MFCCs 추출
for word in words:
    word_folder = os.path.join(data_folder, word)
    for file_name in os.listdir(word_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(word_folder, file_name)
            features = extract_features(file_path)
            X.append(features)
            y.append(word)

# 리스트를 numpy 배열로 변환
X = np.array(X)
y = np.array(y)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA를 사용하여 2D로 차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 2D 시각화
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'purple']
for word, color in zip(words, colors):
    indices = np.where(y == word)
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=word, c=color)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of MFCCs')
plt.legend()
plt.show()