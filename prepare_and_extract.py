import os
import librosa
import numpy as np

# 음성 데이터 폴더
data_folder = 'data'

# 각 단어 폴더
words = ['안녕하세요', '감사합니다', '네', '아니요']

# 데이터와 레이블을 저장할 리스트
X = []
y = []

# 음성 데이터를 읽고 특징 추출
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# 데이터 읽기
for word in words:
    word_folder = os.path.join(data_folder, word)
    for file_name in os.listdir(word_folder):
        # 파일 확장자가 .wav 인지 확인
        if file_name.endswith('.wav'):
            file_path = os.path.join(word_folder, file_name)
            features = extract_features(file_path)
            X.append(features)
            y.append(word)

# 리스트를 numpy 배열로 변환하여 저장
X = np.array(X)
y = np.array(y)

# 데이터를 파일로 저장
np.save('X.npy', X)
np.save('y.npy', y)

print("데이터 준비 및 특징 추출 완료")