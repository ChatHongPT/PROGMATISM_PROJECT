import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import librosa
import os

# 저장된 데이터 불러오기
X = np.load('X.npy')
y = np.load('y.npy')

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
model.fit(X_train, y_train)

# 모델 평가
accuracy = model.score(X_test, y_test)
print(f'모델 정확도: {accuracy * 100:.2f}%')

# 음성 데이터를 읽고 특징 추출
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# 새 음성 파일 예측
def predict_word(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    prediction_probabilities = model.predict_proba(features)
    return prediction[0], prediction_probabilities

# test_data 폴더에서 테스트 파일 예측
test_data_folder = 'test_data'
test_files = ['hello_test.wav', 'thank_you_test.wav', 'no_test.wav', 'yes_test.wav']

for test_file in test_files:
    file_path = os.path.join(test_data_folder, test_file)
    predicted_word, probabilities = predict_word(file_path)
    print(f'파일: "{test_file}"의 예측된 단어: {predicted_word}')
    for word, prob in zip(model.classes_, probabilities[0]):
        print(f'    단어: "{word}" 확률: {prob * 100:.2f}%')