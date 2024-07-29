import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def record_voice(filename, duration=3, fs=16000):
    print(f"Recording {filename}...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, recording)  # Save as WAV file
    print(f"Recording saved as {filename}")

def extract_features(file_name):
    print(f"Extracting features from {file_name}...")
    audio, sample_rate = librosa.load(file_name, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def load_data_from_directory(base_directory):
    print(f"Loading data from directory: {base_directory}")
    features = []
    labels = []
    for label in os.listdir(base_directory):
        label_dir = os.path.join(base_directory, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith(".wav"):
                    filepath = os.path.join(label_dir, filename)
                    mfccs = extract_features(filepath)
                    features.append(mfccs)
                    labels.append(label)
    print(f"Loaded labels: {set(labels)}")
    return np.array(features), np.array(labels)

def train_model(features, labels):
    print("Starting model training...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    print(f"Encoded labels: {set(y_encoded)}")

    X_train, X_test, y_train, y_test = train_test_split(features, y_encoded, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print("Model training complete.")
    
    return knn, le

def predict_voice(model, label_encoder, file_name):
    print(f"Predicting voice for {file_name}...")
    mfccs = extract_features(file_name)
    mfccs = mfccs.reshape(1, -1)
    probabilities = model.predict_proba(mfccs)[0]
    predicted_word = model.predict(mfccs)
    predicted_word_label = label_encoder.inverse_transform(predicted_word)[0]
    
    print(f"Label classes: {label_encoder.classes_}")
    # 각 단어별 확률 출력
    print("Prediction probabilities:")
    for word, prob in zip(label_encoder.classes_, probabilities):
        print(f"{word}: {prob*100:.2f}%")
    
    return predicted_word_label

# 데이터 로드 및 모델 학습
base_directory = 'data'  # 음성 파일이 저장된 기본 디렉토리 경로
print("Loading and preparing data...")
features, labels = load_data_from_directory(base_directory)
knn_model, label_encoder = train_model(features, labels)
print("Data preparation and model training complete.")

# 새로운 음성 예측
recorded_file = "recorded_test.wav"
record_voice(recorded_file)
predicted_word = predict_voice(knn_model, label_encoder, recorded_file)
print(f"Predicted word: {predicted_word}")