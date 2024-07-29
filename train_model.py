import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

def extract_features(file_name):
    print(f"Extracting features from {file_name}...")
    audio, sample_rate = librosa.load(file_name, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def load_data_from_directory(base_directory, label_order):
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

def train_and_save_model(base_directory, model_path, encoder_path, label_order):
    features, labels = load_data_from_directory(base_directory, label_order)
    
    print("Starting model training...")
    le = LabelEncoder()
    le.fit(label_order)  # 명시적인 라벨 순서 설정
    y_encoded = le.transform(labels)
    print(f"Encoded labels: {set(y_encoded)}")

    print(f"Label order: {label_order}")  # 라벨 순서 출력

    X_train, X_test, y_train, y_test = train_test_split(features, y_encoded, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print("Model training complete.")
    
    # 모델 및 인코더 저장
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Model and encoder saved to {model_path} and {encoder_path}")

if __name__ == "__main__":
    base_directory = 'data'  # 음성 파일이 저장된 기본 디렉토리 경로
    model_path = 'knn_model.pkl'
    encoder_path = 'label_encoder.pkl'
    label_order = ["감사합니다", "네", "아니요", "안녕하세요"]
    train_and_save_model(base_directory, model_path, encoder_path, label_order)