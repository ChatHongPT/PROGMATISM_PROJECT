import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def augment_audio(audio, sample_rate):
    noise_amp = 0.005 * np.random.uniform() * np.amax(audio)
    audio = audio + noise_amp * np.random.normal(size=audio.shape)
    return audio

def load_data(data_path):
    features = []
    labels = []
    label_map = {"안녕하세요": 0, "감사합니다": 1, "네": 2, "아니요": 3}
    print(f"Loading data from {data_path}...")
    for label in label_map.keys():
        label_dir = os.path.join(data_path, label)
        for file_name in os.listdir(label_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(label_dir, file_name)
                print(f"Processing file: {file_path}")
                try:
                    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
                    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                    features.append(np.mean(mfccs.T, axis=0))
                    labels.append(label_map[label])

                    # 증강 데이터 추가
                    augmented_audio = augment_audio(audio, sample_rate)
                    mfccs = librosa.feature.mfcc(y=augmented_audio, sr=sample_rate, n_mfcc=40)
                    features.append(np.mean(mfccs.T, axis=0))
                    labels.append(label_map[label])
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    le = LabelEncoder()
    if labels:
        labels = le.fit_transform(labels)
    else:
        labels = np.array([])  # 빈 배열 반환
    
    return np.array(features), np.array(labels), le