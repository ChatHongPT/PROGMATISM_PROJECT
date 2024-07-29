import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
import pickle

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

def predict_voice(model_path, encoder_path, file_name):
    # 모델 및 인코더 로드
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
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

if __name__ == "__main__":
    model_path = 'knn_model.pkl'
    encoder_path = 'label_encoder.pkl'
    recorded_file = "recorded_test.wav"
    record_voice(recorded_file)
    predicted_word = predict_voice(model_path, encoder_path, recorded_file)
    print(f"Predicted word: {predicted_word}")