import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
import pickle
import os

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
    # Load the model and encoder
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

def retrain_model(new_data_path, model_path, encoder_path):
    print(f"Retraining model with new data from {new_data_path}...")
    # Load existing model and data
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    features = []
    labels = []
    
    for label in os.listdir(new_data_path):
        label_dir = os.path.join(new_data_path, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith(".wav"):
                    filepath = os.path.join(label_dir, filename)
                    mfccs = extract_features(filepath)
                    features.append(mfccs)
                    labels.append(label)
    
    # Encode the labels
    y_encoded = label_encoder.transform(labels)
    
    # Add new data to existing model
    model.fit(features, y_encoded)
    
    # Save the updated model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model retraining complete and saved.")

if __name__ == "__main__":
    model_path = 'knn_model.pkl'
    encoder_path = 'label_encoder.pkl'
    recorded_file = "recorded_test.wav"
    
    # Record voice and predict
    record_voice(recorded_file)
    predicted_word = predict_voice(model_path, encoder_path, recorded_file)
    print(f"Predicted word: {predicted_word}")
    
    # Ask user if prediction is correct
    correct = input(f"Is the predicted word '{predicted_word}' correct? (yes/no): ").strip().lower()
    
    if correct == 'no':
        correct_word = input("Please provide the correct word: ").strip()
        label_order = ["감사합니다", "네", "아니요", "안녕하세요"]
        
        if correct_word not in label_order:
            print(f"The provided word '{correct_word}' is not in the label order list.")
        else:
            # Save the incorrectly predicted audio for retraining
            correct_dir = os.path.join('data', correct_word)
            os.makedirs(correct_dir, exist_ok=True)
            corrected_filename = os.path.join(correct_dir, f"{correct_word}_{np.random.randint(10000)}.wav")
            os.rename(recorded_file, corrected_filename)
            print(f"Saved corrected audio as {corrected_filename}")
            
            # Retrain the model
            retrain_model('data', model_path, encoder_path)
    else:
        print("Prediction is correct.")