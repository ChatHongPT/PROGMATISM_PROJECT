import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import librosa
import os

def load_data(json_file):
    """피드백 데이터를 로드하여 반환합니다."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_text(transcriptions):
    """텍스트 데이터를 전처리하고 패딩 처리하여 반환합니다."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(transcriptions)
    sequences = tokenizer.texts_to_sequences(transcriptions)
    word_index = tokenizer.word_index
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences, tokenizer, max_sequence_length

def load_audio(file_path, sr=16000):
    """오디오 파일을 로드하여 반환합니다."""
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def preprocess_audio(audio_files):
    """오디오 데이터를 전처리하고 패딩 처리하여 반환합니다."""
    audio_data = [load_audio(file) for file in audio_files]
    max_audio_length = max(len(audio) for audio in audio_data)
    padded_audio_data = np.array([np.pad(audio, (0, max_audio_length - len(audio))) for audio in audio_data])
    return padded_audio_data, max_audio_length

def build_model(input_shape, output_units):
    """신경망 모델을 구성하고 반환합니다."""
    input_audio = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(input_audio)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output_text = tf.keras.layers.Dense(output_units, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_audio, outputs=output_text)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    """모델을 훈련합니다."""
    history = model.fit(X_train, y_train, epochs=10, batch_size=1)
    return history

def save_model(model, model_path):
    """모델을 저장합니다."""
    model.save(model_path, save_format='keras')

def predict(model, audio_file, max_audio_length, tokenizer):
    """새로운 오디오 파일에 대해 예측을 수행합니다."""
    audio = load_audio(audio_file)
    padded_audio = np.pad(audio, (0, max_audio_length - len(audio)))
    padded_audio = np.expand_dims(padded_audio, axis=0)
    prediction = model.predict(padded_audio)
    predicted_sequence = np.argmax(prediction, axis=1)
    predicted_text = tokenizer.sequences_to_texts([predicted_sequence])
    return predicted_text[0]

def main():
    # 데이터 로드
    data = load_data('feedback_data.json')

    # 오디오 파일과 텍스트를 분리
    audio_files = [item['audio_file'] for item in data]
    transcriptions = [item['feedback'] for item in data]

    # 텍스트 데이터 전처리
    padded_sequences, tokenizer, max_sequence_length = preprocess_text(transcriptions)

    # 오디오 데이터 전처리
    padded_audio_data, max_audio_length = preprocess_audio(audio_files)

    # 데이터셋 생성
    X_audio = padded_audio_data
    y_text = padded_sequences

    # 라벨 one-hot 인코딩
    num_classes = len(tokenizer.word_index) + 1
    y_text_one_hot = tf.keras.utils.to_categorical(y_text, num_classes=num_classes)

    # 모델 구성
    model = build_model((max_audio_length,), num_classes)

    # 모델 훈련
    train_model(model, X_audio, y_text_one_hot)

    # 모델 저장
    save_model(model, 'speech_recognition_model.keras')

    # 예측 결과 확인
    test_audio_file = 'output.wav'
    predicted_text = predict(model, test_audio_file, max_audio_length, tokenizer)
    print(f'Predicted Text: {predicted_text}')

if __name__ == "__main__":
    main()