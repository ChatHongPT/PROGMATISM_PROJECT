import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import librosa
import os

# 데이터 로드
def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

data = load_data('feedback_data.json')

# 오디오 파일과 텍스트를 분리
audio_files = [item['audio_file'] for item in data]
transcriptions = [item['feedback'] for item in data]

# 텍스트 데이터 전처리
tokenizer = Tokenizer()
tokenizer.fit_on_texts(transcriptions)
sequences = tokenizer.texts_to_sequences(transcriptions)
word_index = tokenizer.word_index

# 패딩 처리
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# 오디오 데이터 전처리
def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

audio_data = [load_audio(file) for file in audio_files]

# 오디오 데이터 패딩 처리
max_audio_length = max(len(audio) for audio in audio_data)
padded_audio_data = np.array([np.pad(audio, (0, max_audio_length - len(audio))) for audio in audio_data])

# 데이터셋 생성
X_audio = padded_audio_data
y_text = padded_sequences

# 라벨 one-hot 인코딩
num_classes = len(word_index) + 1  # 단어 인덱스의 길이에 1을 더하여 클래스 수를 정의합니다.
y_text_one_hot = tf.keras.utils.to_categorical(y_text, num_classes=num_classes)

# 모델 구성
input_audio = tf.keras.Input(shape=(max_audio_length,))
x = tf.keras.layers.Dense(128, activation='relu')(input_audio)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
output_text = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=input_audio, outputs=output_text)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = model.fit(X_audio, y_text_one_hot, epochs=10, batch_size=1)

# 모델 저장
model.save('speech_recognition_model.h5')

# 예측 및 평가 함수
def predict(audio_file):
    audio = load_audio(audio_file)
    padded_audio = np.pad(audio, (0, max_audio_length - len(audio)))
    padded_audio = np.expand_dims(padded_audio, axis=0)
    prediction = model.predict(padded_audio)
    predicted_sequence = np.argmax(prediction, axis=1)
    predicted_text = tokenizer.sequences_to_texts([predicted_sequence])
    return predicted_text[0]

# 예측 결과 확인
test_audio_file = 'output.wav'
predicted_text = predict(test_audio_file)
print(f'Predicted Text: {predicted_text}')