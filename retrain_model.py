import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from data_preprocessing import load_data
from speech_model import create_model
from tensorflow.keras.callbacks import EarlyStopping

data_path = "data"
features, labels, label_encoder = load_data(data_path)

# 디버깅 출력 추가
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Labels: {labels}")

num_classes = len(np.unique(labels))  # 라벨 수 계산

# 레이블을 원-핫 인코딩으로 변환
if num_classes > 2:
    labels = to_categorical(labels, num_classes=num_classes)
else:
    labels = labels.reshape(-1, 1)  # 이진 분류의 경우 레이블 차원 조정

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

input_shape = (X_train.shape[1], 1)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

model = create_model(input_shape, num_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save("personalized_speech_recognition_model.keras")

import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)