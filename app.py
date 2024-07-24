from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# 저장된 모델 로드 (최신 Keras 형식 사용)
model = tf.keras.models.load_model("personalized_speech_recognition_model.keras")

# Label encoder 로드
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    mfccs = np.array(data['mfccs']).reshape(1, -1, 1)
    prediction = model.predict(mfccs)
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))

    # 각 단어의 확률 계산
    probabilities = prediction[0]
    probabilities_dict = {str(label_encoder.inverse_transform([i])[0]): float(prob) for i, prob in enumerate(probabilities)}

    return jsonify({
        'prediction': predicted_label.tolist(),
        'probabilities': probabilities_dict
    })

if __name__ == '__main__':
    app.run(debug=True)