import requests
import os
from record_audio import record_audio
from extract_features import extract_features

# 서버 URL
url = "http://127.0.0.1:5000/predict"

# 음성 데이터 녹음
os.makedirs("data", exist_ok=True)
filename_prefix = "data"
duration = 2  # 녹음 시간 (초)
device_index = None  # 사용할 입력 장치의 인덱스 (list_audio_devices.py로 확인 가능)

print("Please say something (ex '안녕하세요')")
filename = record_audio(filename_prefix, duration, device_index=device_index)

# MFCC 특징 추출
mfccs = extract_features(filename)

# 요청 데이터
data = {
    "mfccs": mfccs.tolist()
}

# HTTP POST 요청
response = requests.post(url, json=data)

# 예측 결과 출력
result = response.json()
print("0: 안녕하세요, 1: 감사합니다, 2: 네, 3: 아니요")
print("Predicted label:", result['prediction'][0])