import requests
import json
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# 녹음 설정
DURATION = 2  # 2초
SAMPLE_RATE = 16000  # 16kHz

# 단어 매핑
word_map = {
    0: "안녕하세요",
    1: "감사합니다",
    2: "네",
    3: "아니요"
}

# 녹음 함수
def record_audio():
    print("--------------------")
    print("Please say something: ")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # 녹음 완료될 때까지 대기
    print("Recording finished.")
    return recording

# MFCC 추출 함수 (사용자 정의)
def extract_mfcc(audio, sample_rate):
    # 예: mfcc = some_mfcc_extraction_function(audio, sample_rate)
    # 여기서는 가정된 함수 호출을 사용
    mfcc = np.random.rand(13)  # 이 부분을 실제 MFCC 추출 코드로 대체
    return mfcc

# 서버에 요청 보내기
def send_request(mfccs):
    url = "http://127.0.0.1:5000/predict"
    data = {'mfccs': mfccs.tolist()}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to get a response from the server")
        return None

# 녹음 시작
audio = record_audio()

# 파일로 저장 (선택 사항)
wav.write("sample.wav", SAMPLE_RATE, audio)

# MFCC 추출
mfccs = extract_mfcc(audio, SAMPLE_RATE)

# 서버에 요청 보내기
result = send_request(mfccs)

if result:
    predicted_word = word_map[int(result['prediction'][0])]
    print("--------------------")
    print("예측 결과 :", predicted_word)
    print("--------------------")
    print("일치 확률:")
    for idx, prob in result['probabilities'].items():
        word = word_map[int(idx)]
        print(f"{word}: {prob * 100:.2f}%")