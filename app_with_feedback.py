import io
import os
import pyaudio
import wave
from google.cloud import speech_v1 as speech
import requests
import json

# Google Cloud 인증 정보 설정 (환경 변수 설정)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "rising-method-427516-j3-e5b9ebe21c9b.json" # 사용자가 발급받은 구글 인증 정보 파일

# Google Cloud Speech-to-Text 클라이언트 설정
client = speech.SpeechClient()

# 음성 녹음 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

audio = pyaudio.PyAudio()

# 음성 녹음 시작
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("Recording...")
frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("Recording finished")

# 녹음 종료
stream.stop_stream()
stream.close()
audio.terminate()

# 녹음 파일 저장
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

# 음성 파일을 Google Speech-to-Text API로 전송
def transcribe_speech(file_path):
    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko-KR",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        transcript = result.alternatives[0].transcript
        print("Transcript: {}".format(transcript))
        return transcript

# 음성 인식 수행
transcript = transcribe_speech(WAVE_OUTPUT_FILENAME)

# 사용자 피드백 제출
feedback = input("Enter the correct transcription: ")

def submit_feedback(original, feedback):
    url = "http://127.0.0.1:5000/submit_feedback"
    data = {
        'original': original,
        'feedback': feedback
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        print("Feedback submitted successfully")
    else:
        print("Failed to submit feedback")

# 피드백 제출 수행
submit_feedback(transcript, feedback)