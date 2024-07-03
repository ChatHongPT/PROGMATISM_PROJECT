import io
import os
import pyaudio
import wave
from google.cloud import speech_v1 as speech
import requests
import json
from config import FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, WAVE_OUTPUT_FILENAME

# Google Cloud 인증 정보 설정 (환경 변수 설정)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "rising-method-427516-j3-e5b9ebe21c9b.json"  # 사용자가 발급받은 구글 인증 정보 파일

# Google Cloud Speech-to-Text 클라이언트 설정
client = speech.SpeechClient()

def record_audio():
    """음성을 녹음하여 파일로 저장"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording finished")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def transcribe_speech(file_path):
    """녹음된 음성을 텍스트로 변환"""
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
    return ""

def submit_feedback(original, feedback):
    """피드백을 서버에 제출"""
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

def main():
    record_audio()
    transcript = transcribe_speech(WAVE_OUTPUT_FILENAME)
    if transcript:
        feedback = input("Enter the correct transcription: ")
        submit_feedback(transcript, feedback)

if __name__ == "__main__":
    main()