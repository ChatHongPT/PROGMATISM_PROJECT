import pyaudio
import wave
import os
from datetime import datetime

def record_audio_batch(base_dir="data", duration=2, sample_rate=44100, chunk=1024, channels=1, device_index=None):
    audio = pyaudio.PyAudio()
    os.makedirs(base_dir, exist_ok=True)

    label_map = {
        "0": "안녕하세요",
        "1": "감사합니다",
        "2": "네",
        "3": "아니요"
    }

    while True:
        label_key = input("Enter the number (0: 안녕하세요, 1: 감사합니다, 2: 네, 3: 아니요): ")
        
        if label_key not in label_map:
            print("Invalid input. Please enter a number between 0 and 3.")
            continue

        label = label_map[label_key]
        label_dir = os.path.join(base_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        stream = audio.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate,
                            input=True, input_device_index=device_index, frames_per_buffer=chunk)

        print(f"Recording for label '{label}'...")
        frames = [stream.read(chunk) for _ in range(0, int(sample_rate / chunk * duration))]
        print("Recording finished.")

        stream.stop_stream()
        stream.close()

        filename = os.path.join(label_dir, f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

        print(f"Saved recording to {filename}")

        if input("Continue recording? (y/n): ").lower() != 'y':
            break

    audio.terminate()

if __name__ == "__main__":
    record_audio_batch()