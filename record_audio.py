import pyaudio
import wave
import os
from datetime import datetime

def record_audio(filename_prefix, duration=2, sample_rate=44100, chunk=1024, channels=1, device_index=None):
    audio = pyaudio.PyAudio()
    os.makedirs(filename_prefix, exist_ok=True)

    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=chunk)

    print("Recording...")
    frames = []

    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}/sample_{timestamp}.wav"
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Saved recording to {filename}")
    return filename

if __name__ == "__main__":
    filename_prefix = "data"
    duration = 2
    device_index = None
    record_audio(filename_prefix, duration, device_index=device_index)