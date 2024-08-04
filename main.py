import os
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
import sounddevice as sd
import numpy as np
import librosa
from sklearn.externals import joblib

class VoiceRecognitionApp(App):

    def build(self):
        self.model = joblib.load('voice_recognition_model.pkl')
        self.layout = BoxLayout(orientation='vertical')

        self.label = Label(text='Press the button and speak')
        self.layout.add_widget(self.label)

        self.record_button = Button(text='Record')
        self.record_button.bind(on_press=self.record_audio)
        self.layout.add_widget(self.record_button)

        return self.layout

    def record_audio(self, instance):
        self.label.text = 'Recording...'
        duration = 2  # seconds
        sample_rate = 44100
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        filename = 'recorded.wav'
        librosa.output.write_wav(filename, recording.flatten(), sample_rate)
        self.label.text = 'Recording finished, predicting...'

        predicted_word, probabilities = self.predict_word(filename)
        self.label.text = f'Prediction: {predicted_word}\nProbabilities: {probabilities}'

    def predict_word(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)
        features = features.reshape(1, -1)
        prediction = self.model.predict(features)
        prediction_probabilities = self.model.predict_proba(features)
        probabilities = {word: prob * 100 for word, prob in zip(self.model.classes_, prediction_probabilities[0])}
        return prediction[0], probabilities

if __name__ == '__main__':
    VoiceRecognitionApp().run()