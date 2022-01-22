import speech_recognition as sr
from vis_utils.vis_tools import generate_russian_text
from asr_utils.audio_stream import save_stream
from time import sleep
import sys


class TestAudio:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()
        self.time_limit = 4
        self.text = "[Тихо...]"
        self.TEXT = "..."
        self.FONT_SIZE = 42
        self.MODEL = "google"

    def on_listen(self, recognizer, audio):

        if self.MODEL == "google":
            try:
                self.text = recognizer.recognize_google(audio, language='ru-RU').lower()
            except sr.UnknownValueError as e:
                pass
            else:
                generate_russian_text(self.text, one_line=True, shape=(400, 200),
                                      font_size=self.FONT_SIZE, outfile="asr.jpg")
                print('Вы сказали:', self.text)

        elif self.MODEL == "vosk":
            save_stream()


    def run(self):
        with self.mic as source:
            self.rec.pause_threshold = 1
            print('Тихо...')
            self.rec.adjust_for_ambient_noise(source, duration=2)
        self.rec.listen_in_background(sr.Microphone(), self.on_listen, phrase_time_limit=self.time_limit)
