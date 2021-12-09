import speech_recognition as sr
from vis_utils.vis_tools import generate_russian_text
from time import sleep
import sys


class TestAudio:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()
        self.text = "[Тихо...]"
        self.FONT_SIZE = 42

    def on_listen(self, recognizer, audio):
        try:
            self.text = recognizer.recognize_google(audio, language='ru-RU').lower()
        except sr.UnknownValueError as e:
            pass
        else:
            generate_russian_text(self.text, one_line=True, shape=(400, 200),
                                  font_size=self.FONT_SIZE, outfile="asr.jpg")
            print('Вы сказали:', self.text)

    def run(self):
        with self.mic as source:
            self.rec.pause_threshold = 1
            print('Тихо...')
            self.rec.adjust_for_ambient_noise(source, duration=1)
        self.rec.listen_in_background(sr.Microphone(), self.on_listen, phrase_time_limit=4)


# r = sr.Recognizer()
#
# with sr.Microphone() as source:
#     r.pause_threshold = 1
#     print('Тихо...')
#     r.adjust_for_ambient_noise(source, duration=1)
#
# print('Нажмите Enter для завершения')
# print('Говорите...')
# r.listen_in_background(sr.Microphone(), on_listen, phrase_time_limit=5)
# input()
