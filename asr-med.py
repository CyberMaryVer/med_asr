# from fuzzywuzzy import fuzz
import mediapipe as mp
import numpy as np
import cv2
import time
import os

from constants import *
from vis_utils.vis_tools import generate_russian_text, draw_sticker
from asr_utils.audio_stream import save_stream
from asr_utils.background_asr import TestAudio
from mp_utils.mp_geometry import get_distance
from mp_utils.keypoints import MOUTH_CONTOUR, MOUTH_CONNECTIONS1, MOUTH_CONNECTIONS2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class SpeechCorrector:
    def __init__(self,
                 facemesh=mp_face_mesh,
                 detection_confidence=.5,
                 tracking_confidence=.5,
                 landmarks_list=MOUTH_CONTOUR):
        # save parameters
        self.SAVE_OUTPUT = True
        self.SAVE_PATH = "speech.mp4"
        self.video_output = None
        # kps model parameters
        self.facemesh = facemesh
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.landmarks_list = landmarks_list
        # asr model parameters
        self.model = "vosk"
        self.data = GAME_DICT0
        self.word = np.random.choice(self.data)
        self.text = None
        # tracking parameters
        self.mouseX = None
        self.mouseY = None
        self.current_landmarks = None
        self.history_landmarks = None
        self._history = None
        self._audio = None
        self.click = False
        self.start = 0
        # visualization parameters
        self.SHOW_POINTS_ONLY = True
        self.TRANSPARENT_BACK = False
        self.LIGHT_COLOR = (255, 255, 255)
        self.DARK_COLOR = (0, 0, 0)
        self.DARK_MODE = False
        self.W = 1200
        self.H = 800
        self.FONT_SIZE = 54
        self._back = self._get_background()
        self._webcam_screen = (800, 600)
        self._mouth_screen = (400, 400)
        self._audio_screen = (400, 200)
        self._stats_screen = (400, 200)
        assert self._webcam_screen[0] + self._mouth_screen[0] == self.W
        assert self._mouth_screen[1] + self._audio_screen[1] + self._stats_screen[1] == self.H
        assert self._webcam_screen[1] == self.H - 200
        assert self._mouth_screen[0] == self._stats_screen[0] == self._audio_screen[0]

    def _get_background(self, shape=None):
        h, w = (self.H, self.W) if shape is None else shape
        _back = np.zeros((h, w, 3), dtype=np.uint8)
        color = self.DARK_COLOR if self.DARK_MODE else self.LIGHT_COLOR
        _back[:] = color
        return _back

    def _draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("event")
            save_stream()
            self.data = GAME_DICT0 if self.data==GAME_DICT1 else GAME_DICT1
            self.FONT_SIZE = 24 if self.data == GAME_DICT1 else 54
            ta.FONT_SIZE = 24 if self.data == GAME_DICT1 else 42
            self.word = np.random.choice(self.data)
            self.mouseX, self.mouseY = x, y

    def process(self, asr):
        cap = cv2.VideoCapture(0)
        self.history_landmarks = []
        cv2.namedWindow('MedSpeech')
        cv2.setMouseCallback('MedSpeech', self._draw_circle)


        with self.facemesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence) as face_mesh:
            i = 0
            success_delay = 0
            while cap.isOpened():
                i += 1

                # check asr result and save
                if asr.text is not None:
                    self.text = asr.text

                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                multiface_landmarks = results.multi_face_landmarks
                annotated_image = image.copy()
                mouth_landmarks = []

                if self.SHOW_POINTS_ONLY:
                    annotated_image = self._get_background(image.shape[:-1])

                if multiface_landmarks:

                    for face_landmarks in multiface_landmarks:

                        # detect if mouth is open
                        point_up = face_landmarks.landmark[13].x, face_landmarks.landmark[13].y
                        point_down = face_landmarks.landmark[14].x, face_landmarks.landmark[14].y
                        dist = get_distance(point_down, point_up)
                        lips_color = (0, 255, 0) if dist > .04 else (42, 42, 42)  # (0, 132, 255)

                        for idx in self.landmarks_list:
                            rel_x = face_landmarks.landmark[idx].x
                            rel_y = face_landmarks.landmark[idx].y
                            loc_x = int(rel_x * image.shape[1])
                            loc_y = int(rel_y * image.shape[0])
                            cv2.circle(annotated_image, (loc_x, loc_y), 1, lips_color, -1, cv2.LINE_AA)
                            mouth_landmarks.append((loc_x, loc_y))

                        for p1, p2 in MOUTH_CONNECTIONS1:
                            loc_x1 = int(face_landmarks.landmark[p1].x * image.shape[1])
                            loc_y1 = int(face_landmarks.landmark[p1].y * image.shape[0])
                            loc_x2 = int(face_landmarks.landmark[p2].x * image.shape[1])
                            loc_y2 = int(face_landmarks.landmark[p2].y * image.shape[0])
                            cv2.line(annotated_image, (loc_x1, loc_y1), (loc_x2, loc_y2), lips_color, 1, cv2.LINE_AA)

                        for p1, p2 in MOUTH_CONNECTIONS2:
                            loc_x1 = int(face_landmarks.landmark[p1].x * image.shape[1])
                            loc_y1 = int(face_landmarks.landmark[p1].y * image.shape[0])
                            loc_x2 = int(face_landmarks.landmark[p2].x * image.shape[1])
                            loc_y2 = int(face_landmarks.landmark[p2].y * image.shape[0])
                            cv2.line(annotated_image, (loc_x1, loc_y1), (loc_x2, loc_y2), lips_color, 1, cv2.LINE_AA)

                # save landmarks
                mouth_landmarks = np.array(mouth_landmarks)
                self.current_landmarks = mouth_landmarks
                self.history_landmarks.append(mouth_landmarks)

                # get mouth image crop
                try:
                    border_x = 15
                    mouth_x_min = np.nanmin(mouth_landmarks[:, 0]) - border_x
                    mouth_x_max = np.nanmax(mouth_landmarks[:, 0]) + border_x
                    mouth_y_min = np.nanmin(mouth_landmarks[:, 1])
                    mouth_y_max = np.nanmax(mouth_landmarks[:, 1])
                    border_y = int(((mouth_y_min + (mouth_x_max - mouth_x_min)) - mouth_y_max) / 2)
                    mouth_y_min, mouth_y_max = mouth_y_min - border_y, mouth_y_max + border_y
                    # crop mouth
                    mouth = annotated_image[mouth_y_min:mouth_y_max, mouth_x_min:mouth_x_max]
                    if self.TRANSPARENT_BACK:
                        mouth_ = image[mouth_y_min:mouth_y_max, mouth_x_min:mouth_x_max]
                        mouth = cv2.addWeighted(mouth_, .4, mouth, .6, 1)
                except Exception as e:
                    # print(f"{type(e)}: {e}")
                    mouth = None

                # check
                # image = cv2.circle(image, (mouth_x_max, mouth_y_min), 2, self.LIGHT_COLOR, -1)
                # image = cv2.circle(image, (mouth_x_min, mouth_y_max), 2, (0, 255, 0), -1)
                # image = cv2.rectangle(image, (mouth_x_max, mouth_y_max), (mouth_x_min, mouth_y_min), (255, 0, 0), 1)

                mouth = self._get_background(self._mouth_screen) if mouth is None or mouth.size == 0 else mouth

                # flip images for the best perception
                mouth = cv2.flip(mouth, 1)
                image = cv2.flip(image, 1)

                # resize images and place images on the background
                image = cv2.resize(image, self._webcam_screen)
                mouth = cv2.resize(mouth, self._mouth_screen)
                w, h = self._webcam_screen
                wm, hm = self._mouth_screen
                self._back[:h, :w] = image
                self._back[:hm, w:w + wm] = mouth

                # draw challenge
                generate_russian_text(self.word, font_size=self.FONT_SIZE)
                text_img = cv2.imread("text.jpg")
                self._back[600:800, :800] = text_img

                # draw mic
                mic_img = cv2.imread("mic.jpg")
                self._back[600:800, 800:1200] = mic_img

                # draw res
                res_img = cv2.imread("asr.jpg")
                self._back[400:600, 800:1200] = res_img

                # draw frames
                self._back = cv2.rectangle(self._back, (0, 0), (797, 797), (42, 42, 42), 5)
                self._back = cv2.rectangle(self._back, (797, 0), (1197, 400), (42, 42, 42), 5)
                self._back = cv2.rectangle(self._back, (797, 400), (1197, 797), (42, 42, 42), 5)
                # self._back = cv2.rectangle(self._back, (797, 600), (1197, 797), (42, 42, 42), 5)

                # highlight right response
                if self.text is not None:
                    intersect = set(str(self.text)) & set(str(self.word.lower))
                    print( str(self.text)), set(str(self.word.lower))

                    if str(self.word.lower()) in self.text:
                        self._back = cv2.rectangle(self._back, (800, 400), (1185, 600), (0, 255, 0), 15)
                        success_delay += 1
                        if success_delay > 42:
                            success_delay = 0
                            self.word = np.random.choice(self.data)
                        elif self.data == GAME_DICT0:
                            # show sticker
                            sticker_path = "./stickers/" + STICKER_DICT0[self.word.lower()]
                            self._back = draw_sticker(self._back, point=(400, 600), sticker_path=sticker_path)
                            pass # sticker

                    elif len(set(str(self.text)) & set(str(self.word.lower))) > 2:
                        # print(set(str(self.text)) & set(str(self.word.lower)))
                        self._back = cv2.rectangle(self._back, (797, 400), (1197, 600), (42, 142, 255), 5)

                    # else:
                    #     score = fuzz.ratio(self.text.lower(), str(self.word).lower())
                    #     print(f"FUZZ SCORE: {score}")

                if self.SAVE_OUTPUT:
                    frame_ = self._back.copy()

                    if self.video_output is None:  # open output file when 1st frame is received
                        self.W, self.H, _ = self._back.shape
                        self.video_output = cv2.VideoWriter(filename=self.SAVE_PATH,
                                                            fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=40.,
                                                            frameSize=(self.H, self.W), isColor=True, )
                        self.video_output.write(frame_)

                    if self.video_output is not None:
                        frame_ = cv2.resize(frame_, (self.H, self.W))
                        self.video_output.write(frame_)

                cv2.imshow('MedSpeech', self._back)

                key_pressed = cv2.waitKey(5) & 0xFF
                if key_pressed == 27:
                    break
                elif key_pressed == ord('a'):
                    print(self.mouseX, self.mouseY)
        cap.release()

        if self.SAVE_OUTPUT and self.video_output is not None:
            self.video_output.release()


def extract_keypoints(multiface_landmarks):
    """
    Extract keypoints from data object
    :param results: obj // results of pose estimation by mediapipe
    :return: list // list of keypoints dictionaries
    """
    all_keypoints = []

    if not multiface_landmarks is None:
        for kp in multiface_landmarks:
            print(kp)
            try:
                print(kp.x, kp.y)
                # all_keypoints.append([kp.x, kp.y, kp.z, kp.visibility])  # [X, Y, Z, visibility]
            except Exception as e:
                print(f"{type(e)}: {e}")
        # all_keypoints.append({kp_name: keypoints})
    return all_keypoints


if __name__ == "__main__":
    ta = TestAudio()
    sc = SpeechCorrector()
    ta.run()
    sc.process(ta)
