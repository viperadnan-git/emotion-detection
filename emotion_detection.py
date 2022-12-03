import logging

import cv2
import numpy as np
import pkg_resources
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EmotionDetector")
FACE_BOX_PADDING = 40


class EmotionDetector:
    def __init__(self) -> None:
        self.__emotion_labels = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Normal",
        }
        cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.__face_detector = cv2.CascadeClassifier(cascade_file)
        emotion_model = pkg_resources.resource_filename(
            "fer", "data/emotion_model.hdf5"
        )
        log.debug("Emotion model: {}".format(emotion_model))
        self.__emotion_classifier = load_model(emotion_model, compile=False)
        self.__emotion_classifier.make_predict_function()
        self.__emotion_target_size = self.__emotion_classifier.input_shape[1:3]

    def find_faces(self, img: np.ndarray) -> list:
        gray_image_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.__face_detector.detectMultiScale(
            gray_image_array,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(50, 50),
        )

        return faces

    @staticmethod
    def pad(image):
        row, col = image.shape[:2]
        bottom = image[row - 2 : row, 0:col]
        mean = cv2.mean(bottom)[0]

        padded_image = cv2.copyMakeBorder(
            image,
            top=FACE_BOX_PADDING,
            bottom=FACE_BOX_PADDING,
            left=FACE_BOX_PADDING,
            right=FACE_BOX_PADDING,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean],
        )
        return padded_image

    @staticmethod
    def tosquare(bbox):
        x, y, w, h = bbox
        if h > w:
            diff = h - w
            x -= diff // 2
            w += diff
        elif w > h:
            diff = w - h
            y -= diff // 2
            h += diff
        if w != h:
            log.debug(f"{w} is not {h}")

        return (x, y, w, h)

    def __apply_offsets(self, face_coordinates):
        x, y, width, height = face_coordinates
        x_off, y_off = (10, 10)
        x1 = x - x_off
        x2 = x + width + x_off
        y1 = y - y_off
        y2 = y + height + y_off
        return x1, x2, y1, y2

    @staticmethod
    def __preprocess_input(x, v2=False):
        x = x.astype("float32")
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def detect_emotions(self, img: np.ndarray) -> list:

        face_rectangles = self.find_faces(img)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = self.pad(gray_img)

        emotions = []
        gray_faces = []

        for face_coordinates in face_rectangles:
            face_coordinates = self.tosquare(face_coordinates)

            x1, x2, y1, y2 = self.__apply_offsets(face_coordinates)

            x1 += FACE_BOX_PADDING
            y1 += FACE_BOX_PADDING
            x2 += FACE_BOX_PADDING
            y2 += FACE_BOX_PADDING
            x1 = np.clip(x1, a_min=0, a_max=None)
            y1 = np.clip(y1, a_min=0, a_max=None)

            gray_face = gray_img[max(0, y1) : y2, max(0, x1) : x2]

            try:
                gray_face = cv2.resize(gray_face, self.__emotion_target_size)
            except Exception as e:
                log.warn("{} resize failed: {}".format(gray_face.shape, e))
                continue

            gray_face = self.__preprocess_input(gray_face, True)
            gray_faces.append(gray_face)

        if not len(gray_faces):
            return emotions

        emotion_predictions = self.__emotion_classifier(np.array(gray_faces))

        for face_idx, face in enumerate(emotion_predictions):
            labelled_emotions = {
                self.__emotion_labels[idx]: round(float(score), 2)
                for idx, score in enumerate(face)
            }

            emotions.append(
                dict(box=face_rectangles[face_idx], emotions=labelled_emotions)
            )

        self.emotions = emotions

        return emotions
