"""Emoción facial con HSEmotion (por defecto) o FER.

HSEmotion (A. V. Savchenko, licencia Apache-2.0) es una EfficientNet entrenada
con AffectNet que se ejecuta con onnxruntime. Necesita la cara ya recortada,
que se obtiene de los puntos de MediaPipe. Si onnxruntime o el modelo no están
disponibles, se usa FER, que detecta la cara por su cuenta con MTCNN.
"""
import os
import urllib.request
from collections import deque, Counter

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from fer.fer import FER
except ImportError:
    FER = None

HSEMOTION_MODEL = "enet_b2_7"
HSEMOTION_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/"
                 f"models/affectnet_emotions/onnx/{HSEMOTION_MODEL}.onnx")
HSEMOTION_PATH = os.path.expanduser(f"~/.hsemotion/{HSEMOTION_MODEL}.onnx")


class HSEmotionBackend:
    """EfficientNet-B2 entrenada con AffectNet (7 emociones)."""

    classes = ["Enojo", "Disgusto", "Miedo", "Feliz", "Neutral", "Triste", "Sorpresa"]
    size = 260
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self):
        if ort is None:
            raise ImportError("onnxruntime no está instalado")
        if not os.path.isfile(HSEMOTION_PATH):
            os.makedirs(os.path.dirname(HSEMOTION_PATH), exist_ok=True)
            print(f"Descargando el modelo de emociones en {HSEMOTION_PATH}...")
            urllib.request.urlretrieve(HSEMOTION_URL, HSEMOTION_PATH)
        self.session = ort.InferenceSession(HSEMOTION_PATH, providers=["CPUExecutionProvider"])

    def predict(self, frame, face_box):
        if face_box is None:
            return None
        x0, y0, x1, y1 = face_box
        face = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        if face.size == 0:
            return None
        x = (cv2.resize(face, (self.size, self.size)) / 255.0 - self.mean) / self.std
        x = x.transpose(2, 0, 1).astype("float32")[np.newaxis]
        logits = self.session.run(None, {"input": x})[0][0]
        e = np.exp(logits - logits.max())
        return dict(zip(self.classes, (e / e.sum()).tolist()))


class FERBackend:
    """Biblioteca FER: MTCNN para la cara y una CNN entrenada con FER-2013."""

    names = {"angry": "Enojo", "disgust": "Disgusto", "fear": "Miedo", "happy": "Feliz",
             "sad": "Triste", "surprise": "Sorpresa", "neutral": "Neutral"}

    def __init__(self):
        if FER is None:
            raise ImportError("fer no está instalado")
        self.detector = FER(mtcnn=True)

    def predict(self, frame, face_box):
        result = self.detector.detect_emotions(cv2.resize(frame, (320, 240)))
        if not result:
            return None
        return {self.names.get(e, e): p for e, p in result[0]["emotions"].items()}


class FacialExpressionDetector:
    """Emoción facial suavizada promediando las probabilidades de los últimos
    `buffer_size` fotogramas con cara.

    Solo se acepta una emoción si su probabilidad media llega a
    `min_confidence` y supera a la segunda en al menos `min_margin`; si no
    hay un ganador claro, la cara se considera neutral.
    """

    def __init__(self, backend="hsemotion", min_confidence=0.45, min_margin=0.15, buffer_size=6):
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.probabilities = {}  # probabilidades medias, para mostrarlas en pantalla
        self.last_emotion = "Neutral"
        self.no_face_counter = 0

        self.backend, self.backend_name = None, "ninguno"
        order = [HSEmotionBackend, FERBackend] if backend == "hsemotion" else [FERBackend, HSEmotionBackend]
        for cls in order:
            try:
                self.backend = cls()
                self.backend_name = cls.__name__.replace("Backend", "")
                break
            except Exception as e:
                print(f"No se puede usar {cls.__name__}: {e}")
        print(f"Emoción facial: {self.backend_name}")

    def detect_emotion(self, frame, face_box=None):
        """`face_box` = (x0, y0, x1, y1) en píxeles; lo necesita HSEmotion."""
        try:
            probs = self.backend.predict(frame, face_box) if self.backend else None

            if probs:
                self.no_face_counter = 0
                self.emotion_buffer.append(probs)

                averaged = Counter()
                for emotions in self.emotion_buffer:
                    averaged.update(emotions)
                n = len(self.emotion_buffer)
                self.probabilities = {e: p / n for e, p in averaged.items()}
                (top, top_p), (_, second_p) = sorted(
                    self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:2]

                if top_p >= self.min_confidence and top_p - second_p >= self.min_margin:
                    self.last_emotion = top
                else:
                    self.last_emotion = "Neutral"
                return self.last_emotion, top_p

            self.no_face_counter += 1
            if self.no_face_counter > 30:
                self.emotion_buffer.clear()
                self.probabilities = {}
                self.last_emotion = "No detectado"

        except Exception as e:
            print(f"Error en detección de emoción: {e}")

        return self.last_emotion, 0.0
