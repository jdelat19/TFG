from collections import deque, Counter

import cv2

try:
    from fer.fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER = None
    FER_AVAILABLE = False

class FacialExpressionDetector:
    """Emoción facial con FER, suavizada promediando las probabilidades de
    los últimos `buffer_size` fotogramas con cara.

    Solo se acepta una emoción si su probabilidad media llega a
    `min_confidence` y supera a la segunda en al menos `min_margin`; si no
    hay un ganador claro, la cara se considera neutral.
    """

    def __init__(self, min_confidence=0.45, min_margin=0.15, buffer_size=6):
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.probabilities = {}  # probabilidades medias, para mostrarlas en pantalla
        self.last_emotion = "Neutral"
        self.no_face_counter = 0

        self.emotion_map = {
            "angry": "Enojo",
            "disgust": "Disgusto",
            "fear": "Miedo",
            "happy": "Feliz",
            "sad": "Triste",
            "surprise": "Sorpresa",
            "neutral": "Neutral",
        }

        self.detector = FER(mtcnn=True) if FER_AVAILABLE else None

    def detect_emotion(self, frame):
        try:
            if self.detector is None:
                self.no_face_counter += 1
                if self.no_face_counter > 30:
                    self.emotion_buffer.clear()
                    self.last_emotion = "No detectado"
                return self.last_emotion, 0.0

            small_frame = cv2.resize(frame, (320, 240))
            result = self.detector.detect_emotions(small_frame)

            if result:
                self.no_face_counter = 0
                self.emotion_buffer.append(result[0]["emotions"])

                averaged = Counter()
                for emotions in self.emotion_buffer:
                    averaged.update(emotions)
                n = len(self.emotion_buffer)
                self.probabilities = {self.emotion_map.get(e, e): p / n for e, p in averaged.items()}
                (top, top_p), (_, second_p) = sorted(
                    self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:2]

                if top_p >= self.min_confidence and top_p - second_p >= self.min_margin:
                    self.last_emotion = top
                else:
                    self.last_emotion = "Neutral"
                return self.last_emotion, top_p
            else:
                self.no_face_counter += 1
                if self.no_face_counter > 30:
                    self.emotion_buffer.clear()
                    self.probabilities = {}
                    self.last_emotion = "No detectado"

        except Exception as e:
            print(f"Error en detección de emoción: {e}")

        return self.last_emotion, 0.0
