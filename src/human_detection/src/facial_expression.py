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
    los últimos `buffer_size` fotogramas con cara."""

    def __init__(self, min_confidence=0.4, buffer_size=6):
        self.min_confidence = min_confidence
        self.emotion_buffer = deque(maxlen=buffer_size)
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
                dominant_emotion, total = averaged.most_common(1)[0]
                confidence = total / len(self.emotion_buffer)

                if confidence >= self.min_confidence:
                    self.last_emotion = self.emotion_map.get(dominant_emotion, dominant_emotion)
                    return self.last_emotion, confidence
            else:
                self.no_face_counter += 1
                if self.no_face_counter > 30:
                    self.emotion_buffer.clear()
                    self.last_emotion = "No detectado"

        except Exception as e:
            print(f"Error en detección de emoción: {e}")

        return self.last_emotion, 0.0
