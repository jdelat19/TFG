import cv2
from fer import FER

class FacialExpressionDetector:
    def __init__(self, min_confidence=0.6):
        self.detector = FER(mtcnn=True)
        self.min_confidence = min_confidence
        self.last_emotion = None

    def detect_emotion(self, frame):
        """
        Detecta la emoción principal en el rostro.
        Retorna el nombre de la emoción o None si no hay detección confiable.
        """
        try:
            result = self.detector.detect_emotions(frame)
            if result:
                emotions = result[0]["emotions"]
                emotion = max(emotions, key=emotions.get)
                confidence = emotions[emotion]

                if confidence >= self.min_confidence:
                    self.last_emotion = emotion
                    return emotion
        except Exception as e:
            print("Error en detección de emoción:", e)

        return self.last_emotion
