import cv2
from fer.fer import FER
from collections import deque, Counter
import numpy as np

import cv2
from collections import deque, Counter

try:
    from fer.fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER = None
    FER_AVAILABLE = False

class FacialExpressionDetector:
    def __init__(self, min_confidence=0.6, buffer_size=15, stability_threshold=10):
        self.min_confidence = min_confidence
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.stability_threshold = stability_threshold
        self.last_emotion = "Neutral"
        self.consecutive_detections = 0
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
                emotions = result[0]["emotions"]
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion]
                emotion = self.emotion_map.get(dominant_emotion, dominant_emotion)

                if confidence >= self.min_confidence:
                    self.emotion_buffer.append(emotion)
                    if len(self.emotion_buffer) >= self.emotion_buffer.maxlen:
                        most_common = Counter(self.emotion_buffer).most_common(1)[0]
                        if most_common[1] >= self.stability_threshold:
                            self.last_emotion = most_common[0]
                            self.consecutive_detections += 1
                            return self.last_emotion, confidence
            else:
                self.no_face_counter += 1
                if self.no_face_counter > 30:
                    self.emotion_buffer.clear()
                    self.last_emotion = "No detectado"

        except Exception as e:
            print(f"Error en detección de emoción: {e}")

        return self.last_emotion, 0.0

    def get_emotion_history(self):
        return list(self.emotion_buffer)

    def reset(self):
        self.emotion_buffer.clear()
        self.last_emotion = "Neutral"
        self.consecutive_detections = 0
        self.no_face_counter = 0


    def draw_emotion(frame, emotion, confidence, position=(10, 30)):
        if emotion != "No detectado":
            text = f"Emoción: {emotion} ({confidence:.1%})"
            color = (0, 255, 255)
        else:
            text = "Emoción: No detectada"
            color = (100, 100, 100)

        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

# class FacialExpressionDetector:
#     def __init__(self, min_confidence=0.6, buffer_size=15, stability_threshold=10):
#         self.detector = FER(mtcnn=True)
#         self.min_confidence = min_confidence
#         self.emotion_buffer = deque(maxlen=buffer_size)
#         self.stability_threshold = stability_threshold
#         self.last_emotion = "Neutral"
#         self.consecutive_detections = 0
#         self.no_face_counter = 0
        
#         # Configuración de emociones principales
#         self.emotion_map = {
#             'angry': 'Enojo',
#             'disgust': 'Disgusto',
#             'fear': 'Miedo',
#             'happy': 'Feliz',
#             'sad': 'Triste',
#             'surprise': 'Sorpresa',
#             'neutral': 'Neutral'
#         }
    
#     def detect_emotion(self, frame):
#         try:
#             # Reducir tamaño para mejor rendimiento (opcional)
#             small_frame = cv2.resize(frame, (320, 240))
            
#             # Detectar emociones
#             result = self.detector.detect_emotions(small_frame)
            
#             if result:
#                 self.no_face_counter = 0  # Reiniciar contador de "sin rostro"
#                 emotions = result[0]["emotions"]
#                 dominant_emotion = max(emotions, key=emotions.get)
#                 confidence = emotions[dominant_emotion]
                
#                 # Traducir emoción si es necesario
#                 emotion = self.emotion_map.get(dominant_emotion, dominant_emotion)
                
#                 if confidence >= self.min_confidence:
#                     self.emotion_buffer.append(emotion)
                    
#                     # Verificar estabilidad
#                     if len(self.emotion_buffer) >= self.emotion_buffer.maxlen:
#                         most_common = Counter(self.emotion_buffer).most_common(1)[0]
#                         if most_common[1] >= self.stability_threshold:
#                             self.last_emotion = most_common[0]
#                             self.consecutive_detections += 1
#                             return self.last_emotion, confidence
#             else:
#                 self.no_face_counter += 1
#                 # Si no hay rostro por varios frames, resetear
#                 if self.no_face_counter > 30:
#                     self.emotion_buffer.clear()
#                     self.last_emotion = "No detectado"
                
#         except Exception as e:
#             print(f"Error en detección de emoción: {e}")
        
#         # Retornar última emoción estable
#         return self.last_emotion, 0.0
    
#     def get_emotion_history(self):
#         """Obtener historial de emociones"""
#         return list(self.emotion_buffer)
    
#     def reset(self):
#         """Resetear detector"""
#         self.emotion_buffer.clear()
#         self.last_emotion = "Neutral"
#         self.consecutive_detections = 0
#         self.no_face_counter = 0


# # Función de utilidad para dibujar emociones
# def draw_emotion(frame, emotion, confidence, position=(10, 30)):
#     """Dibujar emoción en el frame"""
#     if emotion != "No detectado":
#         text = f"Emoción: {emotion} ({confidence:.1%})"
#         color = (0, 255, 255)  # Amarillo para emociones
#     else:
#         text = "Emoción: No detectada"
#         color = (100, 100, 100)  # Gris
    
#     cv2.putText(frame, text, position, 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#     return frame