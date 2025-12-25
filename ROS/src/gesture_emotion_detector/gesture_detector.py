import cv2
import mediapipe as mp
import numpy as np 
from typing import List, Tuple
from gestures import BaseGesture, DEFAULT_GESTURES
from facial_expression import FacialExpressionDetector
from collections import deque, Counter


class GestureDetector:
    def __init__(self, gestures: List[BaseGesture] = None,
                 draw_face=True, draw_pose=True, draw_hands=True,
                 enable_emotion_detection=True):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.temporal_window = 12
        self.temporal_threshold = 7
        self.gesture_buffer = deque(maxlen=self.temporal_window)
        self.smoothed_gesture = "Ninguno"
        
        # Configuración de visualización
        self.draw_face = draw_face
        self.draw_pose = draw_pose
        self.draw_hands = draw_hands
        
        # Inicializar modelos
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.gestures = gestures or DEFAULT_GESTURES
        self.current_gesture = "Ninguno"
        
        # Detector de emociones (solo si está habilitado)
        self.enable_emotion_detection = enable_emotion_detection
        if enable_emotion_detection:
            self.emotion_detector = FacialExpressionDetector(min_confidence=0.6)
        else:
            self.emotion_detector = None
        
        # Historial para análisis
        self.gesture_history = []
        self.emotion_history = []
        self.frame_count = 0
    
    def temporal_smoothing(self, detected_gesture: str) -> str:
        self.gesture_buffer.append(detected_gesture)
        
        if len(self.gesture_buffer) < self.temporal_window:
            return "Ninguno"
        
        counter = Counter(self.gesture_buffer)
        gesture, count = counter.most_common(1)[0]
        
        if gesture != "Ninguno" and count >= self.temporal_threshold:
            return gesture
        
        return "Ninguno"
    
    def detect_gestures(self, results, image_shape: tuple) -> list:
        detected_gestures = []
        
        for gesture in self.gestures:
            try:
                if gesture.check(results, image_shape):
                    detected_gestures.append(gesture)
            except Exception as e:
                print(f"Error en gesto {gesture.name}: {e}")
        
        if not detected_gestures:
            return []
        
        # Seleccionar gesto de mayor prioridad
        highest_priority_gesture = max(detected_gestures, key=lambda g: g.priority)
        return [highest_priority_gesture.name]
    
    def process_frame(self, image) -> Tuple[np.ndarray, dict]:
        """Procesa un frame y retorna imagen procesada y datos"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        
        # Detectar gestos
        detected_gestures = self.detect_gestures(results, image.shape[:2])
        raw_gesture = detected_gestures[0] if detected_gestures else "Ninguno"
        self.current_gesture = self.temporal_smoothing(raw_gesture)
        
        # Guardar en historial
        if self.frame_count % 5 == 0:  # Guardar cada 5 frames
            self.gesture_history.append(self.current_gesture)
        
        # Detectar emociones si está habilitado
        emotion_data = {"emotion": "Desactivado", "confidence": 0.0}
        if self.enable_emotion_detection and self.emotion_detector:
            emotion, confidence = self.emotion_detector.detect_emotion(image)
            emotion_data = {"emotion": emotion, "confidence": confidence}
            
            if self.frame_count % 5 == 0:
                self.emotion_history.append(emotion)
        
        # Dibujar landmarks
        self._draw_landmarks(image, results)
        
        # Mostrar información en pantalla
        self._display_info(image, raw_gesture, emotion_data)
        
        # Escalar imagen para mejor visualización
        image = self._resize_image(image, scale_percent=150)
        
        self.frame_count += 1
        return image, {
            "gesture": self.current_gesture,
            "raw_gesture": raw_gesture,
            **emotion_data
        }
    
    def _display_info(self, image, raw_gesture: str, emotion_data: dict):
        """Mostrar información en pantalla"""
        # Gesto suavizado
        cv2.putText(image, f"Gesto: {self.current_gesture}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        
        # # Gesto en crudo
        # cv2.putText(image, f"Gesto (raw): {raw_gesture}",
        #            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 150, 255), 2)
        
        # Emoción
        if self.enable_emotion_detection:
            y_position = 100
            emotion_text = f"Emocion: {emotion_data['emotion']}"
            
            color = (0, 255, 255) if emotion_data['emotion'] != "Desactivado" else (100, 100, 100)
            cv2.putText(image, emotion_text, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    def _draw_landmarks(self, image, results):
        """Dibujar landmarks según configuración"""
        # Cara
        if self.draw_face and results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )
        
        # Cuerpo
        if self.draw_pose and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
        
        # Manos
        if self.draw_hands:
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
    
    def _resize_image(self, image, scale_percent=150):
        """Redimensionar imagen para visualización"""
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        return cv2.resize(image, (width, height))
    
    def get_statistics(self):
        """Obtener estadísticas de detección"""
        if not self.gesture_history or not self.emotion_history:
            return {}
        
        # Estadísticas de gestos
        gesture_counter = Counter(self.gesture_history)
        most_common_gesture = gesture_counter.most_common(1)[0] if gesture_counter else ("Ninguno", 0)
        
        # Estadísticas de emociones
        emotion_counter = Counter(self.emotion_history)
        most_common_emotion = emotion_counter.most_common(1)[0] if emotion_counter else ("Desactivado", 0)
        
        return {
            "total_frames": self.frame_count,
            "most_common_gesture": most_common_gesture[0],
            "gesture_frequency": most_common_gesture[1],
            "most_common_emotion": most_common_emotion[0],
            "emotion_frequency": most_common_emotion[1],
            "unique_gestures": len(gesture_counter),
            "unique_emotions": len(emotion_counter)
        }
    
    def list_gestures(self):
        return [gesture.name for gesture in self.gestures]
    
    def close(self):
        """Liberar recursos"""
        if self.holistic:
            self.holistic.close()