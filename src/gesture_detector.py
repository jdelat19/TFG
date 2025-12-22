import cv2
import mediapipe as mp
from typing import List
from gestures import BaseGesture, DEFAULT_GESTURES
from facial_expression import FacialExpressionDetector

class GestureDetector:
    def __init__(self, gestures: List[BaseGesture] = None, 
                 draw_face=True, draw_pose=True, draw_hands=True):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.gestures = gestures or DEFAULT_GESTURES
        self.current_gesture = "Ninguno"
        self.gesture_history = []

        self.draw_face = draw_face
        self.draw_pose = draw_pose
        self.draw_hands = draw_hands
        self.emotion_detector = FacialExpressionDetector()
    
    def list_gestures(self):
        return [gesture.name for gesture in self.gestures]
    
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

        # Seleccionar solo el gesto de mayor prioridad
        highest_priority_gesture = max(detected_gestures, key=lambda g: g.priority)
        return [highest_priority_gesture.name]  # Devuelve lista con un solo gesto

    # ===========================
    # Deteccion de varios gestos
    # ===========================

    # def detect_gestures(self, results, image_shape: tuple) -> list:
    #     detected_gestures = []

    #     for gesture in self.gestures:
    #         try:
    #             if gesture.check(results, image_shape):
    #                 detected_gestures.append(gesture.name)
    #         except Exception as e:
    #             print(f"Error en gesto {gesture.name}: {e}")

    #     return detected_gestures  # devuelve lista de gestos detectados

    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        detected_gestures = self.detect_gestures(results, image.shape[:2])
        self.current_gesture = detected_gestures[0] if detected_gestures else "Ninguno"

        self._draw_landmarks(image, results)

        # Mostrar el gesto
        cv2.putText(image, f"Gesto: {self.current_gesture}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Mostrar emoción facial
        emotion = self.emotion_detector.detect_emotion(image)
        if emotion:
            cv2.putText(image, f"Emoción: {emotion}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

        # Escalar imagen
        scale_percent = 150
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width, height))

        return image
    
    def _draw_landmarks(self, image, results):
        if self.draw_face and results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )

        if self.draw_pose and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )

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
