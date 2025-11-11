import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple
import time

class GestureDetector:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Diccionario de gestos (aquí puedes agregar más)
        self.gestures = {
            "rascarse_cuello": self._detect_scratch_neck,
            "cruzar_brazos": self._detect_crossed_arms,
            "morderse_unas": self._detect_bite_nails,
            "manos_en_cara": self._detect_hands_face,
            "brazos_abiertos": self._detect_open_arms
        }
        
        self.current_gesture = "Ninguno"
        self.gesture_history = []
        
    def _calculate_distance(self, point1: Tuple, point2: Tuple) -> float:
        """Calcula la distancia entre dos puntos"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _get_landmark_coords(self, landmarks, landmark_idx: int, image_shape: Tuple) -> Tuple:
        """Obtiene coordenadas de un landmark específico"""
        if landmarks is None:
            return None
        landmark = landmarks.landmark[landmark_idx]
        h, w = image_shape
        return (int(landmark.x * w), int(landmark.y * h))
    
    # 🔧 AQUÍ AGREGAS NUEVOS GESTOS - MÉTODOS DE DETECCIÓN
    
    def _detect_scratch_neck(self, results, image_shape: Tuple) -> bool:
        """Detecta si la persona se está rascando el cuello"""
        # Landmarks de mano y cuello
        left_hand_wrist = self._get_landmark_coords(results.left_hand_landmarks, 0, image_shape)
        right_hand_wrist = self._get_landmark_coords(results.right_hand_landmarks, 0, image_shape)
        neck = self._get_landmark_coords(results.pose_landmarks, 0, image_shape)  # Nariz como referencia
        
        if not neck:
            return False
            
        # Verificar si alguna mano está cerca del cuello
        for hand_wrist in [left_hand_wrist, right_hand_wrist]:
            if hand_wrist and self._calculate_distance(hand_wrist, neck) < 100:  # Ajusta este valor
                return True
        return False
    
    def _detect_crossed_arms(self, results, image_shape: Tuple) -> bool:
        """Detecta brazos cruzados"""
        left_wrist = self._get_landmark_coords(results.pose_landmarks, 15, image_shape)  # Muñeca izquierda
        right_wrist = self._get_landmark_coords(results.pose_landmarks, 16, image_shape)  # Muñeca derecha
        left_elbow = self._get_landmark_coords(results.pose_landmarks, 13, image_shape)
        right_elbow = self._get_landmark_coords(results.pose_landmarks, 14, image_shape)
        
        if not all([left_wrist, right_wrist, left_elbow, right_elbow]):
            return False
            
        # Verificar si las muñecas están cerca del codo opuesto
        left_wrist_near_right_elbow = self._calculate_distance(left_wrist, right_elbow) < 80
        right_wrist_near_left_elbow = self._calculate_distance(right_wrist, left_elbow) < 80
        
        return left_wrist_near_right_elbow or right_wrist_near_left_elbow
    
    def _detect_bite_nails(self, results, image_shape: Tuple) -> bool:
        """Detecta morderse las uñas"""
        left_hand_index = self._get_landmark_coords(results.left_hand_landmarks, 8, image_shape)  # Punta índice
        right_hand_index = self._get_landmark_coords(results.right_hand_landmarks, 8, image_shape)
        mouth = self._get_landmark_coords(results.face_landmarks, 13, image_shape)  # Punto cerca de la boca
        
        if not mouth:
            return False
            
        # Verificar si algún dedo está cerca de la boca
        for finger_tip in [left_hand_index, right_hand_index]:
            if finger_tip and self._calculate_distance(finger_tip, mouth) < 50:
                return True
        return False
    
    def _detect_hands_face(self, results, image_shape: Tuple) -> bool:
        """Detecta manos en la cara"""
        left_hand_wrist = self._get_landmark_coords(results.left_hand_landmarks, 0, image_shape)
        right_hand_wrist = self._get_landmark_coords(results.right_hand_landmarks, 0, image_shape)
        face_center = self._get_landmark_coords(results.face_landmarks, 1, image_shape)  # Centro de la cara
        
        if not face_center:
            return False
            
        for hand_wrist in [left_hand_wrist, right_hand_wrist]:
            if hand_wrist and self._calculate_distance(hand_wrist, face_center) < 150:
                return True
        return False
    
    def _detect_open_arms(self, results, image_shape: Tuple) -> bool:
        """Detecta brazos abiertos (ejemplo de gesto adicional)"""
        left_wrist = self._get_landmark_coords(results.pose_landmarks, 15, image_shape)
        right_wrist = self._get_landmark_coords(results.pose_landmarks, 16, image_shape)
        left_shoulder = self._get_landmark_coords(results.pose_landmarks, 11, image_shape)
        right_shoulder = self._get_landmark_coords(results.pose_landmarks, 12, image_shape)
        
        if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return False
            
        # Brazos extendidos lateralmente
        left_arm_extended = abs(left_wrist[0] - left_shoulder[0]) > 150
        right_arm_extended = abs(right_wrist[0] - right_shoulder[0]) > 150
        
        return left_arm_extended and right_arm_extended
    
    def detect_gestures(self, results, image_shape: Tuple) -> str:
        """Detecta todos los gestos y retorna el más probable"""
        detected_gestures = []
        
        for gesture_name, detection_func in self.gestures.items():
            if detection_func(results, image_shape):
                detected_gestures.append(gesture_name)
        
        # Lógica para determinar el gesto principal
        if detected_gestures:
            # Priorizar gestos más específicos
            if "morderse_unas" in detected_gestures:
                return "morderse_unas"
            elif "rascarse_cuello" in detected_gestures:
                return "rascarse_cuello"
            elif "cruzar_brazos" in detected_gestures:
                return "cruzar_brazos"
            else:
                return detected_gestures[0]
        
        return "Ninguno"
    
    def add_gesture(self, gesture_name: str, detection_function):
        """Método para agregar nuevos gestos dinámicamente"""
        self.gestures[gesture_name] = detection_function
        print(f"✅ Gesture '{gesture_name}' added successfully!")
    
    def process_frame(self, image):
        """Procesa un frame y detecta gestos"""
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        
        # Detectar gestos
        self.current_gesture = self.detect_gestures(results, image.shape[:2])
        
        # Dibujar landmarks
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        
        # Mostrar gesto detectado
        cv2.putText(image, f"Gesto: {self.current_gesture}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image

# 📋 EJEMPLO DE USO Y AGREGAR NUEVOS GESTOS
def main():
    detector = GestureDetector()
    cap = cv2.VideoCapture(0)
    
    print("🎯 Sistema de Detección de Gestos Iniciado")
    print("📝 Gestos disponibles:", list(detector.gestures.keys()))
    print("⏹️ Presiona 'q' para salir")
    
    # 🆕 EJEMPLO: CÓMO AGREGAR UN NUEVO GESTO DINÁMICAMENTE
    def detect_saludo(reults, image_shape):
        """Ejemplo: Detectar saludo con la mano"""
        right_hand_wrist = detector._get_landmark_coords(reults.right_hand_landmarks, 0, image_shape)
        head_top = detector._get_landmark_coords(reults.pose_landmarks, 0, image_shape)
        
        if right_hand_wrist and head_top:
            # Mano cerca de la cabeza
            return detector._calculate_distance(right_hand_wrist, head_top) < 120
        return False
    
    # Agregar el nuevo gesto al detector
    detector.add_gesture("saludar", detect_saludo)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Procesar frame
        processed_frame = detector.process_frame(frame)
        
        # Mostrar resultado
        cv2.imshow('Detección de Gestos', processed_frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()