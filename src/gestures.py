import math
import numpy as np
from abc import ABC, abstractmethod
from utils import calculate_distance, get_landmark_coords
from collections import deque

class BaseGesture(ABC):
    def __init__(self, name: str, confidence_threshold: float = 0.5):
        self.name = name
        self.confidence_threshold = confidence_threshold
        self.priority = 1
    
    @abstractmethod
    def check(self, results, image_shape: tuple) -> bool:
        pass
    
    def __str__(self):
        return self.name

# =============================================================================
# GESTOS CORPORALES
# =============================================================================

class CrossedArmsGesture(BaseGesture):
    def __init__(self):
        super().__init__("cruzar_brazos", 0.6)
        self.priority = 4
    
    def check(self, results, image_shape):
        left_wrist = get_landmark_coords(results.pose_landmarks, 15, image_shape)
        right_wrist = get_landmark_coords(results.pose_landmarks, 16, image_shape)
        left_elbow = get_landmark_coords(results.pose_landmarks, 13, image_shape)
        right_elbow = get_landmark_coords(results.pose_landmarks, 14, image_shape)
        
        if not all([left_wrist, right_wrist, left_elbow, right_elbow]):
            return False
        
        # Cruzar brazos: muñecas cerca, codos separados
        wrists_close = calculate_distance(left_wrist, right_wrist) < 120
        elbows_far = calculate_distance(left_elbow, right_elbow) > 150
        
        return wrists_close and elbows_far

class OpenArmsGesture(BaseGesture):
    def __init__(self):
        super().__init__("brazos_abiertos", 0.5)
    
    def check(self, results, image_shape):
        left_wrist = get_landmark_coords(results.pose_landmarks, 15, image_shape)
        right_wrist = get_landmark_coords(results.pose_landmarks, 16, image_shape)
        left_shoulder = get_landmark_coords(results.pose_landmarks, 11, image_shape)
        right_shoulder = get_landmark_coords(results.pose_landmarks, 12, image_shape)
        
        if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return False
            
        # Brazos extendidos hacia los lados
        left_arm_extended = abs(left_wrist[0] - left_shoulder[0]) > 200
        right_arm_extended = abs(right_wrist[0] - right_shoulder[0]) > 200
        # Y manos por encima de la cintura
        hands_above_waist = left_wrist[1] < left_shoulder[1] + 150 and right_wrist[1] < right_shoulder[1] + 150
        
        return left_arm_extended and right_arm_extended and hands_above_waist

class HandsTogetherGesture(BaseGesture):
    def __init__(self):
        super().__init__("manos_juntas", 0.6)
        self.priority = 2
    
    def check(self, results, image_shape):
        left_wrist = get_landmark_coords(results.pose_landmarks, 15, image_shape)
        right_wrist = get_landmark_coords(results.pose_landmarks, 16, image_shape)
        left_elbow = get_landmark_coords(results.pose_landmarks, 13, image_shape)
        right_elbow = get_landmark_coords(results.pose_landmarks, 14, image_shape)

        if not all([left_wrist, right_wrist, left_elbow, right_elbow]):
            return False

        # Manos juntas pero no cruzadas
        wrists_close = calculate_distance(left_wrist, right_wrist) < 100
        elbows_close = calculate_distance(left_elbow, right_elbow) < 150
        return wrists_close and elbows_close

# =============================================================================
# GESTOS DE MANOS Y CARA - CORREGIDOS
# =============================================================================

class ScratchNeckGesture(BaseGesture):
    def __init__(self):
        super().__init__("rascarse_cuello", 0.7)
        self.priority = 6
    
    def check(self, results, image_shape):
        left_shoulder = get_landmark_coords(results.pose_landmarks, 11, image_shape)
        right_shoulder = get_landmark_coords(results.pose_landmarks, 12, image_shape)
        if not left_shoulder or not right_shoulder:
            return False

        neck_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                       (left_shoulder[1] + right_shoulder[1]) // 2)
        top = neck_center[1] - 30
        bottom = neck_center[1] + 60
        left = min(left_shoulder[0], right_shoulder[0]) - 20
        right = max(left_shoulder[0], right_shoulder[0]) + 20

        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                for idx in [0, 8]:  # muñeca y punta índice
                    point = get_landmark_coords(hand_landmarks, idx, image_shape)
                    if point and left <= point[0] <= right and top <= point[1] <= bottom:
                        return True
        return False

# Rascarse el cuello 
# class ScratchNeckGesture(BaseGesture):
#     def __init__(self):
#         super().__init__("rascarse_cuello", 0.7)
#         self.priority = 4

#     def check(self, results, image_shape):
#         left_hand_landmarks = results.left_hand_landmarks
#         right_hand_landmarks = results.right_hand_landmarks

#         left_shoulder = get_landmark_coords(results.pose_landmarks, 11, image_shape)
#         right_shoulder = get_landmark_coords(results.pose_landmarks, 12, image_shape)
#         nose = get_landmark_coords(results.pose_landmarks, 0, image_shape)
#         mouth = get_landmark_coords(results.pose_landmarks, 9, image_shape)  # superior boca

#         if not left_shoulder or not right_shoulder:
#             return False

#         # Calcular centro del cuello (entre hombros)
#         neck_center = (
#             (left_shoulder[0] + right_shoulder[0]) // 2,
#             (left_shoulder[1] + right_shoulder[1]) // 2
#         )

#         upper_limit = neck_center[1] - 40  # un poco arriba del cuello
#         lower_limit = neck_center[1] + 80  # un poco debajo del cuello

#         def is_near_neck(hand_point):
#             if not hand_point:
#                 return False

#             # Evitar confundir con boca o nariz
#             if mouth and calculate_distance(hand_point, mouth) < 60:
#                 return False
#             if nose and calculate_distance(hand_point, nose) < 60:
#                 return False

#             # Definir región del cuello más precisa
#             top = neck_center[1] - 40
#             bottom = neck_center[1] + 40
#             left = neck_center[0] - 50
#             right = neck_center[0] + 50

#             x, y = hand_point
#             if not (left <= x <= right and top <= y <= bottom):
#                 return False

#             # Rechazar puntos por debajo de los hombros (evita pecho)
#             if left_shoulder and y > left_shoulder[1] + 20:
#                 return False

#             return True
        
#         # Verificar mano izquierda
#         if left_hand_landmarks:
#             for idx in [0, 8]:  # muñeca y punta del índice
#                 point = get_landmark_coords(left_hand_landmarks, idx, image_shape)
#                 if point and is_near_neck(point):
#                     return True

#         # Verificar mano derecha
#         if right_hand_landmarks:
#             for idx in [0, 8]:
#                 point = get_landmark_coords(right_hand_landmarks, idx, image_shape)
#                 if point and is_near_neck(point):
#                     return True

#         return False


# Morderse las uñas
class BiteNailsGesture(BaseGesture):
    def __init__(self):
        super().__init__("morderse_unas", 0.8)
        self.priority = 9

    def check(self, results, image_shape):
        if not results.face_landmarks:
            return False

        # Landmarks de la boca
        mouth_ids = [61, 291, 13, 14]
        mouth_points = [
            get_landmark_coords(results.face_landmarks, idx, image_shape)
            for idx in mouth_ids
        ]

        if not all(mouth_points):
            return False

        # Bounding box de la boca
        xs = [p[0] for p in mouth_points]
        ys = [p[1] for p in mouth_points]
        left, right = min(xs) - 10, max(xs) + 10
        top, bottom = min(ys) - 10, max(ys) + 10

        # 👉 AQUÍ VA
        finger_tips = [4, 8, 12, 16, 20]

        # Comprobar ambas manos
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if not hand_landmarks:
                continue

            for idx in finger_tips:
                tip = get_landmark_coords(hand_landmarks, idx, image_shape)
                if not tip:
                    continue

                x, y = tip
                if left <= x <= right and top <= y <= bottom:
                    return True

        return False


# Manos en la cara
class HandsFaceGesture(BaseGesture):
    def __init__(self):
        super().__init__("manos_en_cara", 0.6)
        self.priority = 6
    
    def check(self, results, image_shape):
        if not results.face_landmarks:
            return False

        # Tomar algunos landmarks de referencia de la cara
        face_points_ids = [10, 152, 234, 454]  # frente, mentón, mejillas
        face_coords = [get_landmark_coords(results.face_landmarks, idx, image_shape)
                       for idx in face_points_ids]
        
        if not all(face_coords):
            return False

        # Definir rectángulo de la cara
        x_coords = [pt[0] for pt in face_coords]
        y_coords = [pt[1] for pt in face_coords]
        left, right = min(x_coords) - 20, max(x_coords) + 20
        top, bottom = min(y_coords) - 20, max(y_coords) + 20

        # Verificar ambas manos
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                for idx in [0, 8]:  # muñeca y punta índice
                    point = get_landmark_coords(hand_landmarks, idx, image_shape)
                    if point:
                        x, y = point
                        if left <= x <= right and top <= y <= bottom:
                            return True
        return False


class TouchHeadGesture(BaseGesture):
    def __init__(self):
        super().__init__("tocarse_cabeza", 0.6)
        self.priority = 5

    def check(self, results, image_shape):
        head_top = get_landmark_coords(results.pose_landmarks, 0, image_shape)
        
        if not head_top:
            return False
        
        # Verificar ambas manos
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                wrist = get_landmark_coords(hand_landmarks, 0, image_shape)
                if wrist and calculate_distance(wrist, head_top) < 120:
                    return True
        
        return False

# =============================================================================
# GESTOS FACIALES
# =============================================================================

class HeadTiltGesture(BaseGesture):
    def __init__(self):
        super().__init__("cabeza_inclinada", 0.5)
        self.priority = 9
    
    def check(self, results, image_shape):
        left_eye = get_landmark_coords(results.face_landmarks, 33, image_shape)
        right_eye = get_landmark_coords(results.face_landmarks, 263, image_shape)
        
        if not all([left_eye, right_eye]):
            return False
            
        # Calcular inclinación basada en la posición de los ojos
        eye_slope = abs((right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0] + 0.001))
        return eye_slope > 0.3

# =============================================================================
# GESTOS PERSONALIZADOS - MEJORADOS PARA AMBAS MANOS
# =============================================================================

class ThumbsUpGesture(BaseGesture):
    def __init__(self):
        super().__init__("pulgar_arriba", 0.7)
    
    def check(self, results, image_shape):
        # Verificar ambas manos
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                thumb_tip = get_landmark_coords(hand_landmarks, 4, image_shape)
                index_tip = get_landmark_coords(hand_landmarks, 8, image_shape)
                wrist = get_landmark_coords(hand_landmarks, 0, image_shape)
                
                if all([thumb_tip, index_tip, wrist]):
                    # Pulgar extendido hacia arriba respecto a la muñeca
                    thumb_above_wrist = thumb_tip[1] < wrist[1] - 50
                    # Pulgar por encima del índice
                    thumb_above_index = thumb_tip[1] < index_tip[1] - 20
                    
                    if thumb_above_wrist and thumb_above_index:
                        return True
        
        return False

class PointingGesture(BaseGesture):
    def __init__(self):
        super().__init__("señalar", 0.6)
    
    def check(self, results, image_shape):
        # Verificar ambas manos
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                index_tip = get_landmark_coords(hand_landmarks, 8, image_shape)
                index_mcp = get_landmark_coords(hand_landmarks, 5, image_shape)
                middle_tip = get_landmark_coords(hand_landmarks, 12, image_shape)
                
                if all([index_tip, index_mcp, middle_tip]):
                    # Índice extendido y otros dedos doblados
                    index_extended = abs(index_tip[0] - index_mcp[0]) > 60
                    middle_bent = middle_tip[1] > index_mcp[1] + 30
                    
                    if index_extended and middle_bent:
                        return True
        
        return False

class PeaceSignGesture(BaseGesture):
    def __init__(self):
        super().__init__("paz", 0.7)
    
    def check(self, results, image_shape):
        # Verificar ambas manos
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                index_tip = get_landmark_coords(hand_landmarks, 8, image_shape)
                middle_tip = get_landmark_coords(hand_landmarks, 12, image_shape)
                ring_tip = get_landmark_coords(hand_landmarks, 16, image_shape)
                pinky_tip = get_landmark_coords(hand_landmarks, 20, image_shape)
                wrist = get_landmark_coords(hand_landmarks, 0, image_shape)
                
                if all([index_tip, middle_tip, ring_tip, pinky_tip, wrist]):
                    # Índice y medio extendidos
                    index_extended = index_tip[1] < wrist[1] - 50
                    middle_extended = middle_tip[1] < wrist[1] - 50
                    # Anular y meñique doblados
                    ring_bent = ring_tip[1] > wrist[1] - 20
                    pinky_bent = pinky_tip[1] > wrist[1] - 20
                    
                    if index_extended and middle_extended and ring_bent and pinky_bent:
                        return True
        
        return False

class HandsTogetherGesture(BaseGesture):
    def __init__(self):
        super().__init__("manos_juntas", 0.6)
    
    def check(self, results, image_shape):
        left_wrist = get_landmark_coords(results.pose_landmarks, 15, image_shape)
        right_wrist = get_landmark_coords(results.pose_landmarks, 16, image_shape)
        
        if left_wrist and right_wrist:
            return calculate_distance(left_wrist, right_wrist) < 100
        
        return False
    
    
class LegShakeGesture(BaseGesture):
    def __init__(self):
        super().__init__("piernas_inquietas", 0.6)
        from collections import deque
        self.left_foot_y = deque(maxlen=20)
        self.right_foot_y = deque(maxlen=20)
        self.last_state = False

    def check(self, results, image_shape):
        left_foot = get_landmark_coords(results.pose_landmarks, 27, image_shape)
        right_foot = get_landmark_coords(results.pose_landmarks, 28, image_shape)

        if left_foot:
            self.left_foot_y.append(left_foot[1])
        if right_foot:
            self.right_foot_y.append(right_foot[1])

        if len(self.left_foot_y) < 10 and len(self.right_foot_y) < 10:
            return False

        def detect_shake(positions):
            if len(positions) < 10:
                return False
            avg = np.mean(positions)
            amp = max(positions) - min(positions)
            return amp > 30 and np.std(positions) > 10  # más sensible

        left_moving = detect_shake(self.left_foot_y)
        right_moving = detect_shake(self.right_foot_y)

        self.last_state = left_moving or right_moving
        return self.last_state
    
class HandsOnHipsGesture(BaseGesture):
    def __init__(self):
        super().__init__("manos_en_caderas", 0.6)
    
    def check(self, results, image_shape):
        lw = get_landmark_coords(results.pose_landmarks, 15, image_shape)
        rw = get_landmark_coords(results.pose_landmarks, 16, image_shape)
        lh = get_landmark_coords(results.pose_landmarks, 23, image_shape)
        rh = get_landmark_coords(results.pose_landmarks, 24, image_shape)
        if not all([lw, rw, lh, rh]):
            return False
        return calculate_distance(lw, lh) < 100 and calculate_distance(rw, rh) < 100

# =============================================================================
# LISTA DE TODOS LOS GESTOS DISPONIBLES
# =============================================================================

DEFAULT_GESTURES = [
    # Gestos corporales
    CrossedArmsGesture(),
    OpenArmsGesture(),
    HandsOnHipsGesture(),
    HandsTogetherGesture(),
    
    # Gestos de manos y cara - CORREGIDOS
    ScratchNeckGesture(),
    BiteNailsGesture(),
    HandsFaceGesture(),
    TouchHeadGesture(),
    
    # Gestos faciales
    HeadTiltGesture(),
    
    # Gestos personalizados - MEJORADOS
    ThumbsUpGesture(),
    PointingGesture(),
    PeaceSignGesture(),

    #LegShakeGesture(),
]