from abc import ABC, abstractmethod
from collections import deque

from utils import calculate_distance, get_landmark_coords

class BaseGesture(ABC):
    def __init__(self, name: str):
        self.name = name
        self.priority = 1
        
    @abstractmethod
    def check(self, results, image_shape: tuple) -> bool:
        pass


# =============================================================================
# GESTOS CORPORALES
# =============================================================================

class CrossedArmsGesture(BaseGesture):
    def __init__(self):
        super().__init__("cruzar_brazos")
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
        super().__init__("brazos_abiertos")
    
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

# =============================================================================
# GESTOS DE MANOS Y CARA
# =============================================================================

class ScratchNeckGesture(BaseGesture):
    def __init__(self):
        super().__init__("rascarse_cuello")
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

class BiteNailsGesture(BaseGesture):
    def __init__(self):
        super().__init__("morderse_unas")
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

        finger_tips = [4, 8, 12, 16, 20]

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


class HandsFaceGesture(BaseGesture):
    def __init__(self):
        super().__init__("manos_en_cara")
        self.priority = 6
    
    def check(self, results, image_shape):
        if not results.face_landmarks:
            return False

        face_points_ids = [10, 152, 234, 454]  # frente, mentón, mejillas
        face_coords = [get_landmark_coords(results.face_landmarks, idx, image_shape)
                       for idx in face_points_ids]
        
        if not all(face_coords):
            return False

        x_coords = [pt[0] for pt in face_coords]
        y_coords = [pt[1] for pt in face_coords]
        left, right = min(x_coords) - 20, max(x_coords) + 20
        top, bottom = min(y_coords) - 20, max(y_coords) + 20

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
        super().__init__("tocarse_cabeza")
        self.priority = 5

    def check(self, results, image_shape):
        head_top = get_landmark_coords(results.pose_landmarks, 0, image_shape)
        
        if not head_top:
            return False
        
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
        super().__init__("cabeza_inclinada")
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
# GESTOS PERSONALIZADOS
# =============================================================================

class ThumbsUpGesture(BaseGesture):
    def __init__(self):
        super().__init__("pulgar_arriba")
    
    def check(self, results, image_shape):
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
        super().__init__("señalar")
    
    def check(self, results, image_shape):
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
        super().__init__("paz")
    
    def check(self, results, image_shape):
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
        super().__init__("manos_juntas")
    
    def check(self, results, image_shape):
        left_wrist = get_landmark_coords(results.pose_landmarks, 15, image_shape)
        right_wrist = get_landmark_coords(results.pose_landmarks, 16, image_shape)
        
        if left_wrist and right_wrist:
            return calculate_distance(left_wrist, right_wrist) < 100
        
        return False
    
class HandsOnHipsGesture(BaseGesture):
    def __init__(self):
        super().__init__("manos_en_caderas")
    
    def check(self, results, image_shape):
        lw = get_landmark_coords(results.pose_landmarks, 15, image_shape)
        rw = get_landmark_coords(results.pose_landmarks, 16, image_shape)
        lh = get_landmark_coords(results.pose_landmarks, 23, image_shape)
        rh = get_landmark_coords(results.pose_landmarks, 24, image_shape)
        if not all([lw, rw, lh, rh]):
            return False
        return calculate_distance(lw, lh) < 100 and calculate_distance(rw, rh) < 100

# =============================================================================
# GESTOS POR FORMA DE LA MANO
# =============================================================================
# Índices de la mano en MediaPipe: 0 muñeca, pulgar 1-4, índice 5-8,
# corazón 9-12, anular 13-16, meñique 17-20 (MCP, PIP, DIP, punta).
# Se comparan distancias entre landmarks, así que no depende de si la mano
# está girada.

FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}


def hand_points(hand_landmarks, image_shape):
    """Los 21 puntos de la mano en píxeles, o None si falta alguno."""
    if not hand_landmarks:
        return None
    pts = [get_landmark_coords(hand_landmarks, i, image_shape) for i in range(21)]
    return pts if all(pts) else None


def hand_size(pts):
    """Distancia muñeca - base del dedo corazón, sirve de escala de la mano."""
    return max(calculate_distance(pts[0], pts[9]), 1.0)


def finger_states(pts):
    """Qué dedos están extendidos.

    Un dedo está extendido si su punta está más lejos de la muñeca que su
    articulación central. El pulgar, si su punta está más lejos de la base
    del meñique que su articulación IP.
    """
    states = {
        name: calculate_distance(pts[0], pts[tip]) > 1.1 * calculate_distance(pts[0], pts[pip])
        for name, (pip, tip) in FINGERS.items()
    }
    states["thumb"] = calculate_distance(pts[4], pts[17]) > 1.1 * calculate_distance(pts[3], pts[17])
    return states


class HandShapeGesture(BaseGesture):
    """Gesto que depende solo de la forma de una mano (cualquiera de las dos)."""

    def __init__(self, name, priority=3):
        super().__init__(name)
        self.priority = priority

    def check(self, results, image_shape):
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            pts = hand_points(hand_landmarks, image_shape)
            if pts and self.check_hand(pts, finger_states(pts)):
                return True
        return False

    def check_hand(self, pts, fingers):
        raise NotImplementedError


class FistGesture(HandShapeGesture):
    def __init__(self):
        super().__init__("puño", priority=2)

    def check_hand(self, pts, fingers):
        return not any(fingers.values())


class OpenHandGesture(HandShapeGesture):
    def __init__(self):
        super().__init__("mano_abierta", priority=2)

    def check_hand(self, pts, fingers):
        return all(fingers.values())


class OkGesture(HandShapeGesture):
    def __init__(self):
        super().__init__("ok")

    def check_hand(self, pts, fingers):
        thumb_index_touch = calculate_distance(pts[4], pts[8]) < 0.35 * hand_size(pts)
        return thumb_index_touch and fingers["middle"] and fingers["ring"] and fingers["pinky"]


class RockGesture(HandShapeGesture):
    def __init__(self):
        super().__init__("cuernos")

    def check_hand(self, pts, fingers):
        return (fingers["index"] and fingers["pinky"]
                and not fingers["middle"] and not fingers["ring"])


class CallMeGesture(HandShapeGesture):
    def __init__(self):
        super().__init__("llamame")

    def check_hand(self, pts, fingers):
        return (fingers["thumb"] and fingers["pinky"]
                and not fingers["index"] and not fingers["middle"] and not fingers["ring"])


class ThumbsDownGesture(HandShapeGesture):
    def __init__(self):
        super().__init__("pulgar_abajo")

    def check_hand(self, pts, fingers):
        others_folded = not any(fingers[f] for f in FINGERS)
        # En la imagen la y crece hacia abajo
        lowest_knuckle = max(pts[i][1] for i in (5, 9, 13, 17))
        thumb_down = pts[4][1] > lowest_knuckle + 0.3 * hand_size(pts) and pts[4][1] > pts[3][1]
        return fingers["thumb"] and others_folded and thumb_down


class WaveGesture(BaseGesture):
    """Mano abierta moviéndose de lado a lado."""

    def __init__(self, history=20):
        super().__init__("saludar")
        self.priority = 4
        self.wrist_x = {"left": deque(maxlen=history), "right": deque(maxlen=history)}

    def check(self, results, image_shape):
        waving = False
        for side, hand_landmarks in [("left", results.left_hand_landmarks),
                                     ("right", results.right_hand_landmarks)]:
            pts = hand_points(hand_landmarks, image_shape)
            if not pts or not all(finger_states(pts).values()):
                self.wrist_x[side].clear()
                continue
            self.wrist_x[side].append(pts[0][0])
            waving = waving or self.is_waving(self.wrist_x[side], hand_size(pts))
        return waving

    @staticmethod
    def is_waving(xs, size):
        if len(xs) < 10 or max(xs) - min(xs) < 0.8 * size:
            return False
        # Contar cambios de sentido, ignorando movimientos pequeños
        direction_changes, last_sign = 0, 0
        for a, b in zip(xs, list(xs)[1:]):
            if abs(b - a) < 0.1 * size:
                continue
            sign = 1 if b > a else -1
            if last_sign and sign != last_sign:
                direction_changes += 1
            last_sign = sign
        return direction_changes >= 2

# =============================================================================
# LISTA DE TODOS LOS GESTOS DISPONIBLES
# =============================================================================

DEFAULT_GESTURES = [
    # Gestos corporales
    CrossedArmsGesture(),
    OpenArmsGesture(),
    HandsOnHipsGesture(),
    HandsTogetherGesture(),
    
    # Gestos de manos y cara
    ScratchNeckGesture(),
    BiteNailsGesture(),
    HandsFaceGesture(),
    TouchHeadGesture(),
    
    # Gestos faciales
    HeadTiltGesture(),
    
    # Gestos personalizados
    ThumbsUpGesture(),
    PointingGesture(),
    PeaceSignGesture(),

    # Gestos por forma de la mano
    FistGesture(),
    OpenHandGesture(),
    WaveGesture(),
    OkGesture(),
    RockGesture(),
    CallMeGesture(),
    ThumbsDownGesture(),
]