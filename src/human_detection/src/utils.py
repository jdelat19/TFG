import numpy as np
from typing import Tuple, Optional

def calculate_distance(point1: Tuple, point2: Tuple) -> float:
    """Calcula la distancia entre dos puntos"""
    if point1 is None or point2 is None:
        return float('inf')
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_angle(a: Tuple, b: Tuple, c: Tuple) -> float:
    """Calcula el ángulo entre tres puntos"""
    if None in [a, b, c]:
        return 0.0
        
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1, 1))
    
    return np.degrees(angle)

def get_landmark_coords(landmarks, landmark_idx: int, image_shape: Tuple) -> Optional[Tuple]:
    """Obtiene coordenadas de un landmark específico"""
    if landmarks is None:
        return None
    try:
        landmark = landmarks.landmark[landmark_idx]
        h, w = image_shape
        return (int(landmark.x * w), int(landmark.y * h))
    except (IndexError, AttributeError):
        return None