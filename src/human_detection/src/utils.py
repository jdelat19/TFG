import numpy as np
from typing import Tuple, Optional

def calculate_distance(point1: Tuple, point2: Tuple) -> float:
    """Calcula la distancia entre dos puntos"""
    if point1 is None or point2 is None:
        return float('inf')
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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