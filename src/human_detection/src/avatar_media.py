"""Elección de la imagen o el vídeo que muestra el avatar (2D y 3D).

Modos:
  1  Cara            -> imagen de la emoción
  2  Voz o cara      -> imagen de la emoción
  3  Voz o cara      -> vídeo de la emoción
  4  Cara + gesto    -> vídeo del gesto (o de la emoción si no hay vídeo del gesto)
"""
import os

IMAGE_MODES = (1, 2)
VOICE_MODES = (2, 3)
FALLBACK = "no detectado"


def resolve_emotion(payload, mode):
    """Emoción a mostrar: la de la voz si el modo la usa y no es neutral; si no, la de la cara."""
    face = (payload.get("emotion") or "neutral").lower()
    voice = (payload.get("voice_final_emotion") or "").lower()
    if mode in VOICE_MODES and voice and voice != "neutral":
        return voice
    return face


def candidate_names(payload, mode):
    """Nombres de fichero a probar, del más específico al más general."""
    emotion = resolve_emotion(payload, mode)
    gesture = (payload.get("gesture") or "ninguno").lower()
    if mode == 4 and gesture != "ninguno":
        return [f"{gesture}_{emotion}", gesture, emotion, FALLBACK]
    return [emotion, FALLBACK]


def media_path(base_path, mode, payload):
    """Devuelve (ruta, es_imagen). La ruta es None si no existe ningún recurso."""
    is_image = mode in IMAGE_MODES
    folder, ext = ("imagenes", ".png") if is_image else ("videos", ".mp4")
    for name in candidate_names(payload, mode):
        path = os.path.join(base_path, folder, name + ext)
        if os.path.exists(path):
            return path, is_image
    return None, is_image
