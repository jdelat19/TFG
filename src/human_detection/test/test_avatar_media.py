#!/usr/bin/env python3
"""Pruebas de la elección de imagen o vídeo del avatar con la carpeta media real."""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from avatar_media import media_path, resolve_emotion

MEDIA = os.path.join(os.path.dirname(__file__), "..", "media")


def name(path):
    return os.path.basename(path)


def test_voice_only_counts_in_voice_modes():
    payload = {"emotion": "Feliz", "voice_final_emotion": "triste"}
    assert resolve_emotion(payload, 1) == "feliz"
    assert resolve_emotion(payload, 2) == "triste"
    assert resolve_emotion(payload, 3) == "triste"
    assert resolve_emotion(payload, 4) == "feliz"
    assert resolve_emotion({"emotion": "Feliz", "voice_final_emotion": "neutral"}, 2) == "feliz"


def test_images_in_modes_1_and_2_videos_in_3_and_4():
    payload = {"emotion": "Triste"}
    assert media_path(MEDIA, 1, payload) == (os.path.join(MEDIA, "imagenes", "triste.png"), True)
    path, is_image = media_path(MEDIA, 3, payload)
    assert name(path) == "triste.mp4" and not is_image


def test_mode_4_uses_gesture_video():
    payload = {"emotion": "Neutral", "gesture": "rascarse_cuello"}
    assert name(media_path(MEDIA, 4, payload)[0]) == "rascarse_cuello.mp4"


def test_mode_4_falls_back_to_emotion_then_default():
    assert name(media_path(MEDIA, 4, {"emotion": "Feliz", "gesture": "paz"})[0]) == "feliz.mp4"
    assert name(media_path(MEDIA, 4, {"emotion": "Disgusto", "gesture": "Ninguno"})[0]) == "no detectado.mp4"


def test_missing_image_uses_default():
    # No hay imagen de miedo, sí vídeo
    assert name(media_path(MEDIA, 1, {"emotion": "Miedo"})[0]) == "no detectado.png"
    assert name(media_path(MEDIA, 3, {"emotion": "Miedo"})[0]) == "miedo.mp4"
