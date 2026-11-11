#!/usr/bin/env python3
"""Pruebas de emotions.py: avatar (con la carpeta media real) y reglas de la voz."""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import emotions
from emotions import combine_voice_emotions, media_path, normalize_emotion, resolve_emotion

HERE = os.path.dirname(__file__)
MEDIA = os.path.join(HERE, "..", "media")
RULES = os.path.join(HERE, "..", "config", "emociones.json")


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
    # En el modo 4 la emoción facial está desactivada
    assert name(media_path(MEDIA, 4, {"emotion": "Desactivado", "gesture": "Ninguno"})[0]) == "no detectado.mp4"


def test_disgust_video_exists():
    assert name(media_path(MEDIA, 3, {"emotion": "Disgusto"})[0]) == "disgusto.mp4"


def test_missing_image_uses_default():
    # No hay imagen de miedo, sí vídeo
    assert name(media_path(MEDIA, 1, {"emotion": "Miedo"})[0]) == "no detectado.png"
    assert name(media_path(MEDIA, 3, {"emotion": "Miedo"})[0]) == "miedo.mp4"


def test_every_emotion_has_an_avatar_video():
    for emotion in emotions.EMOTIONS:
        assert name(media_path(MEDIA, 3, {"emotion": emotion})[0]) == emotion + ".mp4"


def test_normalize_emotion():
    assert normalize_emotion(" Feliz ") == "feliz"
    assert normalize_emotion("ENOJO") == "enojo"
    for label in ("No detectado", "Desactivado", "mixta", "", None):
        assert normalize_emotion(label) is None


def test_every_emotion_has_a_behavior():
    assert set(emotions.BEHAVIORS.values()) == {"approach", "comfort", "hide", "retreat", "spin", "idle"}
    assert emotions.BEHAVIORS["enojo"] == emotions.BEHAVIORS["disgusto"] == "hide"


# ---------------------------------------------------------------------------
# Voz
# ---------------------------------------------------------------------------
def test_voice_rules_file_has_all_emotions():
    rules = emotions.load_voice_rules(RULES)
    assert {k for k, _ in emotions.RULE_KEYS} <= set(rules)
    assert all(rules[k] for k, _ in emotions.RULE_KEYS)
    assert all(p == p.lower() for phrases in rules.values() for p in phrases)


def test_missing_rules_file_is_empty():
    assert emotions.load_voice_rules(os.path.join(HERE, "no_existe.json")) == {}


def test_key_phrase_beats_the_models():
    rules = {"fear": ["tengo miedo"], "anger": ["estoy harto"]}
    # Aunque los dos modelos digan feliz, la frase clave manda
    assert combine_voice_emotions(("feliz", 0.9), ("feliz", 0.9), "Tengo miedo de esto", rules) == "miedo"
    assert combine_voice_emotions(("neutral", 0.9), ("neutral", 0.0), "ya estoy harto", rules) == "enojo"


def test_confident_text_model_decides():
    assert combine_voice_emotions(("neutral", 0.9), ("triste", 0.6), "hoy ha sido un mal día", {}) == "triste"
    assert combine_voice_emotions(("triste", 0.9), ("enojo", 0.5), "qué rabia", {}) == "enojo"


def test_happy_text_with_other_tone_is_mixed():
    assert combine_voice_emotions(("feliz", 0.8), ("feliz", 0.7), "qué bien", {}) == "feliz"
    assert combine_voice_emotions(("triste", 0.8), ("feliz", 0.7), "qué bien", {}) == "mixta"


def test_neutral_or_unsure_text_uses_the_tone():
    assert combine_voice_emotions(("enojo", 0.3), ("neutral", 0.9), "son las tres", {}) == "enojo"
    assert combine_voice_emotions(("triste", 0.7), ("feliz", 0.2), "vale", {}) == "triste"
    assert combine_voice_emotions(("triste", 0.4), ("feliz", 0.2), "vale", {}) == "feliz"


def test_clean_text():
    assert emotions.clean_text("  Hola   QUÉ  tal \n") == "hola qué tal"
