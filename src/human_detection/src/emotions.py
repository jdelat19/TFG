"""Vocabulario de emociones y reglas compartidas por todos los nodos.

Este módulo no depende de ROS ni de ningún modelo, así que se puede probar
directamente con pytest. Reúne:

  1. Las emociones que entiende el sistema y el comportamiento del TurtleBot
     asociado a cada una (`BEHAVIORS`).
  2. La elección de imagen o vídeo del avatar (`media_path`).
  3. Las reglas para combinar la emoción del texto y la del tono de voz
     (`load_voice_rules`, `combine_voice_emotions`).

Las emociones se escriben siempre en minúsculas y en español: feliz, triste,
enojo, disgusto, miedo, sorpresa y neutral. "no detectado" significa que no
hay ninguna cara en la imagen.
"""
import json
import os
import re

# ---------------------------------------------------------------------------
# 1. Emociones y comportamiento del TurtleBot
# ---------------------------------------------------------------------------
# Emoción -> comportamiento (ver turtlebot_behaviors.py)
BEHAVIORS = {
    "feliz": "approach",     # se acerca a la persona
    "triste": "comfort",     # se acerca despacio y se queda a más distancia
    "enojo": "hide",         # se esconde debajo de una mesa
    "disgusto": "hide",
    "miedo": "retreat",      # retrocede
    "sorpresa": "spin",      # da una vuelta sobre sí mismo
    "neutral": "idle",       # se queda quieto mirando a la persona
}
EMOTIONS = tuple(BEHAVIORS)


def normalize_emotion(label):
    """Pasa una etiqueta ("Feliz", " ENOJO ") a minúsculas; None si no es una
    de las 7 emociones (por ejemplo "No detectado", "mixta" o "")."""
    if not label:
        return None
    label = str(label).strip().lower()
    return label if label in BEHAVIORS else None


# ---------------------------------------------------------------------------
# 2. Avatar: imagen o vídeo según el modo, la emoción y el gesto
# ---------------------------------------------------------------------------
#   Modo 1  Cara            -> imagen de la emoción
#   Modo 2  Voz o cara      -> imagen de la emoción
#   Modo 3  Voz o cara      -> vídeo de la emoción
#   Modo 4  Cara + gesto    -> vídeo del gesto (o de la emoción si no hay vídeo del gesto)
IMAGE_MODES = (1, 2)
VOICE_MODES = (2, 3)
FALLBACK = "no detectado"   # recurso que se muestra si no hay uno más específico


def resolve_emotion(payload, mode):
    """Emoción a mostrar: la de la voz si el modo la usa y no es neutral; si no, la de la cara.

    `payload` es el JSON de /human_state (claves "emotion" y "voice_final_emotion").
    """
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
    """Devuelve (ruta, es_imagen) del recurso a mostrar.

    Busca en `base_path`/imagenes/*.png (modos 1 y 2) o `base_path`/videos/*.mp4
    (modos 3 y 4). La ruta es None si no existe ningún candidato.
    """
    is_image = mode in IMAGE_MODES
    folder, ext = ("imagenes", ".png") if is_image else ("videos", ".mp4")
    for name in candidate_names(payload, mode):
        path = os.path.join(base_path, folder, name + ext)
        if os.path.exists(path):
            return path, is_image
    return None, is_image


# ---------------------------------------------------------------------------
# 3. Voz: combinar la emoción del texto y la del tono
# ---------------------------------------------------------------------------
# Clases del modelo de tono (superb/wav2vec2-base-superb-er, IEMOCAP)
VOICE_LABELS = {"neu": "neutral", "ang": "enojo", "hap": "feliz", "sad": "triste"}
# Clases del modelo de texto (cardiffnlp/twitter-roberta-base-emotion, TweetEval).
# "optimism" no corresponde a ninguna emoción del sistema: se toma como neutral
TEXT_LABELS = {"anger": "enojo", "joy": "feliz", "sadness": "triste", "optimism": "neutral"}
# Claves de config/emociones.json, en el orden en que se comprueban
RULE_KEYS = (("anger", "enojo"), ("joy", "feliz"), ("sadness", "triste"),
             ("fear", "miedo"), ("surprise", "sorpresa"), ("neutral", "neutral"))
TEXT_MIN_SCORE = 0.40    # confianza mínima del modelo de texto
VOICE_MIN_SCORE = 0.55   # confianza mínima del modelo de tono si el texto no decide


def load_voice_rules(path):
    """Lee config/emociones.json: {"anger": ["frase", ...], "joy": [...], ...}.

    Devuelve las frases en minúsculas, o {} si el fichero no existe o está mal.
    """
    if not os.path.exists(path):
        print(f"No existe el fichero de frases: {path}")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"No se puede leer {path}: {e}")
        return {}
    if not isinstance(data, dict):
        return {}
    return {key.lower(): [p.lower() for p in values if isinstance(p, str)]
            for key, values in data.items() if isinstance(values, list)}


def clean_text(text):
    """Minúsculas, sin espacios repetidos ni al principio o al final."""
    return re.sub(r"\s+", " ", text.strip().lower())


def combine_voice_emotions(voice, text_emotion, text, rules):
    """Emoción final de un fragmento de voz.

    voice:        (etiqueta, confianza) del modelo de tono
    text_emotion: (etiqueta, confianza) del modelo de texto
    text:         transcripción de Whisper
    rules:        frases clave de load_voice_rules

    Orden de prioridad:
      1. Una frase clave de emociones.json dentro del texto ("estoy harto" -> enojo).
      2. El modelo de texto si está seguro (>= 0.40). Si dice feliz pero el tono
         no es feliz ni neutral, la emoción es "mixta" (el avatar no la muestra).
      3. Si el texto es neutral, manda el tono.
      4. Si no, el tono si está seguro (>= 0.55) y, si tampoco, el texto.
    """
    v_label, v_score = voice
    t_label, t_score = text_emotion
    text = text.lower()

    for key, emotion in RULE_KEYS:
        if any(phrase in text for phrase in rules.get(key, [])):
            return emotion

    if t_score >= TEXT_MIN_SCORE and t_label in ("enojo", "triste", "miedo", "sorpresa"):
        return t_label
    if t_score >= TEXT_MIN_SCORE and t_label == "feliz":
        return "feliz" if v_label in ("feliz", "neutral") else "mixta"
    if t_label == "neutral":
        return v_label
    if v_score >= VOICE_MIN_SCORE:
        return v_label
    return t_label
