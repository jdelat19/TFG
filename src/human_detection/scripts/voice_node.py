#!/usr/bin/env python3
"""Nodo de emoción por voz.

Graba fragmentos de DURATION segundos del micrófono y, para cada uno:
  1. Transcribe el audio con Whisper ("base", en español).
  2. Clasifica la emoción del texto con RoBERTa (TweetEval).
  3. Clasifica la emoción del tono con wav2vec 2.0 (IEMOCAP).
  4. Combina las dos con las frases clave de config/emociones.json
     (reglas en emotions.combine_voice_emotions).

Publica:
  /voice_emotion (std_msgs/String, JSON)
     {"text", "voice_emotion", "voice_score", "text_emotion", "text_score",
      "final_emotion", "stamp"}

Parámetros:
  ~microphone  índice de sounddevice (si no existe se usa el de por defecto)

Si el fragmento es silencio (sin texto) no se publica nada, para que se
mantenga la última emoción.
"""
import json
import os
import sys

import numpy as np
import rospy
import sounddevice as sd
import torch
import whisper
from std_msgs.msg import String
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import emotions

SAMPLE_RATE = 16000          # Hz, lo que esperan Whisper y wav2vec 2.0
DURATION = 4                 # segundos de cada fragmento
MICROPHONE_ID = 14           # valor por defecto (el mismo que en los launch)
LANGUAGE = "es"
RULES_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "emociones.json")
TEXT_MODEL = "cardiffnlp/twitter-roberta-base-emotion"
VOICE_MODEL = "superb/wav2vec2-base-superb-er"


class VoiceEmotionNode:
    def __init__(self):
        rospy.init_node("voice_emotion_node", anonymous=True)
        self.pub = rospy.Publisher("/voice_emotion", String, queue_size=10)
        self.rules = emotions.load_voice_rules(RULES_FILE)

        # Los tres modelos se descargan solos la primera vez (~1 GB en ~/.cache)
        self.text_model = pipeline("text-classification", model=TEXT_MODEL, top_k=None)
        self.whisper_model = whisper.load_model("base")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(VOICE_MODEL)
        self.voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(VOICE_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.voice_model.to(self.device)
        self.voice_model.eval()

        self.microphone = self.pick_microphone(rospy.get_param("~microphone", MICROPHONE_ID))
        self.rate = rospy.Rate(1.0 / DURATION)
        rospy.loginfo("VoiceEmotionNode listo")

    @staticmethod
    def pick_microphone(wanted):
        """Usa el micrófono pedido si existe; si no, el de entrada por defecto."""
        try:
            devices = sd.query_devices()
            if 0 <= int(wanted) < len(devices) and devices[int(wanted)]["max_input_channels"] > 0:
                return int(wanted)
            default = sd.default.device[0]
            if default is None or default < 0:
                default = next(i for i, d in enumerate(devices) if d["max_input_channels"] > 0)
            rospy.logwarn("Micrófono %s no válido; uso '%s' (%d)", wanted, devices[default]["name"], default)
            return int(default)
        except Exception as e:
            rospy.logwarn("No se pudo comprobar el micrófono (%s); uso %s", e, wanted)
            return wanted

    def record_audio(self):
        """Graba DURATION segundos en mono a 16 kHz (bloquea mientras graba)."""
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                       dtype="float32", device=self.microphone)
        sd.wait()
        return np.squeeze(audio).astype(np.float32)

    def transcribe(self, audio):
        result = self.whisper_model.transcribe(audio, fp16=False, language=LANGUAGE)
        return emotions.clean_text(result.get("text", ""))

    def predict_voice_emotion(self, audio):
        """(emoción, confianza) según el tono de voz."""
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                        padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(self.voice_model(**inputs).logits, dim=1)[0]
        top = int(torch.argmax(probs).item())
        label = self.voice_model.config.id2label[top].lower()
        return emotions.VOICE_LABELS.get(label, label), float(probs[top])

    def analyze_text(self, text):
        """(emoción, confianza) según el texto; neutral si tiene menos de 2 palabras."""
        if not text or len(text.split()) < 2:
            return "neutral", 0.0
        raw = self.text_model(text)
        scores = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
        top = max(scores, key=lambda x: x["score"])
        return emotions.TEXT_LABELS.get(top["label"].lower(), "neutral"), float(top["score"])

    def run(self):
        while not rospy.is_shutdown():
            try:
                audio = self.record_audio()
                text = self.transcribe(audio)
                if not text:
                    # Silencio o ruido: no se publica para que se mantenga la emoción anterior
                    rospy.loginfo("Voz: silencio (se mantiene la emoción)")
                    self.rate.sleep()
                    continue
                voice_em = self.predict_voice_emotion(audio)
                text_em = self.analyze_text(text)
                final = emotions.combine_voice_emotions(voice_em, text_em, text, self.rules)
                payload = {
                    "source": "voice",
                    "text": text,
                    "voice_emotion": voice_em[0],
                    "voice_score": voice_em[1],
                    "text_emotion": text_em[0],
                    "text_score": text_em[1],
                    "final_emotion": final,
                    "stamp": rospy.Time.now().to_sec(),
                }
                self.pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                rospy.loginfo(f"Voz: {final} | Texto: {text}")
            except Exception as e:
                rospy.logwarn(f"Error en el nodo de voz: {e}")
            self.rate.sleep()


if __name__ == "__main__":
    try:
        VoiceEmotionNode().run()
    except rospy.ROSInterruptException:
        pass
