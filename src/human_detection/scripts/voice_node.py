#!/usr/bin/env python3
import json
import re
import time
from pathlib import Path

import numpy as np
import rospy
import sounddevice as sd
import torch
import whisper
from std_msgs.msg import String
from transformers import (
    pipeline,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)

SAMPLE_RATE = 16000
DURATION = 4
MICROPHONE_ID = 13
LANGUAGE = "es"
RULES_FILE = "emociones.json"
PUBLISH_TOPIC = "/voice_emotion"


def load_rules(filepath):
    script_dir = Path(__file__).resolve().parent
    path = Path(filepath)
    if not path.is_absolute():
        path = script_dir / filepath

    if not path.exists():
        print(f"⚠️ No existe el fichero: {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"⚠️ JSON inválido: se esperaba un objeto/dict y llegó {type(data)}")
            return {}

        return {
            key.lower(): [phrase.lower() for phrase in values if isinstance(phrase, str)]
            for key, values in data.items()
            if isinstance(values, list)
        }

    except json.JSONDecodeError as e:
        print(f"❌ Error JSON en {path}: {e}")
        return {}


def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def find_matches(text, phrases):
    return [p for p in phrases if p in text.lower()]


class VoiceEmotionNode:
    def __init__(self):
        rospy.init_node("voice_emotion_node", anonymous=True)
        self.pub = rospy.Publisher(PUBLISH_TOPIC, String, queue_size=10)

        self.rules = load_rules(RULES_FILE)
        self.explicit_anger = self.rules.get("anger", [])
        self.explicit_joy = self.rules.get("joy", [])
        self.explicit_sadness = self.rules.get("sadness", [])
        self.explicit_fear = self.rules.get("fear", [])
        self.explicit_neutral = self.rules.get("neutral", [])
        self.explicit_surprise = self.rules.get("surprise", [])

        self.text_emotion_model = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=None,
        )

        self.whisper_model = whisper.load_model("base")

        voice_model_name = "superb/wav2vec2-base-superb-er"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(voice_model_name)
        self.voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(voice_model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.voice_model.to(self.device)
        self.voice_model.eval()

        self.rate = rospy.Rate(1.0 / max(DURATION, 1))
        rospy.loginfo("VoiceEmotionNode listo")

    def record_audio(self):
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=MICROPHONE_ID,
        )
        sd.wait()
        return np.squeeze(audio).astype(np.float32)

    def transcribe(self, audio):
        result = self.whisper_model.transcribe(audio, fp16=False, language=LANGUAGE)
        return clean_text(result.get("text", ""))

    def predict_voice_emotion(self, audio):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.voice_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        top_idx = int(torch.argmax(probs).item())
        label = self.voice_model.config.id2label[top_idx].lower()
        score = float(probs[top_idx])

        label_map = {
            "neu": "neutral",
            "ang": "angry",
            "hap": "happy",
            "sad": "sad",
            "fea": "fear",
            "dis": "disgust",
        }
        return label_map.get(label, label), score

    def analyze_text(self, text):
        if not text or len(text.split()) < 2:
            return "neutral", 0.0, []

        raw = self.text_emotion_model(text)
        scores = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
        scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        top = scores[0]

        mapping = {
            "anger": "angry",
            "joy": "happy",
            "surprise": "surprised",
            "sadness": "sad",
            "fear": "fear",
            "neutral": "neutral",
            "disgust": "angry",
        }
        return mapping.get(top["label"].lower(), "neutral"), float(top["score"]), scores

    def combine_emotions(self, voice_emotion, text_emotion, text):
        v_label, v_score = voice_emotion
        t_label, t_score, _ = text_emotion
        text_lower = text.lower()

        if find_matches(text_lower, self.explicit_anger):
            return "angry"
        if find_matches(text_lower, self.explicit_joy):
            return "happy"
        if find_matches(text_lower, self.explicit_sadness):
            return "sad"
        if find_matches(text_lower, self.explicit_fear):
            return "fear"
        if find_matches(text_lower, self.explicit_surprise):
            return "surprised"
        if find_matches(text_lower, self.explicit_neutral):
            return "neutral"

        if t_label == "angry" and t_score >= 0.40:
            return "angry"
        if t_label == "happy" and t_score >= 0.40:
            return "happy" if v_label in ["happy", "neutral"] else "mixed"
        if t_label == "sad" and t_score >= 0.40:
            return "sad"
        if t_label == "fear" and t_score >= 0.40:
            return "fear"
        if t_label == "surprised" and t_score >= 0.40:
            return "surprised"
        if t_label == "neutral":
            return v_label
        if v_score >= 0.55:
            return v_label
        return t_label

    def run(self):
        while not rospy.is_shutdown():
            try:
                audio = self.record_audio()
                text = self.transcribe(audio)
                voice_em = self.predict_voice_emotion(audio)
                text_em = self.analyze_text(text)
                final_emotion = self.combine_emotions(voice_em, text_em, text)

                payload = {
                    "source": "voice",
                    "text": text,
                    "voice_emotion": voice_em[0],
                    "voice_score": voice_em[1],
                    "text_emotion": text_em[0],
                    "text_score": text_em[1],
                    "final_emotion": final_emotion,
                    "stamp": rospy.Time.now().to_sec(),
                }

                msg = String()
                msg.data = json.dumps(payload, ensure_ascii=False)
                self.pub.publish(msg)

                rospy.loginfo(f"Voice: {final_emotion} | Texto: {text}")
            except Exception as e:
                rospy.logwarn(f"Voice node error: {e}")

            self.rate.sleep()


if __name__ == "__main__":
    try:
        VoiceEmotionNode().run()
    except rospy.ROSInterruptException:
        pass