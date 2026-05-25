#!/usr/bin/env python3
import json
import os

import cv2
import rospy
from std_msgs.msg import String


class AvatarNode:
    def __init__(self):
        rospy.init_node("avatar_node")

        self.mode = rospy.get_param("~mode", 1)
        self.base_path = rospy.get_param(
            "~base_path",
            os.path.expanduser("~/Escritorio/TFG/src/human_detection/media"),
        )

        self.current_video = None
        self.cap = None
        self.last_payload = {
            "gesture": "Ninguno",
            "emotion": "neutral",
            "voice_final_emotion": "neutral",
        }

        rospy.Subscriber("/human_state", String, self.callback)
        rospy.loginfo(f"Avatar node iniciado en modo {self.mode}")

    def callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.last_payload = data
        except Exception:
            return

        gesture = self.last_payload.get("gesture", "Ninguno")
        emotion = self.resolve_emotion(self.last_payload)
        key = self.get_key(gesture, emotion)

        if self.mode in [1, 3]:
            self.show_image(key)
        else:
            self.play_video(key)

    def resolve_emotion(self, data):
        voice_emotion = (data.get("voice_final_emotion") or "").lower()
        body_emotion = (data.get("emotion") or "").lower()

        if voice_emotion and voice_emotion != "neutral":
            return voice_emotion

        mapping = {
            "enojo": "angry",
            "feliz": "happy",
            "triste": "sad",
            "sorpresa": "surprised",
            "neutral": "neutral",
            "no detectado": "no detectado",
        }
        return mapping.get(body_emotion, body_emotion or "neutral")

    def get_key(self, gesture, emotion):
        gesture = (gesture or "Ninguno").lower()
        emotion = (emotion or "neutral").lower()

        if self.mode in [1, 2]:
            return emotion
        return f"{gesture}_{emotion}"

    def show_image(self, key):
        path = os.path.join(self.base_path, "imagenes", f"{key}.png")
        if not os.path.exists(path):
            path = os.path.join(self.base_path, "imagenes", "no detectado.png")
        img = cv2.imread(path)
        if img is None:
            return
        cv2.imshow("Avatar", img)
        cv2.waitKey(1)

    def play_video(self, key):
        path = os.path.join(self.base_path, "videos", f"{key}.mp4")
        if not os.path.exists(path):
            path = os.path.join(self.base_path, "videos", "no detectado.mp4")

        if self.current_video != path:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.current_video = path

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return

        cv2.imshow("Avatar", frame)
        cv2.waitKey(30)


if __name__ == "__main__":
    try:
        AvatarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass