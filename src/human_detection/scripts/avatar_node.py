#!/usr/bin/env python3
import json
import os

import cv2
import rospy
from std_msgs.msg import String


class AvatarNode:
    def __init__(self):
        rospy.init_node("avatar_node")

        self.mode = int(rospy.get_param("~mode", 1))

        print("MODE =", self.mode)
        print("TYPE =", type(self.mode))

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
            self.last_payload = json.loads(msg.data)
        except Exception:
            pass

    def resolve_emotion(self, data):

        face_emotion = (
            data.get("emotion")
            or "neutral"
        ).lower()

        voice_emotion = (
            data.get("voice_final_emotion")
            or data.get("voiceemotion")
            or "neutral"
        ).lower()

        # Modos con voz
        if self.mode in [2, 3]:

            # Prioridad absoluta a la voz
            if voice_emotion != "neutral":
                return voice_emotion

            return face_emotion

        return face_emotion

    def get_key(self, gesture, emotion):

        gesture = (gesture or "Ninguno").lower()
        emotion = (emotion or "neutral").lower()

        # Modo 1 -> imágenes emociones
        if self.mode == 1:
            return emotion

        # Modo 2 -> voz+cara -> imágenes emociones
        if self.mode == 2:
            return emotion

        # Modo 3 -> voz+cara -> vídeos emociones
        if self.mode == 3:
            return emotion

        # Modo 4 -> vídeos de gestos
        if self.mode == 4:
            return gesture

        return emotion

    def show_image(self, key):

        path = os.path.join(
            self.base_path,
            "imagenes",
            f"{key}.png",
        )

        if not os.path.exists(path):
            path = os.path.join(
                self.base_path,
                "imagenes",
                "no detectado.png",
            )

        img = cv2.imread(path)

        if img is None:
            return

        cv2.imshow("Avatar", img)
        cv2.waitKey(1)

    def play_video(self, key):

        path = os.path.join(
            self.base_path,
            "videos",
            f"{key}.mp4",
        )

        if not os.path.exists(path):
            path = os.path.join(
                self.base_path,
                "videos",
                "no detectado.mp4",
            )

        if self.current_video != path:

            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(path)
            self.current_video = path

        if self.cap is None:
            return

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

        node = AvatarNode()
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():

            gesture = node.last_payload.get(
                "gesture",
                "Ninguno",
            )

            emotion = node.resolve_emotion(
                node.last_payload
            )

            key = node.get_key(
                gesture,
                emotion,
            )

            # Modo 1
            if node.mode == 1:
                node.show_image(key)

            # Modo 2
            elif node.mode == 2:
                node.show_image(key)

            # Modo 3
            elif node.mode == 3:
                node.play_video(key)

            # Modo 4
            elif node.mode == 4:
                node.play_video(key)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass

