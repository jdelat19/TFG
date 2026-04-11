#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import cv2
import json
import os

class AvatarNode:
    def __init__(self):
        rospy.init_node("avatar_node")

        self.mode = rospy.get_param("~mode", 1)

        self.base_path = os.path.expanduser(
            "~/Escritorio/TFG/src/human_detection/media"
        )

        self.current_video = None
        self.cap = None

        rospy.Subscriber("/human_state", String, self.callback)

        print(f"🎭 Avatar node iniciado en modo {self.mode}")

    def callback(self, msg):
        data = json.loads(msg.data)

        gesture = data["gesture"]
        emotion = data["emotion"]

        key = self.get_key(gesture, emotion)

        if self.mode in [1, 3]:
            self.show_image(key)
        else:
            self.play_video(key)

    def get_key(self, gesture, emotion):
        if self.mode in [1, 2]:
            return emotion.lower()

        elif self.mode in [3, 4]:
            return f"{gesture}_{emotion}".lower()

    def show_image(self, key):
        path = os.path.join(self.base_path, "imagenes", f"{key}.png")

        if not os.path.exists(path):
            print(f"⚠️ Imagen no encontrada: {path}")
            return

        img = cv2.imread(path)
        cv2.imshow("Avatar", img)
        cv2.waitKey(1)

    def play_video(self, key):
        path = os.path.join(self.base_path, "videos", f"{key}.mp4")

        if not os.path.exists(path):
            print(f"⚠️ Video no encontrado: {path}")
            return

        # Si cambia el video → abrir uno nuevo
        if self.current_video != path:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(path)
            self.current_video = path

        # Leer frame
        ret, frame = self.cap.read()

        # 🔁 LOOP REAL
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()   # 👈 CLAVE (leer otra vez)

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