#!/usr/bin/env python3
import json
import os

import cv2
import rospy
from std_msgs.msg import String


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


class AvatarNode:
    def __init__(self):
        rospy.init_node("avatar_node")

        self.mode = int(rospy.get_param("~mode", 1))
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
            "left_hand_landmarks": [],
            "right_hand_landmarks": [],
        }

        rospy.Subscriber("/human_state", String, self.callback)

    def callback(self, msg):
        try:
            self.last_payload = json.loads(msg.data)
        except Exception:
            pass

    def resolve_emotion(self, data):
        face_emotion = (data.get("emotion") or "neutral").lower()
        voice_emotion = (data.get("voice_final_emotion") or "").lower()

        if self.mode == 1:
            return face_emotion
        if self.mode in [2, 3]:
            return voice_emotion if voice_emotion and voice_emotion != "neutral" else face_emotion
        if self.mode == 4:
            return face_emotion
        return face_emotion

    def get_key(self, gesture, emotion):
        gesture = (gesture or "Ninguno").lower()
        emotion = (emotion or "neutral").lower()

        if self.mode in [1, 2, 3]:
            return emotion
        return f"{gesture}_{emotion}"

    def draw_hand_overlay(self, canvas, hand_landmarks, color_points, color_lines,
                      scale=0.85, x_offset=300, y_offset=-20):
        if not hand_landmarks:
            return canvas

        h, w = canvas.shape[:2]
        points = {}

        for i, lm in enumerate(hand_landmarks):
            x = int(lm["px"] * scale) + x_offset
            y = int(lm["py"] * scale) + y_offset
            points[i] = (x, y)
            cv2.circle(canvas, (x, y), 3, color_points, -1)

        for a, b in HAND_CONNECTIONS:
            if a in points and b in points:
                cv2.line(canvas, points[a], points[b], color_lines, 2)

        return canvas

    def show_image(self, key, payload):
        path = os.path.join(self.base_path, "imagenes", f"{key}.png")
        if not os.path.exists(path):
            path = os.path.join(self.base_path, "imagenes", "no detectado.png")

        img = cv2.imread(path)
        if img is None:
            return

        img = self.draw_hand_overlay(img, payload.get("left_hand_landmarks", []), (0, 255, 0), (0, 180, 0))
        img = self.draw_hand_overlay(img, payload.get("right_hand_landmarks", []), (0, 0, 255), (0, 0, 180))

        cv2.imshow("Avatar", img)
        cv2.waitKey(1)

    def play_video(self, key, payload):
        path = os.path.join(self.base_path, "videos", f"{key}.mp4")
        if not os.path.exists(path):
            path = os.path.join(self.base_path, "videos", "no detectado.mp4")

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

        frame = self.draw_hand_overlay(frame, payload.get("left_hand_landmarks", []), (0, 255, 0), (0, 180, 0))
        frame = self.draw_hand_overlay(frame, payload.get("right_hand_landmarks", []), (0, 0, 255), (0, 0, 180))

        cv2.imshow("Avatar", frame)
        cv2.waitKey(30)


if __name__ == "__main__":
    try:
        node = AvatarNode()
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            payload = node.last_payload
            gesture = payload.get("gesture", "Ninguno")
            emotion = node.resolve_emotion(payload)
            key = node.get_key(gesture, emotion)

            if node.mode in [1, 2]:
                node.show_image(key, payload)
            else:
                node.play_video(key, payload)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass