#!/usr/bin/env python3
import json
import os
import sys

import cv2
import rospy
from std_msgs.msg import String

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from avatar_media import media_path


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

    def draw_hand_overlay(self, canvas, hand_landmarks, color_points, color_lines,
                      scale=0.85, x_offset=300, y_offset=-20):
        if not hand_landmarks:
            return canvas

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

    def show_image(self, path, payload):
        img = cv2.imread(path)
        if img is None:
            return

        img = self.draw_hand_overlay(img, payload.get("left_hand_landmarks", []), (0, 255, 0), (0, 180, 0))
        img = self.draw_hand_overlay(img, payload.get("right_hand_landmarks", []), (0, 0, 255), (0, 0, 180))

        cv2.imshow("Avatar", img)
        cv2.waitKey(1)

    def play_video(self, path, payload):
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
            path, is_image = media_path(node.base_path, node.mode, payload)

            if path is None:
                rospy.logwarn_throttle(5, f"No hay imágenes ni vídeos en {node.base_path}")
            elif is_image:
                node.show_image(path, payload)
            else:
                node.play_video(path, payload)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass