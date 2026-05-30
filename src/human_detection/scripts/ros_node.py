#!/usr/bin/env python3
import json
import os
import sys

import cv2
import rospy
from std_msgs.msg import String

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from gesture_detector import GestureDetector


class HumanROSNode:
    def __init__(self):
        rospy.init_node("human_detector", anonymous=True)
        self.pub = rospy.Publisher("/human_state", String, queue_size=10)

        self.mode = int(rospy.get_param("~mode", 1))

        self.detector = GestureDetector(
            draw_face=True,
            draw_pose=True,
            draw_hands=True,
            enable_emotion_detection=(self.mode != 4),
        )

        self.cap = cv2.VideoCapture(0)
        rospy.Subscriber("/voice_emotion", String, self.voice_callback)
        self.last_voice = {
            "voice_emotion": "neutral",
            "text": "",
            "final_emotion": "neutral",
        }

        rospy.loginfo("HumanROSNode listo")

    def voice_callback(self, msg):
        try:
            self.last_voice = json.loads(msg.data)
        except Exception:
            pass

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("No se recibió frame")
                rate.sleep()
                continue

            frame = cv2.flip(frame, 1)
            processed_frame, data = self.detector.process_frame(frame)

            payload = {
                "gesture": data.get("gesture", "Ninguno"),
                "raw_gesture": data.get("raw_gesture", "Ninguno"),
                "emotion": data.get("emotion", "Neutral"),
                "confidence": data.get("confidence", 0.0),
                "voice_emotion": self.last_voice.get("voice_emotion", "neutral"),
                "voice_final_emotion": self.last_voice.get("final_emotion", "neutral"),
                "text": self.last_voice.get("text", ""),
            }

            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub.publish(msg)

            cv2.imshow("ROS Human Node", processed_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()


if __name__ == "__main__":
    try:
        HumanROSNode().run()
    except rospy.ROSInterruptException:
        pass