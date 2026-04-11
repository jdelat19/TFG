#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

import cv2
import json
import os
import sys

# 👉 asegurar imports del paquete
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gesture_detector import GestureDetector


class HumanROSNode:

    def __init__(self):

        rospy.init_node("human_detector", anonymous=True)

        self.pub = rospy.Publisher("/human_state", String, queue_size=10)

        self.detector = GestureDetector(
            draw_face=True,
            draw_pose=True,
            draw_hands=True,
            enable_emotion_detection=True
        )

        self.cap = cv2.VideoCapture(0)

        rospy.loginfo("Human ROS Node iniciado correctamente")

    def run(self):

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            ret, frame = self.cap.read()

            if not ret:
                rospy.logwarn("No frame recibido")
                continue

            frame = cv2.flip(frame, 1)

            _, data = self.detector.process_frame(frame)

            msg = String()
            msg.data = json.dumps(data, ensure_ascii=False)

            self.pub.publish(msg)

            cv2.imshow("ROS Human Node", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = HumanROSNode()
        node.run()
    except rospy.ROSInterruptException:
        pass