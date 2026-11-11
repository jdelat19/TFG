#!/usr/bin/env python3
"""Nodo de detección: cámara -> gesto, emoción facial y manos.

Lee la webcam del PC (o un topic de imagen del robot), procesa cada
fotograma con GestureDetector (MediaPipe + reglas de gestos + HSEmotion) y
muestra la imagen con el esqueleto en la ventana "ROS Human Node" (Esc la
cierra).

Publica:
  /human_state (std_msgs/String, JSON)
     {"gesture", "raw_gesture", "emotion", "confidence",
      "left_hand_landmarks", "right_hand_landmarks",
      "voice_emotion", "voice_final_emotion", "text"}
     Los campos de voz son los últimos recibidos de /voice_emotion, para que
     el avatar tenga todo en un solo mensaje.

Se suscribe a:
  /voice_emotion (String, JSON) del nodo de voz (opcional)
  ~image_topic   (Image o CompressedImage) si se da

Parámetros:
  ~mode             modo del avatar (1-4); en el modo 4 no se calcula la emoción
  ~image_topic      "" = webcam del PC; si termina en /compressed se lee JPEG
  ~flip             espejo horizontal (por defecto sí con la webcam)
  ~emotion_backend  "hsemotion" (por defecto) o "fer"
"""
import json
import os
import sys

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from gesture_detector import GestureDetector


class HumanDetectorNode:
    def __init__(self):
        rospy.init_node("human_detector", anonymous=True)
        self.pub = rospy.Publisher("/human_state", String, queue_size=10)

        self.mode = int(rospy.get_param("~mode", 1))
        # "hsemotion" (más preciso) o "fer"; si el elegido no está disponible se usa el otro
        emotion_backend = rospy.get_param("~emotion_backend", "hsemotion")

        self.detector = GestureDetector(
            draw_face=True,
            draw_pose=True,
            draw_hands=True,
            enable_emotion_detection=(self.mode != 4),
            emotion_backend=emotion_backend,
        )

        # Si image_topic está vacío se usa la webcam local; si no, la cámara de ese topic
        self.image_topic = rospy.get_param("~image_topic", "")
        self.flip = rospy.get_param("~flip", not self.image_topic)
        self.cap = None
        self.last_image = None
        if self.image_topic:
            # .../compressed (JPEG) pesa ~20 veces menos por WiFi que la imagen en crudo
            if self.image_topic.endswith("/compressed"):
                rospy.Subscriber(self.image_topic, CompressedImage, self.compressed_callback,
                                 queue_size=1, buff_size=2**24)
            else:
                rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1,
                                 buff_size=2**24)
            rospy.loginfo("Usando la cámara del topic %s", self.image_topic)
        else:
            self.cap = cv2.VideoCapture(0)
        rospy.Subscriber("/voice_emotion", String, self.voice_callback)
        self.last_voice = {
            "voice_emotion": "neutral",
            "text": "",
            "final_emotion": "neutral",
        }

        rospy.loginfo("HumanDetectorNode listo")

    def image_callback(self, msg):
        """Convierte sensor_msgs/Image a BGR sin depender de cv_bridge."""
        try:
            channels = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}[msg.encoding]
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape(msg.height, msg.step)[:, : msg.width * channels]
            img = img.reshape(msg.height, msg.width, channels).copy()
            code = {"rgb8": cv2.COLOR_RGB2BGR, "mono8": cv2.COLOR_GRAY2BGR,
                    "rgba8": cv2.COLOR_RGBA2BGR, "bgra8": cv2.COLOR_BGRA2BGR}.get(msg.encoding)
            self.last_image = cv2.cvtColor(img, code) if code is not None else img
        except Exception as e:
            rospy.logwarn_throttle(5, "Imagen no soportada (%s): %s", msg.encoding, e)

    def compressed_callback(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            rospy.logwarn_throttle(5, "No se pudo decodificar la imagen comprimida")
            return
        self.last_image = img

    def read_frame(self):
        """(ok, imagen): de la webcam, o la última imagen recibida del topic."""
        if self.cap is not None:
            return self.cap.read()
        frame, self.last_image = self.last_image, None
        return frame is not None, frame

    def voice_callback(self, msg):
        try:
            self.last_voice = json.loads(msg.data)
        except Exception:
            pass

    def run(self):
        # 10 Hz: MediaPipe + HSEmotion tardan ~60-90 ms por fotograma en CPU
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            ret, frame = self.read_frame()
            if not ret:
                rospy.logwarn_throttle(5, "No llegan imágenes de la cámara")
                rate.sleep()
                continue

            if self.flip:
                frame = cv2.flip(frame, 1)
            processed_frame, data = self.detector.process_frame(frame)

            payload = {
                "gesture": data.get("gesture", "Ninguno"),
                "raw_gesture": data.get("raw_gesture", "Ninguno"),
                "emotion": data.get("emotion", "Neutral"),
                "confidence": data.get("confidence", 0.0),
                "left_hand_landmarks": data.get("left_hand_landmarks", []),
                "right_hand_landmarks": data.get("right_hand_landmarks", []),
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

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()


if __name__ == "__main__":
    try:
        HumanDetectorNode().run()
    except rospy.ROSInterruptException:
        pass