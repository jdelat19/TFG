import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from .gesture_detector import GestureDetector


class GestureEmotionNode(Node):
    def __init__(self):
        super().__init__('gesture_emotion_node')

        # Publicadores ROS
        self.pub_info = self.create_publisher(
            String, 'gestures_emotions', 10
        )
        self.pub_image = self.create_publisher(
            Image, 'camera/processed', 10
        )

        self.bridge = CvBridge()

        # Detector (TU CÓDIGO)
        self.detector = GestureDetector(
            draw_face=True,
            draw_pose=True,
            draw_hands=True,
            enable_emotion_detection=True
        )

        # Cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("❌ No se pudo abrir la cámara")
            return

        self.get_logger().info("✅ Nodo Gestos + Emociones iniciado")

        # Timer (~30 FPS)
        self.timer = self.create_timer(1/30.0, self.loop)

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        processed, data = self.detector.process_frame(frame)

        # Publicar texto
        msg = String()
        msg.data = (
            f"Gesto: {data['gesture']} | "
            f"Emoción: {data['emotion']} "
            f"({data['confidence']:.2f})"
        )
        self.pub_info.publish(msg)

        # Publicar imagen
        img_msg = self.bridge.cv2_to_imgmsg(
            processed, encoding='bgr8'
        )
        self.pub_image.publish(img_msg)

        # Ventana local
        cv2.imshow("Gestos y Emociones", processed)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = GestureEmotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
