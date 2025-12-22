import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
from gesture_detector import GestureDetector
from facial_expression import FacialExpressionDetector  # 👈 Nuevo import

def main():
    # Inicializar detector de gestos
    detector = GestureDetector(draw_face=True, draw_pose=True, draw_hands=True)
    # Inicializar detector de emociones
    emotion_detector = FacialExpressionDetector(min_confidence=0.6)

    print("🎯 Sistema de Detección de Gestos y Emociones Iniciado")
    print("📝 Gestos cargados:", len(detector.list_gestures()))
    print("⏹️ Presiona 'q' para salir")
    print("📊 Presiona 'g' para ver gestos disponibles")
    print("🔲 Presiona '1' para alternar landmarks de la cara")
    print("🔲 Presiona '2' para alternar landmarks del cuerpo")
    print("🔲 Presiona '3' para alternar landmarks de las manos")
    print("😊 Presiona 'e' para activar/desactivar detección de emociones")

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return

    detect_emotions = True  # 👈 Estado de detección de emociones

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: No se pudo leer el frame")
            break

        frame = cv2.flip(frame, 1)
        processed_frame = detector.process_frame(frame)

        # 👇 Detección facial si está activada
        if detect_emotions:
            emotion = emotion_detector.detect_emotion(frame)
            if emotion:
                cv2.putText(
                    processed_frame,
                    f"Emoción: {emotion}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow('Detección de Gestos y Emociones - Presiona Q para salir', processed_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('g'):
            print("\n🧠 GESTOS DISPONIBLES:")
            for i, gesture in enumerate(detector.list_gestures()):
                print(f"  {i+1}. {gesture}")
        elif key == ord('1'):
            detector.draw_face = not detector.draw_face
            print(f"Landmarks de la cara: {'ON' if detector.draw_face else 'OFF'}")
        elif key == ord('2'):
            detector.draw_pose = not detector.draw_pose
            print(f"Landmarks del cuerpo: {'ON' if detector.draw_pose else 'OFF'}")
        elif key == ord('3'):
            detector.draw_hands = not detector.draw_hands
            print(f"Landmarks de las manos: {'ON' if detector.draw_hands else 'OFF'}")
        elif key == ord('e'):
            detect_emotions = not detect_emotions
            print(f"Detección de emociones: {'ON' if detect_emotions else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Sistema cerrado exitosamente!")

if __name__ == "__main__":
    main()
