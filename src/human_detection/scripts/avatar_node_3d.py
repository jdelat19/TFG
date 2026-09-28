#!/usr/bin/env python3
# =============================================================================
# avatar_node3d.py
# Nodo ROS que visualiza en 3D las manos detectadas (landmarks) usando OpenGL.
# - Suscribe a "/human_state" para recibir landmarks de manos.
# - Dibuja cilindros (huesos) y esferas (articulaciones) en 3D.
# - Muestra de fondo una imagen o un vídeo según el modo, la emoción y el
#   gesto (misma lógica que el avatar 2D, en src/avatar_media.py).
# =============================================================================

import json
import math
import os
import sys

import cv2
import glfw
import numpy as np
import rospy
from OpenGL.GL import *
from OpenGL.GLU import *
from std_msgs.msg import String

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from avatar_media import media_path

# Los píxeles se dividen entre 140 para pasar a unidades de la escena y la
# imagen mide 1280 px de ancho, así que la z de MediaPipe (en unidades del
# ancho de la imagen) se multiplica por 1280 / 140 para tener la misma escala.
DEPTH_SCALE = 1280.0 / 140.0


class AvatarNode3D:
    def __init__(self):
        # Inicializa el nodo ROS con nombre "avatar_node3d"
        rospy.init_node("avatar_node3d", anonymous=True)

        # Modo del avatar (1-4) y carpeta de imágenes y vídeos, desde el launch
        self.mode = int(rospy.get_param("~mode", 1))
        self.basepath = rospy.get_param(
            "~base_path",
            os.path.expanduser("~/Escritorio/TFG/src/human_detection/media"),
        )

        # Último payload recibido por elSubscriber (con landmarks de manos)
        self.lastpayload = {
            "left_hand_landmarks": [],
            "right_hand_landmarks": [],
        }

        # Variables de ventana OpenGL y textura de video
        self.window = None
        self.video_texture = None
        self.cap = None               # Captura de video de OpenCV
        self.current_background = None  # Ruta de la imagen o vídeo cargado

        # Suscripción al topic "/human_state" que envía String con JSON
        rospy.Subscriber("/human_state", String, self.callback)

    # -------------------------------------------------------------------------
    # Callback del Subscriber: recibe el mensaje y guarda el payload JSON
    # -------------------------------------------------------------------------
    def callback(self, msg):
        try:
            # Convierte el string del mensaje a un diccionario Python
            self.lastpayload = json.loads(msg.data)
        except Exception:
            # Si falla el parseo, simplemente ignora y mantiene el último válido
            pass

    # -------------------------------------------------------------------------
    # Inicializa la ventana GLFW y el contexto OpenGL
    # -------------------------------------------------------------------------
    def init_window(self):
        # Inicializa GLFW
        if not glfw.init():
            raise RuntimeError("No se pudo inicializar GLFW")

        # Configura versión del contexto OpenGL (2.1 para compatibilidad)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        # Crea ventana de 1280x720 llamada "Avatar 3D"
        self.window = glfw.create_window(1280, 720, "Avatar 3D", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("No se pudo crear la ventana")

        # Hace este hilo el que usa el contexto OpenGL
        glfw.make_context_current(self.window)

        # Habilita pruebas y características de OpenGL
        glEnable(GL_DEPTH_TEST)        # Prueba de profundidad para 3D
        glEnable(GL_NORMALIZE)         # Normalización de normales
        glEnable(GL_LIGHTING)          # Iluminación
        glEnable(GL_LIGHT0)            # Primera luz
        glEnable(GL_COLOR_MATERIAL)    # Usar color de material para lighting
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Color de fondo: negro con transparencia
        glClearColor(0.0, 0.0, 0.0, 0.0)
        # Habilita blending para transparencia
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Posiciona la luz 0
        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 5.0, 5.0, 1.0])

        # Crea la textura que se usará para el video de fondo
        self.video_texture = self.create_texture()

    # -------------------------------------------------------------------------
    # Actualiza el fondo (imagen o vídeo) según el modo, la emoción y el gesto
    # -------------------------------------------------------------------------
    def update_background(self):
        path, is_image = media_path(self.basepath, self.mode, self.lastpayload)
        if path is None:
            return

        # Si cambió el recurso, libera el vídeo anterior y carga el nuevo
        if path != self.current_background:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.current_background = path
            if is_image:
                # Una imagen solo hay que subirla a la textura una vez
                image = cv2.imread(path)
                if image is not None:
                    self.update_video_texture(image)
                return
            self.cap = cv2.VideoCapture(path)

        # Si no hay vídeo (fondo de imagen), no hay nada que actualizar
        if self.cap is None:
            return

        # Lee un frame del video
        ret, frame = self.cap.read()
        # Si llega al final, vuelve al principio
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        # Si se lee un frame, actualiza la textura de video
        if ret:
            self.update_video_texture(frame)

    # -------------------------------------------------------------------------
    # Dibuja una esfera (para las articulaciones)
    # -------------------------------------------------------------------------
    def draw_sphere(self, radius=0.03):
        quad = gluNewQuadric()
        gluSphere(quad, radius, 16, 16)  # esferas con 16x16 segmentos
        gluDeleteQuadric(quad)

    # -------------------------------------------------------------------------
    # Dibuja un cilindro entre dos puntos (para los "huesos" de los dedos)
    # -------------------------------------------------------------------------
    def draw_cylinder_between(self, p1, p2, radius=0.025):
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)

        # Vector de p1 a p2 y su longitud
        v = p2 - p1
        length = np.linalg.norm(v)
        if length < 1e-6:
            return

        v /= length  # normaliza

        # Eje Z por defecto (0,0,1)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Calcula el eje de rotación (producto cruz) y el ángulo
        axis = np.cross(z_axis, v)
        axis_len = np.linalg.norm(axis)

        if axis_len < 1e-6:
            # Si v está alineado con Z, no hay rotación
            angle = 0.0
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            axis /= axis_len
            angle = math.degrees(
                math.acos(np.clip(np.dot(z_axis, v), -1.0, 1.0))
            )

        # Guarda matriz actual, traslada a p1 y rota para alinear con el vector
        glPushMatrix()
        glTranslatef(p1[0], p1[1], p1[2])
        glRotatef(angle, axis[0], axis[1], axis[2])

        # Dibuja cilindro de radio constante y longitud = distancia entre p1 y p2
        quad = gluNewQuadric()
        gluCylinder(quad, radius, radius, length, 12, 1)
        gluDeleteQuadric(quad)

        glPopMatrix()

    # -------------------------------------------------------------------------
    # Convierte landmarks 2D (con profundidad opcional) a coordenadas 3D
    # -------------------------------------------------------------------------
    def lm_to_world(self, lm):
        pts = []
        for p in lm:
            # Mapea px, py de imagen (640x360) a coordenadas 3D
            x = (p["px"] - 640.0) / 140.0 + 1.5
            y = -(p["py"] - 360.0) / 140.0
            # Profundidad de MediaPipe (negativa = más cerca de la cámara)
            z = -p.get("z", 0.0) * DEPTH_SCALE
            pts.append(np.array([x, y, z], dtype=np.float32))
        return pts

    # -------------------------------------------------------------------------
    # Dibuja los landmarks de una mano en 3D:
    # - cilindros entre articulaciones
    # - esferas en cada landmark
    # -------------------------------------------------------------------------
    def draw_hand_landmarks_3d(self, lm):
        if len(lm) < 21:
            return

        # Convierte landmarks a 3D
        pts = self.lm_to_world(lm)

        # Definición de segmentos entre landmarks (dedos + palma)
        # Cada dedo tiene 4 segmentos: raíz → punta
        finger_segments = [
            (0, 1), (1, 2), (2, 3), (3, 4),   # pulgar
            (0, 5), (5, 6), (6, 7), (7, 8),   # índice
            (0, 9), (9, 10), (10, 11), (11, 12),  # medio
            (0, 13), (13, 14), (14, 15), (15, 16),  # anular
            (0, 17), (17, 18), (18, 19), (19, 20),  # meñique
        ]

        # Dibuja cilindros (huesos) en gris
        glColor3f(0.7, 0.7, 0.7)
        for a, b in finger_segments:
            self.draw_cylinder_between(pts[a], pts[b], radius=0.04)

        # Dibuja esferas (articulaciones) en rojo
        glColor3f(0.9, 0.3, 0.3)
        for p in pts:
            glPushMatrix()
            glTranslatef(p[0], p[1], p[2])
            self.draw_sphere(0.055)
            glPopMatrix()

    # -------------------------------------------------------------------------
    # Bucle de renderizado: actualiza video, limpia, dibuja fondo y manos
    # -------------------------------------------------------------------------
    def render(self):
        # Actualiza el fondo según la emoción
        self.update_background()

        # Limpia buffer de color y profundidad
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Dibuja el video como fondo
        self.draw_video_background()

        # Configura proyección perspectivica
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, 1280.0 / 720.0, 0.1, 100.0)

        # Configura vista de cámara
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, 8.0,  # cámara en (0,0,8)
                  0.0, 0.0, 0.0,  # mira al origen
                  0.0, 1.0, 0.0)  # eje "arriba" es Y

        # Obtiene landmarks de ambas manos
        left_lm = self.lastpayload.get("left_hand_landmarks", [])
        right_lm = self.lastpayload.get("right_hand_landmarks", [])

        # Dibuja cada mano si tiene al menos 21 landmarks
        for lm in (left_lm, right_lm):
            if len(lm) >= 21:
                self.draw_hand_landmarks_3d(lm)

        # Intercambia buffers (dibuja el frame en pantalla)
        glfw.swap_buffers(self.window)

    # -------------------------------------------------------------------------
    # Crea una textura OpenGL vacía (lista para usar como textura de video)
    # -------------------------------------------------------------------------
    def create_texture(self):
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        # Filas de píxeles sin relleno: si el ancho*3 no es múltiplo de 4, la
        # alineación por defecto (4) deformaría la imagen
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        # Filtro lineal para minimización y magnificación
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tex

    # -------------------------------------------------------------------------
    # Actualiza la textura de video con un frame de OpenCV (RGB)
    # -------------------------------------------------------------------------
    def update_video_texture(self, frame):
        # OpenCV carga en BGR, OpenGL necesita RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        glBindTexture(GL_TEXTURE_2D, self.video_texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            frame.shape[1],  # ancho
            frame.shape[0],  # alto
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            frame,
        )

    # -------------------------------------------------------------------------
    # Dibuja el video como fondo (2D en la pantalla completa)
    # -------------------------------------------------------------------------
    def draw_video_background(self):
        # Desactiva pruebas de profundidad e iluminación para el fondo
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        glColor3f(1.0, 1.0, 1.0)  # color blanco (textura se usa tal cual)

        # Cambia a proyección ortográfica (2D)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Activa textura y la une al video
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.video_texture)

        # Dibuja un cuadrado que cubre toda la pantalla
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1)
        glVertex2f(-1, -1)

        glTexCoord2f(1, 1)
        glVertex2f(1, -1)

        glTexCoord2f(1, 0)
        glVertex2f(1, 1)

        glTexCoord2f(0, 0)
        glVertex2f(-1, 1)
        glEnd()

        # Restaura estado para 3D
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

    # -------------------------------------------------------------------------
    # Bucle principal: inicializa ventana, renderiza a 60 FPS hasta cerrar
    # -------------------------------------------------------------------------
    def run(self):
        self.init_window()

        rate = rospy.Rate(60)  # 60 FPS
        while not rospy.is_shutdown() and not glfw.window_should_close(self.window):
            glfw.poll_events()  # procesa eventos de GLFW (cierre, teclado, etc.)
            self.render()       # renderiza un frame
            rate.sleep()        # mantiene 60 FPS

        glfw.terminate()  # libera GLFW al salir


# =============================================================================
# Punto de entrada: crea y ejecuta el nodo AvatarNode3D
# =============================================================================
if __name__ == "__main__":
    try:
        AvatarNode3D().run()
    except rospy.ROSInterruptException:
        pass