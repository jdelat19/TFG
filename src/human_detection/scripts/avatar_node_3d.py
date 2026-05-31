#!/usr/bin/env python3
import json
import os
import math
import numpy as np
import rospy
import cv2

from std_msgs.msg import String

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

class AvatarNode3D:
    def __init__(self):
        rospy.init_node("avatar_node3d", anonymous=True)
        self.basepath = rospy.get_param(
            "basepath",
            os.path.expanduser("~/Escritorio/TFG/src/human_detection/media")
        )
        self.modelpath = os.path.join(self.basepath, "models", "hand.obj")

        self.lastpayload = {
            "left_hand_landmarks": [],
            "right_hand_landmarks": []
        }

        self.window = None
        self.vertices = []
        self.faces = []
        self.model_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.model_size = 1.0

        self.hand_poses = {
            "left": {
                "has_pose": False,
                "pos": np.zeros(3, dtype=np.float32),
                "rot": np.eye(3, dtype=np.float32),
                "scale": 1.0,
                "sm_pos": np.zeros(3, dtype=np.float32),
                "sm_scale": 1.0,
            },
            "right": {
                "has_pose": False,
                "pos": np.zeros(3, dtype=np.float32),
                "rot": np.eye(3, dtype=np.float32),
                "scale": 1.0,
                "sm_pos": np.zeros(3, dtype=np.float32),
                "sm_scale": 1.0,
            }
        }

        self.cap = None
        self.current_video = None
        self.video_texture = None

        rospy.Subscriber("/human_state", String, self.callback)

    def callback(self, msg):
        try:
            self.lastpayload = json.loads(msg.data)
        except Exception:
            pass

    def init_window(self):
        if not glfw.init():
            raise RuntimeError("No se pudo inicializar GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        self.window = glfw.create_window(1280, 720, "Avatar 3D", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("No se pudo crear la ventana")
        glfw.make_context_current(self.window)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        light_pos = [2.0, 5.0, 5.0, 1.0]
        glLightfv(
            GL_LIGHT0,
            GL_POSITION,
            light_pos
        )

        glEnable(GL_COLOR_MATERIAL)

        glColorMaterial(
            GL_FRONT_AND_BACK,
            GL_AMBIENT_AND_DIFFUSE
        )
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.video_texture = self.create_texture()

    def update_video(self):

        voice_emotion = self.lastpayload.get(
            "voice_final_emotion",
            "neutral"
        ).lower()

        face_emotion = self.lastpayload.get(
            "emotion",
            "neutral"
        ).lower()

        # Prioridad a la voz excepto cuando sea neutral
        if voice_emotion != "neutral":
            emotion = voice_emotion
        else:
            emotion = face_emotion

        video_path = os.path.join(
            self.basepath,
            "videos",
            f"{emotion}.mp4"
        )

        if not os.path.exists(video_path):
            video_path = os.path.join(
                self.basepath,
                "videos",
                "no detectado.mp4"
            )

        if video_path != self.current_video:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(video_path)
            self.current_video = video_path

        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            ret, frame = self.cap.read()

        if ret:
            self.update_video_texture(frame)        

    def load_obj(self, path):
        vertices = []
        faces = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    _, x, y, z = line.split()[:4]
                    vertices.append((float(x), float(y), float(z)))
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    idxs = [int(p.split("/")[0]) - 1 for p in parts if p.split("/")[0]]
                    if len(idxs) >= 3:
                        for i in range(1, len(idxs) - 1):
                            faces.append((idxs[0], idxs[i], idxs[i + 1]))
        self.vertices = vertices
        self.faces = faces
        xs = np.array([v[0] for v in vertices], dtype=np.float32)
        ys = np.array([v[1] for v in vertices], dtype=np.float32)
        zs = np.array([v[2] for v in vertices], dtype=np.float32)
        self.model_center = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, (zs.min() + zs.max()) / 2], dtype=np.float32)
        self.model_size = float(max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min(), 1e-6))

    def n2(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v


  

    def draw_sphere(self, radius=0.03):
        quad = gluNewQuadric()
        gluSphere(quad, radius, 16, 16)
        gluDeleteQuadric(quad)


    def draw_cylinder_between(self, p1, p2, radius=0.025):

        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)

        v = p2 - p1
        length = np.linalg.norm(v)

        if length < 1e-6:
            return

        v /= length

        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        axis = np.cross(z_axis, v)
        axis_len = np.linalg.norm(axis)

        if axis_len < 1e-6:
            angle = 0.0
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis /= axis_len
            angle = math.degrees(
                math.acos(
                    np.clip(np.dot(z_axis, v), -1.0, 1.0)
                )
            )

        glPushMatrix()

        glTranslatef(p1[0], p1[1], p1[2])

        glRotatef(
            angle,
            axis[0],
            axis[1],
            axis[2]
        )

        quad = gluNewQuadric()

        gluCylinder(
            quad,
            radius,
            radius,
            length,
            12,
            1
        )

        gluDeleteQuadric(quad)

        glPopMatrix()


    def lm_to_world(self, lm):

        pts = []

        for p in lm:

            x = (p["px"] - 640.0) / 140.0 + 1.5
            y = -(p["py"] - 360.0) / 140.0

            z = 0.0

            if "pz" in p:
                z = -p["pz"] * 3.0

            pts.append(
                np.array([x, y, z], dtype=np.float32)
            )

        return pts

    def draw_hand_landmarks_3d(self, lm):

        if len(lm) < 21:
            return

        pts = self.lm_to_world(lm)

        finger_segments = [

            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            (0, 9),
            (9,10),
            (10,11),
            (11,12),

            (0,13),
            (13,14),
            (14,15),
            (15,16),

            (0,17),
            (17,18),
            (18,19),
            (19,20),
        ]

        glColor3f(
            0.7,
            0.7,
            0.7
        )

        for a, b in finger_segments:

            self.draw_cylinder_between(
                pts[a],
                pts[b],
                radius=0.04
            )

        glColor3f(
            0.9,
            0.3,
            0.3
        )

        for p in pts:

            glPushMatrix()

            glTranslatef(
                p[0],
                p[1],
                p[2]
            )

            self.draw_sphere(0.055)

            glPopMatrix()

    def render(self):

        self.update_video()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.draw_video_background()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        gluPerspective(
            45.0,
            1280.0 / 720.0,
            0.1,
            100.0
        )

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        gluLookAt(
            0.0, 0.0, 8.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        )

        hands = {
            "left": self.lastpayload.get(
                "left_hand_landmarks",
                []
            ),
            "right": self.lastpayload.get(
                "right_hand_landmarks",
                []
            )
        }

        for _, lm in hands.items():

            if len(lm) < 21:
                continue

            self.draw_hand_landmarks_3d(lm)

        glfw.swap_buffers(self.window)

    def create_texture(self):
        tex = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, tex)

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_MIN_FILTER,
            GL_LINEAR
        )

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_LINEAR
        )

        return tex

    def update_video_texture(self, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        glBindTexture(GL_TEXTURE_2D, self.video_texture)

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            frame.shape[1],
            frame.shape[0],
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            frame
        )

    def draw_video_background(self):
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        glColor3f(1.0, 1.0, 1.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, self.video_texture)

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

        glDisable(GL_TEXTURE_2D)

        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

    def run(self):
        self.init_window()
        
        rate = rospy.Rate(60)
        while not rospy.is_shutdown() and not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.render()
            rate.sleep()
        glfw.terminate()


if __name__ == "__main__":
    try:
        AvatarNode3D().run()
    except rospy.ROSInterruptException:
        pass