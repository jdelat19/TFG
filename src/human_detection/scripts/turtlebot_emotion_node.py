#!/usr/bin/env python3
"""Nodo del TurtleBot2: elige un comportamiento según la emoción de la persona.

Este fichero es la parte de ROS (entradas, salidas y decisión). Cómo se mueve
el robot en cada comportamiento está en src/turtlebot_behaviors.py, y la
geometría del /scan (patas, mesas, personas) en src/scan_utils.py.

Entradas:
  /human_state      (String JSON)  emoción facial publicada por human_detector_node.py
  /voice_emotion    (String JSON)  emoción por voz publicada por voice_node.py
  /emotion_override (String)       fuerza una emoción para pruebas ("auto" la desactiva,
                                   "parar" detiene el robot hasta recibir otra)
  /scan             (LaserScan)    lidar 360º o cámara de profundidad (depthimage_to_laserscan)
  /odom             (Odometry)
  /mobile_base/events/bumper (kobuki_msgs/BumperEvent, opcional)

Salidas:
  /mobile_base/commands/velocity (Twist)  velocidad del Kobuki (~cmd_vel_topic)
  /turtlebot_emotion/state (String JSON)  emoción, comportamiento y fase activos
  /turtlebot_emotion/map   (String JSON)  "mapa mental" para mental_map_node.py

Ciclo de control (step, a ~rate Hz):
  1. Emoción: override > voz reciente y no neutral > cara (fused_emotion).
  2. Una emoción nueva solo cambia el comportamiento si se mantiene
     emotion_hold s (neutral_hold para neutral) y el actual lleva al menos
     min_behavior_time s (update_emotion). Así no da tirones.
  3. Se calcula el Twist del comportamiento, se limita la aceleración (ramp)
     y se anula el avance si hay un obstáculo delante (safety_filter).
  4. Si /odom deja de llegar más de odom_timeout s, el robot se para.

El /scan puede venir de un lidar 360º o de la cámara de profundidad (~63º y
nada a menos de 0.45 m). Los puntos se pasan al marco de la base con el TF
del sensor y se colocan en odom con la pose del instante del scan (no con la
última), para que no se desplacen al girar.

Se asume que al arrancar el robot está mirando hacia la persona, a
person_initial_distance metros.
"""
import json
import math
import os
import sys
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

try:
    import tf2_ros
except ImportError:
    tf2_ros = None
try:
    from kobuki_msgs.msg import BumperEvent
except ImportError:
    BumperEvent = None

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import scan_utils as su
from emotions import BEHAVIORS, normalize_emotion
from turtlebot_behaviors import TurtlebotBehaviors, make_twist


class TurtlebotEmotionNode(TurtlebotBehaviors):
    def __init__(self):
        rospy.init_node("turtlebot_emotion_node")

        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.max_v = rospy.get_param("~max_linear", 0.18)
        self.max_w = rospy.get_param("~max_angular", 1.2)
        self.approach_distance = rospy.get_param("~approach_distance", 0.6)
        self.comfort_distance = rospy.get_param("~comfort_distance", 0.9)
        self.retreat_distance = rospy.get_param("~retreat_distance", 1.5)
        # Radio de la base Kobuki del TurtleBot2 (el lidar va en el centro)
        self.robot_radius = rospy.get_param("~robot_radius", 0.18)
        self.safety_distance = rospy.get_param("~safety_distance", 0.35)
        # Segundos que una emoción debe mantenerse para cambiar de comportamiento
        self.emotion_hold = rospy.get_param("~emotion_hold", 3.0)
        # Segundos mínimos que dura un comportamiento antes de aceptar otro
        self.min_behavior_time = rospy.get_param("~min_behavior_time", 5.0)
        # La cara "neutral" suele ser duda del clasificador: tiene que sostenerse más
        self.neutral_hold = rospy.get_param("~neutral_hold", 8.0)
        # Si no llegan frames en este tiempo, la emoción de la cara deja de contar
        self.face_timeout = rospy.get_param("~face_timeout", 1.0)
        self.voice_timeout = rospy.get_param("~voice_timeout", 6.0)
        self.person_initial_distance = rospy.get_param("~person_initial_distance", 1.0)
        # Hasta 15 cm: las columnas de las mesas de pie central o en T (6-12 cm, y
        # algo más con el ruido de la cámara) no se toman por un trozo de pared
        self.leg_max_width = rospy.get_param("~leg_max_width", 0.15)
        # Aceleraciones máximas (m/s² y rad/s²): arrancar suave evita que patinen
        self.acc_linear = rospy.get_param("~acc_linear", 0.3)
        self.acc_angular = rospy.get_param("~acc_angular", 1.5)
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        # [x, y, yaw] del sensor en la base; si no se da, se lee del TF
        self.sensor_param = rospy.get_param("~sensor_offset", None)
        self.behaviors = dict(BEHAVIORS, **rospy.get_param("~behaviors", {}))
        # Giros y distancias se miden con /odom (encoders del Kobuki y rumbo de su
        # giroscopio), no por tiempo: con tiempo el robot real gira ~60% de lo pedido
        # por la aceleración de la base y el patinaje. Si la odometría tiene más de
        # odom_timeout segundos, el robot se para (giraría o avanzaría a ciegas)
        self.odom_timeout = rospy.get_param("~odom_timeout", 0.5)

        self.init_behavior_params()

        self.pose = None      # (x, y, yaw) en odom
        self.odom_time = -1e9  # rospy.get_time() del último /odom recibido
        self.odom_hist = deque(maxlen=150)  # (t, pose) para la pose de cada scan
        self.sensor_poses = {}   # frame del scan -> (x, y, yaw) en la base
        self.blind_front = 0.0   # por debajo de esto (desde el centro) el sensor no ve
        self.bump_time = -1e9
        self.last_cmd = make_twist()
        self.bump_pending = False
        self.stopped = False
        self.scan = None      # último barrido procesado
        self.scan_fov = 2 * math.pi  # ángulo que cubre el /scan
        self.sweep = None     # puntos (x, y) en odom acumulados durante una vuelta
        self.person = None    # posición estimada de la persona en odom
        self.person_confirmed = False

        self.face_emotion = None
        self.face_time = 0.0
        self.voice_emotion = None
        self.voice_time = 0.0
        self.override = None

        self.active_emotion = None
        self.candidate = None
        self.candidate_since = 0.0
        self.behavior_since = -1e9
        self.behavior = "idle"
        self.st = {}          # estado interno del comportamiento activo
        self.exit_from = None  # pose al empezar a salir de debajo de la mesa
        self.exit_distance = 0.0

        self.tf_buffer = None
        if tf2_ros is not None and self.sensor_param is None:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cmd_pub = rospy.Publisher(
            rospy.get_param("~cmd_vel_topic", "/mobile_base/commands/velocity"), Twist, queue_size=1)
        self.state_pub = rospy.Publisher("/turtlebot_emotion/state", String, queue_size=1)
        self.map_pub = rospy.Publisher("/turtlebot_emotion/map", String, queue_size=1)
        self.map_time = -1e9
        self.map_view = None   # última vuelta de búsqueda: puntos, patas y mesas en odom

        rospy.Subscriber("/human_state", String, self.human_callback, queue_size=1)
        rospy.Subscriber("/voice_emotion", String, self.voice_callback, queue_size=1)
        rospy.Subscriber("/emotion_override", String, self.override_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan,
                         self.scan_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/odom"), Odometry,
                         self.odom_callback, queue_size=1)

        if BumperEvent is not None:
            rospy.Subscriber("/mobile_base/events/bumper", BumperEvent,
                             self.bumper_callback, queue_size=5)
        rospy.on_shutdown(lambda: self.cmd_pub.publish(make_twist()))
        rospy.loginfo("TurtlebotEmotionNode listo")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def human_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        self.face_emotion = normalize_emotion(data.get("emotion"))
        self.face_time = rospy.get_time()

    def voice_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        # Sin texto transcrito suele ser silencio o ruido: no cuenta como emoción
        if not str(data.get("text", "")).strip():
            return
        self.voice_emotion = normalize_emotion(data.get("final_emotion"))
        self.voice_time = rospy.get_time()

    def override_callback(self, msg):
        label = msg.data.strip().lower()
        if label in ("parar", "stop"):
            rospy.logwarn("PARADO: no me muevo hasta recibir otra emoción")
            self.stopped, self.override = True, None
            self.active_emotion, self.candidate = None, None
            self.behavior, self.st, self.sweep, self.exit_from = "stop", {}, None, None
            return
        self.stopped = False
        self.override = None if label == "auto" else normalize_emotion(label)
        rospy.loginfo(f"Override de emoción: {self.override or 'desactivado'}")

    def bumper_callback(self, msg):
        if msg.state == BumperEvent.PRESSED:
            rospy.logwarn("Bumper pulsado")
            self.bump_time = rospy.get_time()
            self.bump_pending = True

    def scan_callback(self, msg):
        if self.pose is None:
            return
        _, _, xs, ys = su.scan_to_points(
            msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max)
        # Del marco del sensor al de la base (la cámara va 9 cm por detrás del centro)
        sensor = self.sensor_pose(msg.header.frame_id)
        xs, ys = su.robot_to_odom(xs, ys, sensor)
        self.scan_fov = abs(msg.angle_increment) * len(msg.ranges)
        self.blind_front = max(0.0, msg.range_min + sensor[0])
        pose = self.pose_at(msg.header.stamp.to_sec())
        self.scan = self.build_scan(np.arctan2(ys, xs), np.hypot(xs, ys), pose)
        sweep = self.sweep
        if sweep is not None and len(xs):
            ox, oy = su.robot_to_odom(xs, ys, pose)
            # Ángulo respecto al eje del sensor: en la vista acumulada se prefiere
            # el punto visto más centrado
            sx, sy = su.odom_to_robot(xs, ys, sensor)
            # Un solo append por scan: este callback va en otro hilo y la vuelta
            # puede terminar en cualquier momento
            sweep.append((ox, oy, np.abs(np.arctan2(sy, sx))))

    def sensor_pose(self, frame):
        """(x, y, yaw) del sensor en la base: parámetro ~sensor_offset o TF."""
        if self.sensor_param is not None:
            return tuple(float(v) for v in self.sensor_param)
        frame = frame.lstrip("/")
        if frame in self.sensor_poses:
            return self.sensor_poses[frame]
        if self.tf_buffer is None or not frame or frame == self.base_frame:
            return (0.0, 0.0, 0.0)
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, frame, rospy.Time(0))
        except Exception as e:  # tf2 aún no tiene el árbol
            rospy.logwarn_throttle(5, f"Sin TF {self.base_frame} -> {frame} ({e}); uso el centro")
            return (0.0, 0.0, 0.0)
        tr, q = t.transform.translation, t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.sensor_poses[frame] = (tr.x, tr.y, yaw)
        rospy.loginfo(f"Sensor {frame} en la base: x={tr.x:.3f} y={tr.y:.3f} "
                      f"yaw={math.degrees(yaw):.1f}º")
        return self.sensor_poses[frame]

    def pose_at(self, t):
        """Pose de odom interpolada en el instante t (la del scan)."""
        hist = list(self.odom_hist)
        if t <= 0 or not hist or t >= hist[-1][0]:
            return self.pose
        if t <= hist[0][0]:
            return hist[0][1]
        for (t0, p0), (t1, p1) in zip(reversed(hist[:-1]), reversed(hist[1:])):
            if t0 <= t:
                k = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                yaw = p0[2] + k * su.normalize_angle(p1[2] - p0[2])
                return (p0[0] + k * (p1[0] - p0[0]), p0[1] + k * (p1[1] - p0[1]),
                        su.normalize_angle(yaw))
        return self.pose

    @staticmethod
    def build_scan(ang, r, pose):
        """Agrupa un barrido (ángulos, distancias) en objetos, patas y personas.

        `pose` es la pose de odom desde la que se tomó.
        """
        xs, ys = r * np.cos(ang), r * np.sin(ang)
        clusters = su.describe_clusters(xs, ys, su.cluster_points(xs, ys))
        return {"angles": ang, "ranges": r, "xs": xs, "ys": ys, "clusters": clusters,
                "people": su.find_people(clusters), "pose": pose}

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)
        self.odom_time = rospy.get_time()
        self.odom_hist.append((msg.header.stamp.to_sec(), self.pose))
        if self.person is None:
            self.person = su.robot_to_odom(self.person_initial_distance, 0.0, self.pose)

    # ------------------------------------------------------------------
    # Emoción -> comportamiento
    # ------------------------------------------------------------------
    def fused_emotion(self, now):
        if self.override:
            return self.override
        voice_fresh = now - self.voice_time < self.voice_timeout
        if voice_fresh and self.voice_emotion not in (None, "neutral"):
            return self.voice_emotion
        if now - self.face_time > self.face_timeout:
            return None   # sin frames: se mantiene la emoción actual
        return self.face_emotion

    def update_emotion(self, now):
        emotion = self.fused_emotion(now)
        if emotion is None:
            return
        if emotion != self.candidate:
            self.candidate, self.candidate_since = emotion, now
        if emotion == self.active_emotion:
            return
        if self.override:
            self.set_behavior(emotion, now)
            return
        hold = self.neutral_hold if emotion == "neutral" else self.emotion_hold
        if (now - self.candidate_since >= hold
                and now - self.behavior_since >= self.min_behavior_time):
            self.set_behavior(emotion, now)

    def set_behavior(self, emotion, now=None):
        new_behavior = self.behaviors.get(emotion, "idle")
        rospy.loginfo(f"Emoción: {emotion} -> comportamiento: {new_behavior}")
        self.active_emotion = emotion
        self.behavior_since = rospy.get_time() if now is None else now
        if new_behavior == self.behavior:
            return
        # Si estaba debajo de la mesa, primero sale marcha atrás
        # (en "hidden" sin mesa es que no encontró ninguna y no hay de dónde salir)
        phase = self.st.get("phase")
        if self.behavior == "hide" and (phase in ("enter", "check", "correct")
                                        or (phase == "hidden" and "depth" in self.st)):
            self.start_exit()
        self.behavior = new_behavior
        self.st = {}
        self.sweep = None
    # ------------------------------------------------------------------
    def step(self):
        now = rospy.get_time()
        if self.stopped:
            self.last_cmd = make_twist()
            self.cmd_pub.publish(self.last_cmd)
            self.state_pub.publish(String(data=json.dumps(
                {"emotion": None, "behavior": "stop", "phase": None})))
            self.publish_map(now)
            return
        self.update_emotion(now)

        if self.exit_from is not None:
            cmd = self.behave_exit()
        elif self.behavior == "approach":
            cmd = self.behave_approach(self.approach_distance, self.max_v)
        elif self.behavior == "comfort":
            cmd = self.behave_approach(self.comfort_distance, 0.5 * self.max_v, keep_distance=True)
        elif self.behavior == "retreat":
            cmd = self.behave_retreat()
        elif self.behavior == "spin":
            cmd = self.behave_spin()
        elif self.behavior == "hide":
            cmd = self.behave_hide()
        else:
            cmd = self.behave_idle()

        self.bump_pending = False
        cmd = self.safety_filter(self.ramp(cmd))
        self.last_cmd = cmd
        if self.exit_from is None and self.behavior == "hide" and self.st.get("phase") in ("enter", "correct"):
            self.check_stuck(cmd, now)
        self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data=json.dumps({
            "emotion": self.active_emotion,
            "behavior": self.behavior,
            "phase": self.st.get("phase"),
        })))
        self.publish_map(now)

    # ------------------------------------------------------------------
    # Mapa mental (lo que el robot cree que hay, en coordenadas de odom)
    # ------------------------------------------------------------------
    @staticmethod
    def _pts(xs, ys, max_points=600):
        step = max(1, len(xs) // max_points)
        return [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs[::step], ys[::step])]

    def store_map_view(self, view):
        pose = view["pose"]
        xs, ys = su.robot_to_odom(view["xs"], view["ys"], pose)
        legs = su.find_legs(view["clusters"], max_width=self.leg_max_width)
        tables = self.find_all_tables(view)
        self.map_view = {
            "points": self._pts(xs, ys, 1500),
            "legs": [list(su.robot_to_odom(l["x"], l["y"], pose)) for l in legs],
            "tables": [[list(p) for p in self.legs_in_odom(t, pose)] for t in tables],
            "pose": list(pose),
        }

    def publish_map(self, now, period=0.25):
        if now - self.map_time < period or self.map_pub.get_num_connections() == 0:
            return
        self.map_time = now
        s = self.scan
        xs, ys = su.robot_to_odom(s["xs"], s["ys"], s["pose"])
        legs = su.find_legs(s["clusters"], max_width=self.leg_max_width)
        st = self.st
        table = None
        if "legs" in st:
            table = {k: [list(p) for p in st["legs"]] if k == "legs" else list(st[k])
                     for k in ("legs", "pre", "inside", "table", "normal") if k in st}
        data = {
            "t": now,
            "pose": list(self.pose),
            "scan": self._pts(xs, ys),
            "scan_legs": [list(su.robot_to_odom(l["x"], l["y"], s["pose"])) for l in legs],
            "fov": self.scan_fov,
            "view": self.map_view,
            "table": table,
            "route": [list(p) for p in st.get("route") or []],
            "bad_legs": [list(p) for p in st.get("bad_legs", [])],
            "person": list(self.person) if self.person else None,
            "person_confirmed": self.person_confirmed,
            "emotion": self.active_emotion,
            "behavior": "stop" if self.stopped else self.behavior,
            "phase": st.get("phase"),
            "exiting": self.exit_from is not None,
            "bumper": self.bumped(),
        }
        self.map_pub.publish(String(data=json.dumps(data)))

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.pose is None or self.scan is None:
                rospy.logwarn_throttle(5, "Esperando /odom y /scan...")
            elif rospy.get_time() - self.odom_time > self.odom_timeout:
                # Sin odometría reciente no se sabe cuánto se ha girado o avanzado
                rospy.logwarn_throttle(2, f"/odom sin datos desde hace más de "
                                          f"{self.odom_timeout:.1f} s: robot parado")
                self.last_cmd = make_twist()
                self.cmd_pub.publish(self.last_cmd)
            else:
                self.step()
            rate.sleep()


if __name__ == "__main__":
    try:
        TurtlebotEmotionNode().run()
    except rospy.ROSInterruptException:
        pass
