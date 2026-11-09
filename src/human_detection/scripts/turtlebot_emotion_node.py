#!/usr/bin/env python3
"""Comportamiento del TurtleBot según el estado de ánimo detectado.

Entradas:
  /human_state      (String JSON)  emoción facial publicada por ros_node.py
  /voice_emotion    (String JSON)  emoción por voz publicada por voice_node.py
  /emotion_override (String)       fuerza una emoción para pruebas ("auto" la desactiva)
  /scan             (LaserScan)    lidar del TurtleBot
  /odom             (Odometry)
  /mobile_base/events/bumper (kobuki_msgs/BumperEvent, opcional)

Salidas:
  /mobile_base/commands/velocity (Twist)  velocidad del Kobuki
  /turtlebot_emotion/state (String JSON)  emoción y comportamiento activos

Comportamientos:
  feliz            -> se acerca a la persona
  triste           -> se acerca despacio y se queda a más distancia (acompañar)
  enojo / disgusto -> busca una mesa con el lidar y se esconde debajo
                      (si se atasca al entrar, sale marcha atrás y busca otra)
  miedo            -> retrocede manteniendo distancia con la persona
  sorpresa         -> da una vuelta sobre sí mismo
  neutral          -> se queda quieto mirando a la persona
  "parar" en /emotion_override -> se para del todo hasta recibir otra emoción

El /scan puede venir de un lidar 360º o de la cámara de profundidad
(depthimage_to_laserscan, ~63º y nada a menos de 0.45 m). Los puntos se pasan
al marco de la base con el TF del sensor y se colocan en odom con la pose del
instante del scan (no con la última), para que no se desplacen al girar.

Se asume que al arrancar el robot está mirando hacia la persona.
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

BEHAVIORS = {
    "feliz": "approach",
    "triste": "comfort",
    "enojo": "hide",
    "disgusto": "hide",
    "miedo": "retreat",
    "sorpresa": "spin",
    "neutral": "idle",
}



def clip(value, limit):
    return max(-limit, min(limit, value))


def make_twist(v=0.0, w=0.0):
    t = Twist()
    t.linear.x = v
    t.angular.z = w
    return t


class TurtlebotEmotionNode:
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
        self.leg_max_width = rospy.get_param("~leg_max_width", 0.09)
        self.table_min_spacing = rospy.get_param("~table_min_spacing", 0.6)
        self.table_max_spacing = rospy.get_param("~table_max_spacing", 1.6)
        self.under_table_depth = rospy.get_param("~under_table_depth", 0.3)
        # Distancia al lado de la mesa desde la que se empieza a entrar. Con un
        # sensor de campo estrecho se aleja lo necesario para ver las dos patas
        self.pre_distance = rospy.get_param("~pre_distance", 0.35)
        # Distancia máxima entre donde se espera una pata y donde se ve
        self.leg_gate = rospy.get_param("~leg_gate", 0.2)
        # Lo mismo al volver a buscar la mesa tras la vuelta (error del giro)
        self.acquire_gate = rospy.get_param("~acquire_gate", 0.6)
        # Mesas descartadas como máximo antes de rendirse
        self.max_rejected = rospy.get_param("~max_rejected", 6)
        self.enter_speed = rospy.get_param("~enter_speed", 0.1)
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        # [x, y, yaw] del sensor en la base; si no se da, se lee del TF
        self.sensor_param = rospy.get_param("~sensor_offset", None)
        # Carpeta donde guardar cada vuelta de búsqueda (.npz) para analizarla
        self.debug_dir = rospy.get_param("~debug_dir", "")
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
        # Si al entrar bajo la mesa avanza menos de stall_distance en stall_time
        # segundos, se ha atascado (tablero demasiado bajo): sale y busca otra
        self.stall_time = rospy.get_param("~stall_time", 2.0)
        self.stall_distance = rospy.get_param("~stall_distance", 0.03)
        self.behaviors = dict(BEHAVIORS, **rospy.get_param("~behaviors", {}))
        # Si el /scan cubre menos de este ángulo (p. ej. el de la cámara de
        # profundidad, ~60º), antes de buscar mesa da una vuelta acumulando puntos
        self.sweep_fov = math.radians(rospy.get_param("~sweep_below_fov_deg", 300.0))
        self.sweep_speed = rospy.get_param("~sweep_speed", 0.5)
        # En la vuelta solo cuentan los puntos a menos de este ángulo del eje del sensor
        self.sweep_center = rospy.get_param("~sweep_center", 0.15)

        self.pose = None      # (x, y, yaw) en odom
        self.odom_hist = deque(maxlen=150)  # (t, pose) para la pose de cada scan
        self.sensor_poses = {}   # frame del scan -> (x, y, yaw) en la base
        self.blind_front = 0.0   # por debajo de esto (desde el centro) el sensor no ve
        self.bump_time = -1e9
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

        self.tf_buffer = None
        if tf2_ros is not None and self.sensor_param is None:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cmd_pub = rospy.Publisher(
            rospy.get_param("~cmd_vel_topic", "/mobile_base/commands/velocity"), Twist, queue_size=1)
        self.state_pub = rospy.Publisher("/turtlebot_emotion/state", String, queue_size=1)

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
    @staticmethod
    def normalize_emotion(label):
        if not label:
            return None
        label = str(label).strip().lower()
        return label if label in BEHAVIORS else None

    def human_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        self.face_emotion = self.normalize_emotion(data.get("emotion"))
        self.face_time = rospy.get_time()

    def voice_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        # Sin texto transcrito suele ser silencio o ruido: no cuenta como emoción
        if not str(data.get("text", "")).strip():
            return
        self.voice_emotion = self.normalize_emotion(data.get("final_emotion"))
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
        self.override = None if label == "auto" else self.normalize_emotion(label)
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
            sweep[0].append(ox)
            sweep[1].append(oy)
            # Ángulo respecto al eje del sensor: en la vista acumulada se prefiere
            # el punto visto más centrado
            sx, sy = su.odom_to_robot(xs, ys, sensor)
            sweep[2].append(np.abs(np.arctan2(sy, sx)))

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
        if self.behavior == "hide" and self.st.get("phase") in ("enter", "hidden"):
            self.exit_from = self.pose
        self.behavior = new_behavior
        self.st = {}
        self.sweep = None

    # ------------------------------------------------------------------
    # Utilidades de movimiento
    # ------------------------------------------------------------------
    def drive_to(self, tx, ty, tol, speed):
        """Control proporcional hacia un punto en odom. Devuelve (Twist, alcanzado)."""
        rx, ry = su.odom_to_robot(tx, ty, self.pose)
        dist, ang = math.hypot(rx, ry), math.atan2(ry, rx)
        if dist < tol:
            return make_twist(), True
        w = clip(1.8 * ang, self.max_w)
        v = min(speed, 0.6 * dist + 0.03) if abs(ang) < 0.4 else 0.0
        return make_twist(v, w), False

    def face_angle(self, ang, tol=0.1):
        """Gira en el sitio hacia un ángulo del marco del robot."""
        if abs(ang) < tol:
            return make_twist(), True
        w = clip(1.5 * ang, self.max_w)
        return make_twist(0.0, math.copysign(max(abs(w), 0.25), w)), False

    def track_person(self):
        """Actualiza la posición de la persona con el lidar.

        Solo se consideran objetos con forma de persona (su.find_people), para no
        confundirla con una pata de silla o una pared. Si no la ve, se mantiene
        la última posición conocida. Devuelve (distancia, ángulo) en el marco
        del robot.
        """
        px, py = su.odom_to_robot(self.person[0], self.person[1], self.pose)
        ang = math.atan2(py, px)
        near = (px, py) if self.person_confirmed else None
        target = su.closest_cluster(self.scan["people"], center=ang, half_width=0.5,
                                    max_dist=3.5, near=near)
        if target is not None:
            ox, oy = su.robot_to_odom(target["x"], target["y"], self.pose)
            if self.person_confirmed:
                ox = 0.6 * ox + 0.4 * self.person[0]
                oy = 0.6 * oy + 0.4 * self.person[1]
            self.person = (ox, oy)
            self.person_confirmed = True
            return target["dist"], target["angle"]
        return math.hypot(px, py), ang

    def free_space(self, direction, half_width=0.26, scan=None):
        scan = scan or self.scan
        return su.min_range_in_sector(scan["angles"], scan["ranges"], direction, half_width)

    def seen(self, distance):
        """Distancia mínima que se puede comprobar con el sensor (la cámara de
        profundidad no ve nada a menos de 0.45 m)."""
        return max(distance, self.blind_front + 0.05)

    def bumped(self):
        return rospy.get_time() - self.bump_time < 1.0

    def blocked_ahead(self):
        xs, ys = self.scan["xs"], self.scan["ys"]
        corridor = abs(ys) < self.robot_radius + 0.03
        return bool(((xs > 0) & (xs < self.seen(self.safety_distance)) & corridor).any())

    def with_detour(self, cmd):
        """Rodea un obstáculo que corta el camino (p. ej. la pata de una esquina).

        Si va a avanzar y hay algo delante, gira hacia el lado más despejado hasta
        tener el camino libre y avanza 0.3 m recto; luego vuelve a `cmd`.
        """
        d = self.st.get("detour")
        if d is None:
            if cmd.linear.x <= 0 or not self.blocked_ahead():
                return cmd
            left = self.free_space(math.pi / 3, 0.5)
            right = self.free_space(-math.pi / 3, 0.5)
            d = self.st["detour"] = {"side": 1.0 if left >= right else -1.0, "from": None}
        if self.blocked_ahead():
            d["from"] = None
            return make_twist(0.0, d["side"] * 0.6)
        if d["from"] is None:
            d["from"] = self.pose[:2]
        if math.hypot(self.pose[0] - d["from"][0], self.pose[1] - d["from"][1]) > 0.3:
            self.st["detour"] = None
            return cmd
        return make_twist(0.12, 0.0)

    def safety_filter(self, cmd):
        """Anula el avance/retroceso si hay un obstáculo en el pasillo del robot."""
        xs, ys = self.scan["xs"], self.scan["ys"]
        corridor = abs(ys) < self.robot_radius + 0.03
        ahead = self.seen(self.safety_distance)
        if cmd.linear.x > 0 and (self.bumped() or ((xs > 0) & (xs < ahead) & corridor).any()):
            cmd.linear.x = 0.0
        if cmd.linear.x < 0 and ((xs < 0) & (xs > -self.safety_distance) & corridor).any():
            cmd.linear.x = 0.0
        return cmd

    # ------------------------------------------------------------------
    # Comportamientos
    # ------------------------------------------------------------------
    def behave_idle(self):
        _, ang = self.track_person()
        cmd, _ = self.face_angle(ang, tol=0.3)
        return cmd

    def behave_approach(self, stop_distance, speed, keep_distance=False):
        """Se acerca a la persona hasta `stop_distance`.

        Con `keep_distance` también retrocede si está demasiado cerca.
        """
        dist, ang = self.track_person()
        w = clip(1.5 * ang, self.max_w)
        v = 0.0
        if abs(ang) < 0.4 and dist > stop_distance:
            v = min(speed, 0.5 * (dist - stop_distance) + 0.03)
        elif abs(ang) < 0.4 and keep_distance and dist < stop_distance - 0.1:
            v = -speed
        return make_twist(v, w)

    def behave_retreat(self):
        dist, ang = self.track_person()
        w = clip(1.5 * ang, self.max_w)
        v = -0.7 * self.max_v if dist < self.retreat_distance else 0.0
        return make_twist(v, w)

    def behave_spin(self):
        yaw = self.pose[2]
        turned = self.st.get("turned", 0.0)
        if "last_yaw" in self.st:
            turned += abs(su.normalize_angle(yaw - self.st["last_yaw"]))
        self.st["turned"], self.st["last_yaw"] = turned, yaw
        return make_twist(0.0, self.max_w) if turned < 2 * math.pi else make_twist()

    def behave_exit(self):
        """Sale de debajo de la mesa marcha atrás."""
        moved = math.hypot(self.pose[0] - self.exit_from[0], self.pose[1] - self.exit_from[1])
        if moved > self.under_table_depth + 0.35 or self.free_space(math.pi, 0.4) < self.robot_radius + 0.1:
            self.exit_from = None
            return make_twist()
        return make_twist(-0.1, 0.0)

    # --- Esconderse debajo de una mesa -------------------------------
    def detect_tables(self, scan=None):
        s = scan or self.scan
        legs = su.find_legs(s["clusters"], max_width=self.leg_max_width)
        tables = su.find_tables(legs, s["xs"], s["ys"],
                                self.table_min_spacing, self.table_max_spacing)
        bad = self.st.get("bad_legs", [])
        return [t for t in tables if self.entry_is_free(t, s) and not any(
            math.hypot(lx - bx, ly - by) < 0.15 for bx, by in bad
            for lx, ly in self.legs_in_odom(t, s["pose"]))]

    def entry_is_free(self, table, scan):
        """El pasillo por el que entraría (desde el punto previo hasta debajo de la
        mesa, con el ancho del robot) no tiene nada: así cabe entre las patas y no
        elige lados falsos cuyo punto de entrada cae encima de una silla."""
        a, b, c = table["a"], table["b"], table["c"]
        mx, my = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
        side = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        nx, ny = -(b["y"] - a["y"]) / side, (b["x"] - a["x"]) / side
        if nx * (c["x"] - mx) + ny * (c["y"] - my) < 0:
            nx, ny = -nx, -ny
        d = self.entry_distance(side)
        p0 = (mx - d * nx, my - d * ny)
        p1 = (mx + self.under_table_depth * nx, my + self.under_table_depth * ny)
        return su.segment_clearance(scan["xs"], scan["ys"], p0, p1) > self.robot_radius + 0.03

    @staticmethod
    def legs_in_odom(table, pose):
        return [su.robot_to_odom(table[k]["x"], table[k]["y"], pose) for k in "abc"]

    def entry_distance(self, side):
        """Distancia al lado de la mesa desde la que se entra.

        Con un sensor de campo estrecho, lo bastante lejos para que, mirando a la
        mesa, las dos patas del lado queden dentro del campo de visión.
        """
        if self.scan_fov >= self.sweep_fov:
            return self.pre_distance
        half = max(self.scan_fov / 2 - 0.1, 0.2)
        return max(self.pre_distance, (side / 2 + 0.08) / math.tan(half), self.blind_front + 0.3)

    def set_table(self, a, b, c):
        """Guarda la mesa en odom: a-b es el lado por el que entra y c la pata de
        la esquina, que indica hacia dónde está el interior."""
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        nx, ny = -(b[1] - a[1]) / length, (b[0] - a[0]) / length
        if nx * (c[0] - mx) + ny * (c[1] - my) < 0:
            nx, ny = -nx, -ny
        d = self.st.setdefault("entry_distance", self.entry_distance(length))
        self.st.update(legs=[a, b, c], table=(mx, my), normal=(nx, ny), side=length,
                       pre=(mx - d * nx, my - d * ny),
                       inside=(mx + self.under_table_depth * nx, my + self.under_table_depth * ny))

    def choose_table(self, table, pose):
        self.st.pop("entry_distance", None)
        a, b, c = self.legs_in_odom(table, pose)
        self.set_table(a, b, c)
        self.st["side0"] = self.st["side"]
        self.st["anchor"] = self.st["table"]
        mx, my = self.st["table"]
        px, py = self.st["pre"]
        rospy.loginfo(f"[hide] Mesa: lado de {self.st['side']:.2f} m con centro en "
                      f"({mx:.2f}, {my:.2f}); entro desde ({px:.2f}, {py:.2f})")

    def track_table(self, gate=None):
        """Corrige la mesa con las dos patas del lado de entrada si se ven ahora.

        Así el error de odometría acumulado desde que se detectó la mesa no
        afecta: el robot se guía por las patas mientras las ve. Devuelve True si
        las ha visto.
        """
        s = self.scan
        legs = su.find_legs(s["clusters"], max_width=self.leg_max_width, max_points=10 ** 6)
        a, b, c = self.st["legs"]
        ea = su.odom_to_robot(a[0], a[1], s["pose"])
        eb = su.odom_to_robot(b[0], b[1], s["pose"])
        pair = su.match_leg_pair(legs, ea, eb, self.st["side0"], gate or self.leg_gate)
        if pair is None:
            return False
        seen = [su.robot_to_odom(l["x"], l["y"], s["pose"]) for l in pair]
        # Sin gate grande, la mesa no puede alejarse mucho de donde se localizó
        # (si no, se iría enganchando a patas de sillas cercanas)
        mx = (seen[0][0] + seen[1][0]) / 2 - self.st["anchor"][0]
        my = (seen[0][1] + seen[1][1]) / 2 - self.st["anchor"][1]
        if not gate and math.hypot(mx, my) > self.leg_gate:
            return False
        # Se suaviza con la estimación anterior para no dar tirones (salvo al
        # localizarla de nuevo con un gate grande)
        k = 1.0 if gate else 0.5
        a2 = (k * seen[0][0] + (1 - k) * a[0], k * seen[0][1] + (1 - k) * a[1])
        b2 = (k * seen[1][0] + (1 - k) * b[0], k * seen[1][1] + (1 - k) * b[1])
        # c se mueve igual que el centro del lado (solo da el sentido del interior)
        dx = (a2[0] + b2[0] - a[0] - b[0]) / 2
        dy = (a2[1] + b2[1] - a[1] - b[1]) / 2
        self.set_table(a2, b2, (c[0] + dx, c[1] + dy))
        if not self.st.get("tracked"):
            rospy.loginfo("[hide] Veo las patas de la mesa, me guío por ellas")
        self.st["tracked"] = True
        return True

    def confirm_table(self):
        """Comprueba en directo, ya con las patas a y b localizadas, que la
        esquina c también está (o la tapa otra pata) y que dentro no hay nada.
        Evita meterse entre patas de sillas distintas que en la vuelta parecían
        una mesa."""
        s = self.scan
        a, b, c = (su.odom_to_robot(p[0], p[1], s["pose"]) for p in self.st["legs"])
        legs = [{"x": p[0], "y": p[1], "dist": math.hypot(*p), "angle": math.atan2(p[1], p[0])}
                for p in (a, b)]
        c_leg = {"x": c[0], "y": c[1]}
        # Si la esquina cae fuera del campo de visión o demasiado cerca no se exige
        c_visible = (abs(math.atan2(c[1], c[0])) < self.scan_fov / 2 - 0.05
                     and math.hypot(*c) > self.blind_front + 0.1)
        return ((not c_visible
                 or su._fourth_leg_possible(s["xs"], s["ys"], c[0], c[1], legs, tol=0.15))
                and su._inside_is_empty(s["xs"], s["ys"], legs[0], legs[1], c_leg))

    def give_up_table(self, reason, back_out=True, retry_view=None):
        """Descarta la mesa actual (todas sus patas) y vuelve a buscar.

        Con `back_out`, antes sale marcha atrás (se había empezado a meter debajo).
        Con `retry_view` busca otra mesa en esa misma vuelta en vez de dar otra:
        en una vuelta nueva la deriva de la odometría movería las patas
        descartadas y volvería a elegir la misma mesa.
        """
        rospy.logwarn(f"[hide] {reason}, busco otra")
        self.st.setdefault("bad_legs", []).extend(self.st["legs"])
        self.st["stall_ref"] = None
        self.st["rejected"] = self.st.get("rejected", 0) + 1
        if back_out:
            self.exit_from = self.pose
        if retry_view is not None:
            self.st["view"] = retry_view
        self.set_phase("search")

    def set_phase(self, phase):
        x, y, _ = self.pose
        rospy.loginfo(f"[hide] {self.st.get('phase')} -> {phase} en ({x:.2f}, {y:.2f})")
        self.st["phase"] = phase
        self.st["phase_start"] = rospy.get_time()
        self.st["detour"] = None

    def behave_hide(self):
        if "phase" not in self.st:
            self.st["attempts"] = 0
            self.set_phase("search")
        phase = self.st["phase"]
        elapsed = rospy.get_time() - self.st["phase_start"]

        if phase == "search":
            # Con un sensor de campo estrecho primero mira alrededor dando una vuelta
            view = self.st.pop("view", None)
            if view is None and self.scan_fov < self.sweep_fov:
                self.start_look()
                return make_twist()
            tables = self.detect_tables(view)
            if tables and self.st.get("rejected", 0) < self.max_rejected:
                self.choose_table(tables[0], (view or self.scan)["pose"])
                self.st["chosen_from"] = view
                self.set_phase("acquire" if view is not None else "goto_pre")
            elif self.st["attempts"] >= 3 or self.st.get("rejected", 0) >= self.max_rejected:
                rospy.logwarn("[hide] No encuentro ninguna mesa, me quedo quieto")
                self.set_phase("hidden")
            else:
                self.st["attempts"] += 1
                self.start_flee(view)
            return make_twist()

        if phase == "look":
            return self.behave_look(elapsed)

        if phase == "flee":
            return self.behave_flee(elapsed)

        if phase == "acquire":
            # La mesa se vio durante la vuelta, con el error de odometría del giro:
            # se mira hacia ella y se vuelven a localizar las patas en directo
            mx, my = self.st["table"]
            rx, ry = su.odom_to_robot(mx, my, self.pose)
            cmd, facing = self.face_angle(math.atan2(ry, rx), tol=0.1)
            if facing and self.track_table(gate=self.acquire_gate) and self.confirm_table():
                mx2, my2 = self.st["anchor"] = self.st["table"]
                rospy.loginfo(f"[hide] Mesa localizada de nuevo (corrección "
                              f"{math.hypot(mx2 - mx, my2 - my):.2f} m)")
                self.set_phase("goto_pre")
            elif elapsed > 8.0:
                # Si mirándola de frente no aparecen las dos patas, era una mesa
                # falsa (patas de sillas distintas que encajaban por casualidad)
                self.give_up_table("No vuelvo a ver la mesa al mirarla", back_out=False,
                                   retry_view=self.st.get("chosen_from"))
                return make_twist()
            return cmd

        if self.bump_pending and phase in ("goto_pre", "align", "enter"):
            self.give_up_table("Choque con el bumper")
            return make_twist()

        if phase == "goto_pre":
            self.track_table()
            cmd, reached = self.drive_to(*self.st["pre"], tol=0.08, speed=self.max_v)
            cmd = self.with_detour(cmd)
            if reached:
                self.set_phase("align")
            elif elapsed > 30.0:
                self.give_up_table("No consigo llegar a la mesa", back_out=False)
                return make_twist()
            return cmd

        if phase == "align":
            # Se orienta perpendicular al lado de la mesa (no hacia un punto, que
            # con un pequeño error de posición daría un ángulo torcido)
            self.track_table()
            nx, ny = self.st["normal"]
            cmd, aligned = self.face_angle(
                su.normalize_angle(math.atan2(ny, nx) - self.pose[2]), tol=0.05)
            if aligned or elapsed > 15.0:
                self.set_phase("enter")
            return cmd

        if phase == "enter":
            self.track_table()
            mx, my = self.st["table"]
            nx, ny = self.st["normal"]
            dx, dy = self.pose[0] - mx, self.pose[1] - my
            along = dx * nx + dy * ny      # < 0 todavía fuera de la mesa
            cross = -dx * ny + dy * nx     # > 0 desviado a la izquierda de la línea
            if along >= self.under_table_depth - 0.02:
                rospy.loginfo(f"[hide] Escondido debajo de la mesa "
                              f"(desvío lateral {cross:+.2f} m)")
                self.set_phase("hidden")
                return make_twist()
            limit = (self.st["entry_distance"] + self.under_table_depth) / self.enter_speed
            if elapsed > 2 * limit + 5.0:
                self.give_up_table("No consigo meterme debajo")
                return make_twist()
            # Sigue la recta perpendicular al lado que pasa por su centro
            heading = math.atan2(ny, nx) - math.atan(2.5 * cross)
            err = su.normalize_angle(heading - self.pose[2])
            v = self.enter_speed if abs(err) < 0.3 else 0.0
            return make_twist(v, clip(2.0 * err, self.max_w))

        return make_twist()  # hidden

    def check_stuck(self, cmd, now):
        """Detecta que se ha atascado al entrar bajo la mesa (manda avanzar y no avanza)."""
        if cmd.linear.x < 0.02:
            self.st["stall_ref"] = None
            return
        ref = self.st.get("stall_ref")
        if ref is None or math.hypot(self.pose[0] - ref[1], self.pose[1] - ref[2]) > self.stall_distance:
            self.st["stall_ref"] = (now, self.pose[0], self.pose[1])
        elif now - ref[0] > self.stall_time:
            self.give_up_table("Atascado al entrar (¿mesa demasiado baja?)")

    def start_look(self):
        self.sweep = ([], [], [])
        self.st["look_turned"], self.st["look_yaw"] = 0.0, self.pose[2]
        self.set_phase("look")

    def behave_look(self, elapsed):
        """Gira 360º en el sitio acumulando el /scan y arma un barrido virtual."""
        yaw = self.pose[2]
        self.st["look_turned"] += abs(su.normalize_angle(yaw - self.st["look_yaw"]))
        self.st["look_yaw"] = yaw
        timeout = 2 * (2 * math.pi / self.sweep_speed) + 5.0
        if self.st["look_turned"] < 2 * math.pi and elapsed < timeout:
            return make_twist(0.0, self.sweep_speed)
        xs, ys, centered = self.sweep
        self.sweep = None
        ox = np.concatenate(xs) if xs else np.array([])
        oy = np.concatenate(ys) if ys else np.array([])
        prio = np.concatenate(centered) if centered else np.array([])
        # Solo los puntos vistos cerca del centro del sensor: cada zona sale de unos
        # pocos scans seguidos y el error de odometría del giro no la emborrona
        near_center = prio < self.sweep_center
        if near_center.sum() > 0.2 * len(prio):
            ox, oy, prio = ox[near_center], oy[near_center], prio[near_center]
        ang, r = su.points_to_scan(ox, oy, self.pose, priority=prio)
        rospy.loginfo(f"[hide] Vuelta completa: {len(ox)} puntos, {len(r)} direcciones vistas")
        self.st["view"] = self.build_scan(ang, r, self.pose)
        if self.debug_dir:
            path = os.path.join(self.debug_dir, f"vuelta_{rospy.get_time():.0f}.npz")
            np.savez(path, xs=ox, ys=oy, priority=prio, pose=self.pose, angles=ang, ranges=r)
            rospy.loginfo(f"[hide] Vuelta guardada en {path}")
        self.set_phase("search")
        return make_twist()

    def start_flee(self, scan=None):
        """Elige la dirección más despejada lejos de la persona."""
        px, py = su.odom_to_robot(*self.person, self.pose)
        person_ang = math.atan2(py, px)
        best_dir, best_free = None, -1.0
        for k in range(24):
            direction = su.normalize_angle(k * math.pi / 12)
            if abs(su.normalize_angle(direction - person_ang)) < math.pi / 2:
                continue
            free = self.free_space(direction, scan=scan)
            if free > best_free:
                best_dir, best_free = direction, free
        self.st["flee_yaw"] = su.normalize_angle(self.pose[2] + best_dir)
        self.st["flee_from"] = self.pose
        self.set_phase("flee")

    def behave_flee(self, elapsed):
        ang = su.normalize_angle(self.st["flee_yaw"] - self.pose[2])
        moved = math.hypot(self.pose[0] - self.st["flee_from"][0],
                           self.pose[1] - self.st["flee_from"][1])
        cmd, aligned = self.face_angle(ang, tol=0.15)
        # El espacio libre se mira ya girado hacia la dirección de huida
        blocked = (aligned and self.free_space(0.0) < self.seen(self.robot_radius + 0.25)) \
            or self.bump_pending
        if moved > 1.5 or elapsed > 10.0 or blocked:
            self.set_phase("search")
            return make_twist()
        if aligned:
            cmd = make_twist(self.max_v, clip(1.5 * ang, self.max_w))
        return cmd

    # ------------------------------------------------------------------
    def step(self):
        now = rospy.get_time()
        if self.stopped:
            self.cmd_pub.publish(make_twist())
            self.state_pub.publish(String(data=json.dumps(
                {"emotion": None, "behavior": "stop", "phase": None})))
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
        cmd = self.safety_filter(cmd)
        if self.exit_from is None and self.behavior == "hide" and self.st.get("phase") == "enter":
            self.check_stuck(cmd, now)
        self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data=json.dumps({
            "emotion": self.active_emotion,
            "behavior": self.behavior,
            "phase": self.st.get("phase"),
        })))

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.pose is None or self.scan is None:
                rospy.logwarn_throttle(5, "Esperando /odom y /scan...")
            else:
                self.step()
            rate.sleep()


if __name__ == "__main__":
    try:
        TurtlebotEmotionNode().run()
    except rospy.ROSInterruptException:
        pass
