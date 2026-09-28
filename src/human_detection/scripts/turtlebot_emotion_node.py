#!/usr/bin/env python3
"""Comportamiento del TurtleBot según el estado de ánimo detectado.

Entradas:
  /human_state      (String JSON)  emoción facial publicada por ros_node.py
  /voice_emotion    (String JSON)  emoción por voz publicada por voice_node.py
  /emotion_override (String)       fuerza una emoción para pruebas ("auto" la desactiva)
  /scan             (LaserScan)    lidar del TurtleBot
  /odom             (Odometry)

Salidas:
  /cmd_vel                 (Twist)
  /turtlebot_emotion/state (String JSON)  emoción y comportamiento activos

Comportamientos:
  feliz            -> se acerca a la persona
  triste           -> se acerca despacio y se queda a más distancia (acompañar)
  enojo / disgusto -> busca una mesa con el lidar y se esconde debajo
  miedo            -> retrocede manteniendo distancia con la persona
  sorpresa         -> da una vuelta sobre sí mismo
  neutral          -> se queda quieto mirando a la persona

Se asume que al arrancar el robot está mirando hacia la persona.
"""
import json
import math
import os
import sys

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

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

# Anchura del pasillo (medio ancho) que se comprueba para no chocar
ROBOT_HALF_WIDTH = 0.13


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
        self.approach_distance = rospy.get_param("~approach_distance", 0.5)
        self.comfort_distance = rospy.get_param("~comfort_distance", 0.8)
        self.retreat_distance = rospy.get_param("~retreat_distance", 1.5)
        self.safety_distance = rospy.get_param("~safety_distance", 0.2)
        self.emotion_hold = rospy.get_param("~emotion_hold", 1.5)
        self.voice_timeout = rospy.get_param("~voice_timeout", 6.0)
        self.person_initial_distance = rospy.get_param("~person_initial_distance", 1.0)
        self.leg_max_width = rospy.get_param("~leg_max_width", 0.09)
        self.table_min_spacing = rospy.get_param("~table_min_spacing", 0.4)
        self.table_max_spacing = rospy.get_param("~table_max_spacing", 1.6)
        self.under_table_depth = rospy.get_param("~under_table_depth", 0.3)
        self.behaviors = dict(BEHAVIORS, **rospy.get_param("~behaviors", {}))

        self.pose = None      # (x, y, yaw) en odom
        self.scan = None      # último barrido procesado
        self.person = None    # posición estimada de la persona en odom
        self.person_confirmed = False

        self.face_emotion = None
        self.voice_emotion = None
        self.voice_time = 0.0
        self.override = None

        self.active_emotion = None
        self.candidate = None
        self.candidate_since = 0.0
        self.behavior = "idle"
        self.st = {}          # estado interno del comportamiento activo
        self.exit_from = None  # pose al empezar a salir de debajo de la mesa

        self.cmd_pub = rospy.Publisher(
            rospy.get_param("~cmd_vel_topic", "/cmd_vel"), Twist, queue_size=1)
        self.state_pub = rospy.Publisher("/turtlebot_emotion/state", String, queue_size=1)

        rospy.Subscriber("/human_state", String, self.human_callback, queue_size=1)
        rospy.Subscriber("/voice_emotion", String, self.voice_callback, queue_size=1)
        rospy.Subscriber("/emotion_override", String, self.override_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan,
                         self.scan_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/odom"), Odometry,
                         self.odom_callback, queue_size=1)

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

    def voice_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        self.voice_emotion = self.normalize_emotion(data.get("final_emotion"))
        self.voice_time = rospy.get_time()

    def override_callback(self, msg):
        label = msg.data.strip().lower()
        self.override = None if label == "auto" else self.normalize_emotion(label)
        rospy.loginfo(f"Override de emoción: {self.override or 'desactivado'}")

    def scan_callback(self, msg):
        ang, r, xs, ys = su.scan_to_points(
            msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max)
        clusters = su.describe_clusters(xs, ys, su.cluster_points(xs, ys))
        self.scan = {"angles": ang, "ranges": r, "xs": xs, "ys": ys, "clusters": clusters}

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)
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
        return self.face_emotion

    def update_emotion(self, now):
        emotion = self.fused_emotion(now)
        if emotion is None:
            return
        if emotion != self.candidate:
            self.candidate, self.candidate_since = emotion, now
        hold = 0.0 if self.override else self.emotion_hold
        if emotion != self.active_emotion and now - self.candidate_since >= hold:
            self.set_behavior(emotion)

    def set_behavior(self, emotion):
        new_behavior = self.behaviors.get(emotion, "idle")
        rospy.loginfo(f"Emoción: {emotion} -> comportamiento: {new_behavior}")
        self.active_emotion = emotion
        if new_behavior == self.behavior:
            return
        # Si estaba debajo de la mesa, primero sale marcha atrás
        if self.behavior == "hide" and self.st.get("phase") in ("enter", "hidden"):
            self.exit_from = self.pose
        self.behavior = new_behavior
        self.st = {}

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

        Devuelve (distancia, ángulo) en el marco del robot.
        """
        px, py = su.odom_to_robot(self.person[0], self.person[1], self.pose)
        ang = math.atan2(py, px)
        near = (px, py) if self.person_confirmed else None
        target = su.closest_cluster(self.scan["clusters"], center=ang, half_width=0.5,
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

    def free_space(self, direction, half_width=0.26):
        return su.min_range_in_sector(self.scan["angles"], self.scan["ranges"],
                                      direction, half_width)

    def safety_filter(self, cmd):
        """Anula el avance/retroceso si hay un obstáculo en el pasillo del robot."""
        xs, ys = self.scan["xs"], self.scan["ys"]
        corridor = abs(ys) < ROBOT_HALF_WIDTH
        if cmd.linear.x > 0 and ((xs > 0) & (xs < self.safety_distance) & corridor).any():
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
        if moved > self.under_table_depth + 0.35 or self.free_space(math.pi, 0.4) < 0.25:
            self.exit_from = None
            return make_twist()
        return make_twist(-0.1, 0.0)

    # --- Esconderse debajo de una mesa -------------------------------
    def detect_tables(self):
        s = self.scan
        legs = su.find_legs(s["clusters"], max_width=self.leg_max_width)
        return su.find_tables(legs, s["xs"], s["ys"],
                              self.table_min_spacing, self.table_max_spacing)

    def set_table(self, table):
        """Guarda la mesa en odom y calcula el punto previo y el punto de debajo."""
        ax, ay = su.robot_to_odom(table["a"]["x"], table["a"]["y"], self.pose)
        bx, by = su.robot_to_odom(table["b"]["x"], table["b"]["y"], self.pose)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        length = math.hypot(bx - ax, by - ay)
        nx, ny = -(by - ay) / length, (bx - ax) / length
        # La normal debe apuntar desde el robot hacia el interior de la mesa
        if nx * (mx - self.pose[0]) + ny * (my - self.pose[1]) < 0:
            nx, ny = -nx, -ny
        self.st["table"] = (mx, my)
        self.st["pre"] = (mx - 0.35 * nx, my - 0.35 * ny)
        self.st["inside"] = (mx + self.under_table_depth * nx, my + self.under_table_depth * ny)

    def refine_table(self):
        """Actualiza la mesa guardada si se vuelve a ver cerca de donde estaba."""
        for table in self.detect_tables():
            mx, my = su.robot_to_odom(table["mid_x"], table["mid_y"], self.pose)
            if math.hypot(mx - self.st["table"][0], my - self.st["table"][1]) < 0.25:
                self.set_table(table)
                return

    def set_phase(self, phase):
        rospy.loginfo(f"[hide] {self.st.get('phase')} -> {phase}")
        self.st["phase"] = phase
        self.st["phase_start"] = rospy.get_time()

    def behave_hide(self):
        if "phase" not in self.st:
            self.st["attempts"] = 0
            self.set_phase("search")
        phase = self.st["phase"]
        elapsed = rospy.get_time() - self.st["phase_start"]

        if phase == "search":
            tables = self.detect_tables()
            if tables:
                self.set_table(tables[0])
                self.set_phase("goto_pre")
            elif self.st["attempts"] >= 3:
                rospy.logwarn("[hide] No encuentro ninguna mesa, me quedo quieto")
                self.set_phase("hidden")
            else:
                self.st["attempts"] += 1
                self.start_flee()
            return make_twist()

        if phase == "flee":
            return self.behave_flee(elapsed)

        if phase == "goto_pre":
            self.refine_table()
            cmd, reached = self.drive_to(*self.st["pre"], tol=0.08, speed=self.max_v)
            if reached:
                self.set_phase("align")
            elif elapsed > 25.0:
                self.set_phase("search")
            return cmd

        if phase == "align":
            ix, iy = su.odom_to_robot(*self.st["inside"], self.pose)
            cmd, aligned = self.face_angle(math.atan2(iy, ix), tol=0.08)
            if aligned:
                self.set_phase("enter")
            return cmd

        if phase == "enter":
            cmd, reached = self.drive_to(*self.st["inside"], tol=0.06, speed=0.1)
            if reached or elapsed > 15.0:
                rospy.loginfo("[hide] Escondido debajo de la mesa")
                self.set_phase("hidden")
            return cmd

        return make_twist()  # hidden

    def start_flee(self):
        """Elige la dirección más despejada lejos de la persona."""
        px, py = su.odom_to_robot(*self.person, self.pose)
        person_ang = math.atan2(py, px)
        best_dir, best_free = None, -1.0
        for k in range(24):
            direction = su.normalize_angle(k * math.pi / 12)
            if abs(su.normalize_angle(direction - person_ang)) < math.pi / 2:
                continue
            free = self.free_space(direction)
            if free > best_free:
                best_dir, best_free = direction, free
        self.st["flee_yaw"] = su.normalize_angle(self.pose[2] + best_dir)
        self.st["flee_from"] = self.pose
        self.set_phase("flee")

    def behave_flee(self, elapsed):
        ang = su.normalize_angle(self.st["flee_yaw"] - self.pose[2])
        moved = math.hypot(self.pose[0] - self.st["flee_from"][0],
                           self.pose[1] - self.st["flee_from"][1])
        if moved > 1.5 or elapsed > 10.0 or self.free_space(0.0) < 0.4:
            self.set_phase("search")
            return make_twist()
        cmd, aligned = self.face_angle(ang, tol=0.15)
        if aligned:
            cmd = make_twist(self.max_v, clip(1.5 * ang, self.max_w))
        return cmd

    # ------------------------------------------------------------------
    def step(self):
        now = rospy.get_time()
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

        self.cmd_pub.publish(self.safety_filter(cmd))
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
