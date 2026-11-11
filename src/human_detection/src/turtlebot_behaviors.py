"""Comportamientos del TurtleBot2: cómo se mueve con cada emoción.

Este módulo es la mitad "motora" de turtlebot_emotion_node.py. La clase
TurtlebotBehaviors es un mixin: no funciona sola, sino como clase base de
TurtlebotEmotionNode, que le da el estado (self.pose, self.scan, self.person,
self.st, ...) y la llama una vez por ciclo de control (10 Hz) desde step().
Cada método behave_*() devuelve el Twist que hay que mandar en ese ciclo.

  Emoción            Comportamiento     Método
  feliz              approach           behave_approach (hasta approach_distance)
  triste             comfort            behave_approach (despacio, comfort_distance)
  enojo / disgusto   hide               behave_hide     (máquina de estados, abajo)
  miedo              retreat            behave_retreat
  sorpresa           spin               behave_spin     (360º medidos con odometría)
  neutral            idle               behave_idle     (mira a la persona)

Esconderse (behave_hide) es una máquina de estados guardada en self.st["phase"]:

  search ──(sensor estrecho)──> look ──> search
    │  sin mesa: flee (se aleja de la persona) ──> search   (3 intentos y se rinde)
    └─ con mesa ──> acquire ──> goto_pre ──> align ──> enter ──> check ──> hidden
                    (vuelve a ver    (ruta fija   (perpen-  (recta     │
                     las patas)       por odom)    dicular)  al centro) └─> correct ──> check
  En cualquier fase, un choque, un atasco o no llegar descarta la mesa
  (give_up_table) y vuelve a search.

Convenios: posiciones en metros en el marco odom; ángulos en radianes,
positivos en sentido antihorario. Las funciones geométricas puras están en
scan_utils.py (su).
"""
import math
import os

import numpy as np
import rospy
from geometry_msgs.msg import Twist

import scan_utils as su


def clip(value, limit):
    """Limita value a [-limit, limit]."""
    return max(-limit, min(limit, value))


def make_twist(v=0.0, w=0.0):
    """Twist con velocidad lineal v (m/s) y angular w (rad/s)."""
    t = Twist()
    t.linear.x = v
    t.angular.z = w
    return t


class TurtlebotBehaviors:
    """Mixin con los comportamientos; ver la descripción del módulo."""

    def init_behavior_params(self):
        """Parámetros de ROS propios de los comportamientos (sobre todo de esconderse)."""
        # Mesas y sillas: valen si el robot cabe. Lados (entre centros de pata) de al
        # menos 45 cm (ancho del robot más el grosor de una pata y holgura)
        self.table_min_spacing = rospy.get_param("~table_min_spacing", 0.45)
        # Holgura mínima a cada lado del robot en el pasillo de entrada
        self.entry_margin = rospy.get_param("~entry_margin", 0.02)
        self.table_max_spacing = rospy.get_param("~table_max_spacing", 1.6)
        # Mesas de dos patas (pie central o en T): columnas de al menos este ancho
        # y como mucho a esta distancia entre sí
        self.two_leg_tables = rospy.get_param("~two_leg_tables", True)
        self.two_leg_min_width = rospy.get_param("~two_leg_min_width", 0.05)
        self.two_leg_max_spacing = rospy.get_param("~two_leg_max_spacing", 1.8)
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
        # Al esconderse va más despacio: las ruedas patinan menos y la odometría
        # (de la que depende la ruta, que no se recalcula) se desvía menos
        self.hide_speed = rospy.get_param("~hide_speed", 0.1)
        self.hide_turn = rospy.get_param("~hide_turn", 0.5)
        self.enter_speed = rospy.get_param("~enter_speed", 0.06)
        # Tolerancia de la comprobación de "estoy debajo" (aproximada)
        self.under_tolerance = rospy.get_param("~under_tolerance", 0.1)
        self.max_corrections = rospy.get_param("~max_corrections", 2)
        # Carpeta donde guardar cada vuelta de búsqueda (.npz) para analizarla
        self.debug_dir = rospy.get_param("~debug_dir", "")
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
        # Si al entrar bajo la mesa avanza menos de stall_distance en stall_time
        # segundos, se ha atascado (tablero demasiado bajo): sale y busca otra
        self.stall_time = rospy.get_param("~stall_time", 2.0)
        self.stall_distance = rospy.get_param("~stall_distance", 0.03)
        # Si el /scan cubre menos de este ángulo (p. ej. el de la cámara de
        # profundidad, ~60º), antes de buscar mesa da una vuelta acumulando puntos
        self.sweep_fov = math.radians(rospy.get_param("~sweep_below_fov_deg", 300.0))
        self.sweep_speed = rospy.get_param("~sweep_speed", 0.35)
        # En la vuelta solo cuentan los puntos a menos de este ángulo del eje del sensor
        self.sweep_center = rospy.get_param("~sweep_center", 0.15)
        # En los últimos turn_slowdown grados de un giro frena hasta turn_min_angular
        # (rad/s, por debajo el Kobuki puede no moverse) y para a turn_tolerance grados
        self.turn_slowdown = math.radians(rospy.get_param("~turn_slowdown_deg", 25.0))
        self.turn_min_angular = rospy.get_param("~turn_min_angular", 0.2)
        self.turn_tolerance = math.radians(rospy.get_param("~turn_tolerance_deg", 1.0))

    # ------------------------------------------------------------------
    # Utilidades de movimiento
    # ------------------------------------------------------------------
    def drive_to(self, tx, ty, tol, speed, w_max=None):
        """Control proporcional hacia un punto en odom. Devuelve (Twist, alcanzado)."""
        rx, ry = su.odom_to_robot(tx, ty, self.pose)
        dist, ang = math.hypot(rx, ry), math.atan2(ry, rx)
        if dist < tol:
            return make_twist(), True
        w = clip(1.8 * ang, w_max or self.max_w)
        v = min(speed, 0.6 * dist + 0.03) if abs(ang) < 0.4 else 0.0
        return make_twist(v, w), False

    def face_angle(self, ang, tol=0.1, w_max=None):
        """Gira en el sitio hacia un ángulo del marco del robot."""
        if abs(ang) < tol:
            return make_twist(), True
        w = clip(1.5 * ang, w_max or self.max_w)
        return make_twist(0.0, math.copysign(max(abs(w), 0.2), w)), False

    def ramp(self, cmd):
        """Limita la aceleración (frenar sí puede ser inmediato)."""
        last, dt = self.last_cmd, 1.0 / self.rate_hz
        for attr, acc in (("x", self.acc_linear), ("z", self.acc_angular)):
            vec = "linear" if attr == "x" else "angular"
            new, old = getattr(getattr(cmd, vec), attr), getattr(getattr(last, vec), attr)
            if abs(new) > abs(old) or new * old < 0:
                limit = acc * dt
                setattr(getattr(cmd, vec), attr, max(old - limit, min(old + limit, new)))
        return cmd

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

    def turn_rate(self, remaining, w_max):
        """Velocidad de un giro medido con odometría (frena al final, 0 = terminado)."""
        return su.turn_rate(remaining, w_max, self.turn_slowdown,
                            self.turn_min_angular, self.turn_tolerance)

    def behave_spin(self):
        """Da una vuelta de 360º medida con la odometría."""
        yaw, now = self.pose[2], rospy.get_time()
        turned = self.st.get("turned", 0.0)
        if "last_yaw" in self.st:
            turned += abs(su.normalize_angle(yaw - self.st["last_yaw"]))
        else:
            self.st["spin_start"] = now
        self.st["turned"], self.st["last_yaw"] = turned, yaw
        if self.st.get("spin_done"):
            return make_twist()
        w = self.turn_rate(2 * math.pi - turned, self.max_w)
        # Si tarda 3 veces lo esperado (atascado), se da por terminada
        if w == 0.0 or now - self.st["spin_start"] > 3 * 2 * math.pi / self.max_w + 5.0:
            rospy.loginfo(f"[spin] Vuelta terminada: {math.degrees(turned):.0f}º según odometría")
            self.st["spin_done"] = True
            return make_twist()
        return make_twist(0.0, w)

    def start_exit(self):
        """Empieza a salir marcha atrás: lo que se haya metido más un margen."""
        depth = self.st.get("depth")
        self.exit_distance = (depth / 2 if depth else self.under_table_depth) + 0.4
        self.exit_from = self.pose

    def behave_exit(self):
        """Sale de debajo de la mesa marcha atrás."""
        moved = math.hypot(self.pose[0] - self.exit_from[0], self.pose[1] - self.exit_from[1])
        if moved > self.exit_distance or self.free_space(math.pi, 0.4) < self.robot_radius + 0.1:
            self.exit_from = None
            return make_twist()
        return make_twist(-0.1, 0.0)

    # --- Esconderse debajo de una mesa -------------------------------
    def find_all_tables(self, s):
        """Mesas de 4 patas (o sillas) y, si no las tapan, mesas de 2 patas."""
        legs = su.find_legs(s["clusters"], max_width=self.leg_max_width)
        tables = su.find_tables(legs, s["xs"], s["ys"],
                                self.table_min_spacing, self.table_max_spacing)
        if self.two_leg_tables:
            used = {id(t[k]) for t in tables for k in "abc"}
            tables += [t for t in su.find_two_leg_tables(
                legs, s["xs"], s["ys"], self.table_min_spacing, self.two_leg_max_spacing,
                self.two_leg_min_width) if id(t["a"]) not in used and id(t["b"]) not in used]
        return tables

    def detect_tables(self, scan=None):
        s = scan or self.scan
        tables = self.find_all_tables(s)
        bad = self.st.get("bad_legs", [])
        ok = [t for t in tables if self.entry_is_free(t, s) and not any(
            math.hypot(lx - bx, ly - by) < 0.15 for bx, by in bad
            for lx, ly in self.legs_in_odom(t, s["pose"]))]
        # La más grande primero (en la que mejor cabe); de una misma mesa, el lado
        # más largo (más holgura al entrar); a igualdad, la más cercana
        def key(t):
            pts = [(t[k]["x"], t[k]["y"]) for k in "abc"]
            _, _, side, depth = self.table_geometry(*pts)
            if t.get("two_leg"):
                depth = side / 2   # el tablero va de lado a lado; fondo supuesto
            return (-round(side * depth / 0.05), -round(side, 1), t["dist"])
        return sorted(ok, key=key)

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
        depth = abs((c["x"] - a["x"]) * nx + (c["y"] - a["y"]) * ny)
        p0 = (mx - d * nx, my - d * ny)
        p1 = (mx + depth / 2 * nx, my + depth / 2 * ny)
        return (su.segment_clearance(scan["xs"], scan["ys"], p0, p1)
                > self.robot_radius + self.entry_margin)

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

    @staticmethod
    def table_geometry(a, b, c):
        """Centro del lado a-b, normal hacia el interior, ancho y fondo de la mesa."""
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        nx, ny = -(b[1] - a[1]) / length, (b[0] - a[0]) / length
        if nx * (c[0] - mx) + ny * (c[1] - my) < 0:
            nx, ny = -nx, -ny
        depth = abs((c[0] - a[0]) * nx + (c[1] - a[1]) * ny)
        return (mx, my), (nx, ny), length, depth

    def set_table(self, a, b, c):
        """Guarda la mesa en odom: a-b es el lado por el que entra y c la pata de
        la esquina, que indica hacia dónde está el interior. El objetivo es el
        centro de la mesa (o silla)."""
        (mx, my), (nx, ny), length, depth = self.table_geometry(a, b, c)
        d = self.st.setdefault("entry_distance", self.entry_distance(length))
        k = depth / 2
        self.st.update(legs=[a, b, c], table=(mx, my), normal=(nx, ny), side=length,
                       depth=depth, pre=(mx - d * nx, my - d * ny),
                       inside=(mx + k * nx, my + k * ny))

    def choose_table(self, table, pose):
        self.st.pop("entry_distance", None)
        self.st.pop("route", None)
        self.st["corrections"] = 0
        self.st["two_leg"] = bool(table.get("two_leg"))
        a, b, c = self.legs_in_odom(table, pose)
        self.set_table(a, b, c)
        self.st["side0"] = self.st["side"]
        self.st["anchor"] = self.st["table"]
        mx, my = self.st["table"]
        px, py = self.st["pre"]
        kind = ("de 2 patas, " if self.st["two_leg"] else "") + \
            f"de {self.st['side']:.2f} x {self.st['depth']:.2f} m"
        rospy.loginfo(f"[hide] Mesa {kind}, lado con centro en ({mx:.2f}, {my:.2f}); entro desde ({px:.2f}, {py:.2f})")

    def track_table(self, gate=None):
        """Corrige la mesa con las dos patas del lado de entrada si se ven ahora.

        Así el error de odometría acumulado desde que se detectó la mesa no
        afecta: el robot se guía por las patas mientras las ve. Devuelve True si
        las ha visto.
        """
        s = self.scan
        legs = su.find_legs(s["clusters"], max_width=self.leg_max_width)
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
        # c se mueve con el lado a-b (desplazamiento y giro), así el error de
        # rumbo de la vuelta también se corrige en la esquina
        self.set_table(a2, b2, su.move_with_segment(c, a, b, a2, b2))
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
        if self.st.get("two_leg"):
            # Sin esquina: basta con que entre las dos columnas no haya nada
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            return su.segment_clearance(s["xs"], s["ys"], (mx, my), (mx, my)) > 0.25
        legs = [{"x": p[0], "y": p[1], "dist": math.hypot(*p), "angle": math.atan2(p[1], p[0])}
                for p in (a, b)]
        c_leg = {"x": c[0], "y": c[1]}
        # Si la esquina cae fuera del campo de visión o demasiado cerca no se exige
        c_visible = (abs(math.atan2(c[1], c[0])) < self.scan_fov / 2 - 0.05
                     and math.hypot(*c) > self.blind_front + 0.1)
        return ((not c_visible
                 or su._fourth_leg_possible(s["xs"], s["ys"], c[0], c[1], legs, tol=0.15))
                and su._inside_is_empty(s["xs"], s["ys"], legs[0], legs[1], c_leg))

    def plan_route(self):
        """Ruta fija hasta el punto de entrada, rodeando la propia mesa.

        Si la recta desde el robot cruza la mesa (con el ancho del robot), se
        rodea por sus esquinas (a una distancia segura) por el lado más corto.
        """
        a, b, c = self.st["legs"]
        _, (nx, ny), length, depth = self.table_geometry(a, b, c)
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        to_xy = lambda p: (a[0] + p[0] * ux + p[1] * nx, a[1] + p[0] * uy + p[1] * ny)
        to_uv = lambda p: ((p[0] - a[0]) * ux + (p[1] - a[1]) * uy,
                           (p[0] - a[0]) * nx + (p[1] - a[1]) * ny)
        pre = self.st["pre"]
        robot = to_uv(self.pose)
        r = self.robot_radius + 0.05

        def crosses(p, q):
            for k in np.linspace(0.0, 1.0, 30):
                x, y = p[0] + k * (q[0] - p[0]), p[1] + k * (q[1] - p[1])
                if -r < x < length + r and -r < y < depth + r:
                    return True
            return False

        if not crosses(robot, to_uv(pre)):
            return [pre]
        m = self.robot_radius + 0.2
        corners = [(-m, -m), (length + m, -m), (length + m, depth + m), (-m, depth + m)]
        start = min(range(4), key=lambda i: math.hypot(corners[i][0] - robot[0],
                                                       corners[i][1] - robot[1]))
        best = None
        for step in (1, -1):
            path, i = [corners[start]], start
            while i not in (0, 1):
                i = (i + step) % 4
                path.append(corners[i])
            pts = [robot] + path + [to_uv(pre)]
            cost = sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(pts, pts[1:]))
            if best is None or cost < best[0]:
                best = (cost, path)
        return [to_xy(p) for p in best[1]] + [pre]

    def under_check(self):
        """¿Está el robot debajo? Es decir, entre 2 o 4 patas (aproximado).

        Se mira en el marco de la mesa: a lo ancho entre las patas a y b, y de
        fondo entre el lado a-b y el de enfrente, con `under_tolerance` de margen.
        Si el /scan ve ahora alguna de las patas cerca de donde se esperaban, la
        mesa se desplaza a lo que ve; si no (la cámara no ve a menos de 0.45 m),
        se usa la posición guardada y la odometría. Devuelve
        (ok, ancho, fondo, fuente) y, si corrige, actualiza el objetivo.
        """
        a, b, c = self.st["legs"]
        source = "odometría"
        s = self.scan
        legs = su.find_legs(s["clusters"], max_width=self.leg_max_width)
        offsets = []
        # En una mesa de 2 patas c es ficticia (pegada a b): no se busca en el scan
        for p in ((a, b) if self.st.get("two_leg") else (a, b, c)):
            ex, ey = su.odom_to_robot(p[0], p[1], s["pose"])
            best = min(legs, key=lambda l: math.hypot(l["x"] - ex, l["y"] - ey), default=None)
            if best is not None and math.hypot(best["x"] - ex, best["y"] - ey) < self.leg_gate:
                ox, oy = su.robot_to_odom(best["x"], best["y"], s["pose"])
                offsets.append((ox - p[0], oy - p[1]))
        if offsets:
            dx = sum(o[0] for o in offsets) / len(offsets)
            dy = sum(o[1] for o in offsets) / len(offsets)
            a, b, c = [(p[0] + dx, p[1] + dy) for p in (a, b, c)]
            self.set_table(a, b, c)
            source = f"scan, {len(offsets)} patas"
        (mx, my), (nx, ny), length, depth = self.table_geometry(a, b, c)
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        px, py = self.pose[0] - a[0], self.pose[1] - a[1]
        u = px * ux + py * uy          # a lo ancho, desde la pata a
        v = px * nx + py * ny          # de fondo, desde el lado a-b
        tol = self.under_tolerance
        ok = -tol <= u <= length + tol and -tol <= v <= depth + tol
        return ok, u, v, source

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
            self.start_exit()
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
            cmd, facing = self.face_angle(math.atan2(ry, rx), tol=0.1, w_max=self.hide_turn)
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

        if self.bump_pending and phase in ("goto_pre", "align", "enter", "correct"):
            self.give_up_table("Choque con el bumper")
            return make_twist()

        # A partir de aquí la ruta está fijada: no se recalcula la mesa, se va por
        # odometría (despacio) y al llegar se comprueba si está debajo
        if phase == "goto_pre":
            route = self.st.get("route")
            if route is None:
                route = self.st["route"] = self.plan_route()
                if len(route) > 1:
                    rospy.loginfo(f"[hide] La mesa está en medio: la rodeo por "
                                  f"{len(route) - 1} esquina(s)")
            last = len(route) == 1
            cmd, reached = self.drive_to(*route[0], tol=0.06 if last else 0.12,
                                         speed=self.hide_speed, w_max=self.hide_turn)
            cmd = self.with_detour(cmd)
            if reached and not last:
                route.pop(0)
                return make_twist()
            if reached:
                self.set_phase("align")
            elif elapsed > 60.0:
                self.give_up_table("No consigo llegar a la mesa", back_out=False)
                return make_twist()
            return cmd

        if phase == "align":
            # Se orienta perpendicular al lado de la mesa (no hacia un punto, que
            # con un pequeño error de posición daría un ángulo torcido)
            if elapsed < 0.5:
                return make_twist()   # se para un momento antes de girar
            nx, ny = self.st["normal"]
            cmd, aligned = self.face_angle(
                su.normalize_angle(math.atan2(ny, nx) - self.pose[2]), tol=0.04,
                w_max=self.hide_turn)
            if aligned or elapsed > 20.0:
                self.set_phase("enter")
            return cmd

        if phase == "enter":
            if elapsed < 0.5:
                return make_twist()
            mx, my = self.st["table"]
            nx, ny = self.st["normal"]
            dx, dy = self.pose[0] - mx, self.pose[1] - my
            along = dx * nx + dy * ny      # < 0 todavía fuera de la mesa
            cross = -dx * ny + dy * nx     # > 0 desviado a la izquierda de la línea
            if along >= self.st["depth"] / 2 - 0.02:
                self.set_phase("check")
                return make_twist()
            limit = (self.st["entry_distance"] + self.st["depth"]) / self.enter_speed
            if elapsed > 2 * limit + 5.0:
                self.give_up_table("No consigo meterme debajo")
                return make_twist()
            # Sigue la recta perpendicular al lado que pasa por su centro
            heading = math.atan2(ny, nx) - math.atan(2.5 * cross)
            err = su.normalize_angle(heading - self.pose[2])
            v = self.enter_speed if abs(err) < 0.3 else 0.0
            return make_twist(v, clip(2.0 * err, self.hide_turn))

        if phase == "check":
            if elapsed < 0.7:
                return make_twist()   # quieto para medir
            ok, u, v, source = self.under_check()
            corrections = self.st.get("corrections", 0)
            if ok:
                rospy.loginfo(f"[hide] Escondido: estoy entre las patas (según {source}; "
                              f"{u:.2f} m a lo ancho y {v:.2f} m de fondo)")
                self.set_phase("hidden")
            elif corrections < self.max_corrections:
                self.st["corrections"] = corrections + 1
                rospy.logwarn(f"[hide] No estoy debajo (según {source}; {u:.2f} m a lo ancho, "
                              f"{v:.2f} m de fondo), corrijo ({corrections + 1})")
                self.set_phase("correct")
            else:
                self.give_up_table("No consigo quedarme debajo")
            return make_twist()

        if phase == "correct":
            cmd, reached = self.drive_to(*self.st["inside"], tol=0.04, speed=self.enter_speed,
                                         w_max=self.hide_turn)
            if reached or elapsed > 20.0:
                self.set_phase("check")
                return make_twist()
            return cmd

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
        self.sweep = []
        self.st["look_turned"], self.st["look_yaw"] = 0.0, self.pose[2]
        self.set_phase("look")

    def behave_look(self, elapsed):
        """Gira 360º en el sitio acumulando el /scan y arma un barrido virtual."""
        yaw = self.pose[2]
        self.st["look_turned"] += abs(su.normalize_angle(yaw - self.st["look_yaw"]))
        self.st["look_yaw"] = yaw
        timeout = 2 * (2 * math.pi / self.sweep_speed) + 5.0
        w = self.turn_rate(2 * math.pi - self.st["look_turned"], self.sweep_speed)
        if w > 0.0 and elapsed < timeout:
            return make_twist(0.0, w)
        chunks = list(self.sweep or [])
        self.sweep = None
        ox = np.concatenate([c[0] for c in chunks]) if chunks else np.array([])
        oy = np.concatenate([c[1] for c in chunks]) if chunks else np.array([])
        prio = np.concatenate([c[2] for c in chunks]) if chunks else np.array([])
        # Solo los puntos vistos cerca del centro del sensor: cada zona sale de unos
        # pocos scans seguidos y el error de odometría del giro no la emborrona
        near_center = prio < self.sweep_center
        if near_center.sum() > 0.2 * len(prio):
            ox, oy, prio = ox[near_center], oy[near_center], prio[near_center]
        ang, r = su.points_to_scan(ox, oy, self.pose, priority=prio)
        rospy.loginfo(f"[hide] Vuelta completa: {len(ox)} puntos, {len(r)} direcciones vistas")
        self.st["view"] = self.build_scan(ang, r, self.pose)
        self.store_map_view(self.st["view"])
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
