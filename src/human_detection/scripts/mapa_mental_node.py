#!/usr/bin/env python3
"""Ventana con el "mapa mental" del TurtleBot, visto desde arriba.

Dibuja lo que publica turtlebot_emotion_node.py en /turtlebot_emotion/map, en
coordenadas de odometría y centrado en el robot:

  gris claro   última vuelta de búsqueda (barrido virtual de 360º)
  negro        /scan actual
  naranja      patas candidatas (círculos pequeños)
  verde fino   lados de mesa encontrados en la última vuelta
  rojo         mesa elegida (patas a, b y esquina c) y su contorno
  verde        pasillo de entrada: punto previo -> debajo de la mesa, y la ruta
               fija hasta él (discontinua)
  gris (x)     patas de mesas descartadas
  magenta      persona (relleno si la está viendo con el /scan)
  azul         robot, su rumbo, el campo de visión y el camino recorrido

Teclas: + / - zoom, c borra el camino, Esc cierra.
"""
import json
import math

import cv2
import numpy as np
import rospy
from std_msgs.msg import String

SIZE = 760
ROBOT_RADIUS = 0.177

BG = (255, 255, 255)
GRID = (238, 238, 238)
VIEW = (205, 205, 205)
SCAN = (30, 30, 30)
LEG = (0, 140, 255)
TABLE_CAND = (120, 200, 120)
TABLE = (0, 0, 220)
ENTRY = (0, 160, 0)
BAD = (150, 150, 150)
PERSON = (200, 0, 200)
ROBOT = (200, 80, 0)
TRAIL = (240, 180, 120)
TEXT = (40, 40, 40)


class MentalMap:
    def __init__(self):
        rospy.init_node("mapa_mental")
        self.scale = rospy.get_param("~pixels_per_meter", 110.0)
        self.data = None
        self.trail = []
        rospy.Subscriber("/turtlebot_emotion/map", String, self.callback, queue_size=1)

    def callback(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        x, y, _ = data["pose"]
        if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.02:
            self.trail.append((x, y))
            self.trail = self.trail[-3000:]
        self.data = data

    # ------------------------------------------------------------------
    def to_px(self, p, center):
        return (int(round(SIZE / 2 + (p[0] - center[0]) * self.scale)),
                int(round(SIZE / 2 - (p[1] - center[1]) * self.scale)))

    def draw(self):
        img = np.full((SIZE, SIZE, 3), BG, np.uint8)
        d = self.data
        if d is None:
            cv2.putText(img, "Esperando /turtlebot_emotion/map...", (20, SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1, cv2.LINE_AA)
            return img
        px, py, yaw = d["pose"]
        c = (px, py)
        P = lambda p: self.to_px(p, c)

        # Cuadrícula de 1 m alineada con odom
        half = SIZE / 2 / self.scale
        for k in range(int(math.floor(px - half)), int(math.ceil(px + half)) + 1):
            u = P((k, 0))[0]
            cv2.line(img, (u, 0), (u, SIZE), GRID, 1)
        for k in range(int(math.floor(py - half)), int(math.ceil(py + half)) + 1):
            v = P((0, k))[1]
            cv2.line(img, (0, v), (SIZE, v), GRID, 1)
        cv2.circle(img, P((0, 0)), 4, (180, 180, 180), 1)   # origen de odom

        # Campo de visión del sensor
        fov = d.get("fov") or 2 * math.pi
        if fov < 2 * math.pi - 0.1:
            reach = 3.5
            pts = [P((px, py))] + [P((px + reach * math.cos(yaw + a), py + reach * math.sin(yaw + a)))
                                   for a in np.linspace(-fov / 2, fov / 2, 20)]
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(pts, np.int32)], (250, 238, 225))
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        view = d.get("view")
        if view:
            for p in view["points"]:
                cv2.circle(img, P(p), 1, VIEW, -1)
            for a, b, cc in view["tables"]:
                cv2.line(img, P(a), P(b), TABLE_CAND, 1, cv2.LINE_AA)
                cv2.line(img, P(a), P(cc), TABLE_CAND, 1, cv2.LINE_AA)
            for p in view["legs"]:
                cv2.circle(img, P(p), 5, LEG, 1, cv2.LINE_AA)

        if len(self.trail) > 1:
            cv2.polylines(img, [np.array([P(p) for p in self.trail], np.int32)], False, TRAIL, 2)

        for p in d["scan"]:
            cv2.circle(img, P(p), 1, SCAN, -1)
        for p in d["scan_legs"]:
            cv2.circle(img, P(p), 4, LEG, 2, cv2.LINE_AA)

        for p in d["bad_legs"]:
            u, v = P(p)
            cv2.line(img, (u - 5, v - 5), (u + 5, v + 5), BAD, 2)
            cv2.line(img, (u - 5, v + 5), (u + 5, v - 5), BAD, 2)

        t = d.get("table")
        if t and "legs" in t:
            a, b, cc = t["legs"]
            fourth = (b[0] + cc[0] - a[0], b[1] + cc[1] - a[1])
            # c es la esquina contigua a a o a b: se dibuja el rectángulo correcto
            if math.hypot(cc[0] - a[0], cc[1] - a[1]) > math.hypot(cc[0] - b[0], cc[1] - b[1]):
                fourth = (a[0] + cc[0] - b[0], a[1] + cc[1] - b[1])
                corners = [a, b, cc, fourth]
            else:
                corners = [b, a, cc, fourth]
            cv2.polylines(img, [np.array([P(p) for p in corners], np.int32)], True, TABLE, 1,
                          cv2.LINE_AA)
            cv2.line(img, P(a), P(b), TABLE, 3, cv2.LINE_AA)
            for name, p in (("a", a), ("b", b), ("c", cc)):
                cv2.circle(img, P(p), 6, TABLE, -1, cv2.LINE_AA)
                cv2.putText(img, name, (P(p)[0] + 7, P(p)[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, TABLE, 1, cv2.LINE_AA)
            if "pre" in t and "inside" in t:
                cv2.line(img, P(t["pre"]), P(t["inside"]), ENTRY, 2, cv2.LINE_AA)
                cv2.circle(img, P(t["pre"]), 7, ENTRY, 2, cv2.LINE_AA)
                cv2.drawMarker(img, P(t["inside"]), ENTRY, cv2.MARKER_CROSS, 14, 2)
                # Ancho del robot a lo largo del pasillo
                nx, ny = t.get("normal", (0, 0))
                r = ROBOT_RADIUS
                for s in (-1, 1):
                    q0 = (t["pre"][0] - s * ny * r, t["pre"][1] + s * nx * r)
                    q1 = (t["inside"][0] - s * ny * r, t["inside"][1] + s * nx * r)
                    cv2.line(img, P(q0), P(q1), (170, 220, 170), 1, cv2.LINE_AA)

        route = d.get("route") or []
        if route:
            pts = [P((px, py))] + [P(p) for p in route]
            for p, q in zip(pts, pts[1:]):
                n = max(1, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 8))
                for k in range(0, n, 2):   # línea discontinua
                    a = (p[0] + (q[0] - p[0]) * k // n, p[1] + (q[1] - p[1]) * k // n)
                    b = (p[0] + (q[0] - p[0]) * (k + 1) // n, p[1] + (q[1] - p[1]) * (k + 1) // n)
                    cv2.line(img, a, b, ENTRY, 2, cv2.LINE_AA)
            for p in route[:-1]:
                cv2.circle(img, P(p), 4, ENTRY, -1, cv2.LINE_AA)

        if d.get("person"):
            cv2.circle(img, P(d["person"]), 9, PERSON, -1 if d.get("person_confirmed") else 2,
                       cv2.LINE_AA)

        # Robot
        rr = max(2, int(ROBOT_RADIUS * self.scale))
        color = (0, 0, 255) if d.get("bumper") else ROBOT
        cv2.circle(img, P((px, py)), rr, color, 2, cv2.LINE_AA)
        cv2.line(img, P((px, py)), P((px + 0.3 * math.cos(yaw), py + 0.3 * math.sin(yaw))),
                 color, 2, cv2.LINE_AA)

        # Texto de estado
        behavior = d.get("behavior") or "-"
        if d.get("exiting"):
            behavior += " (saliendo marcha atrás)"
        lines = [f"emocion: {d.get('emotion') or '-'}",
                 f"comportamiento: {behavior}",
                 f"fase: {d.get('phase') or '-'}",
                 f"odom: x={px:.2f} y={py:.2f} yaw={math.degrees(yaw):.0f}",
                 f"escala: {self.scale:.0f} px/m (+/-)"]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1,
                        cv2.LINE_AA)
        legend = [("scan", SCAN), ("vuelta", VIEW), ("patas", LEG), ("mesa", TABLE),
                  ("entrada", ENTRY), ("persona", PERSON), ("robot", ROBOT)]
        x = 10
        for name, col in legend:
            cv2.circle(img, (x + 5, SIZE - 14), 5, col, -1)
            cv2.putText(img, name, (x + 14, SIZE - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1,
                        cv2.LINE_AA)
            x += 22 + 8 * len(name)
        return img

    def run(self):
        rate = rospy.Rate(10)
        cv2.namedWindow("Mapa mental del TurtleBot", cv2.WINDOW_NORMAL)
        while not rospy.is_shutdown():
            cv2.imshow("Mapa mental del TurtleBot", self.draw())
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("+"), ord("=")):
                self.scale = min(400.0, self.scale * 1.25)
            elif key == ord("-"):
                self.scale = max(20.0, self.scale / 1.25)
            elif key == ord("c"):
                self.trail = []
            rate.sleep()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        MentalMap().run()
    except rospy.ROSInterruptException:
        pass
