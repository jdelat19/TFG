"""Genera videos/disgusto.mp4 a partir de videos/neutral.mp4 cambiando la cara del robot.

Uso: python3 media/make_disgusto.py
"""
import math, os, subprocess, sys
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "videos", "neutral.mp4")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "videos", "disgusto.mp4")
X0, Y0, X1, Y1 = 565, 182, 675, 268       # zona de la cara que se redibuja
S = 4                                     # supersampling para suavizar el trazo


def stroke(img, pts, thick, dx, dy):
    p = np.array([[(x + dx - X0) * S, (y + dy - Y0) * S] for x, y in pts], np.int32)
    cv2.polylines(img, [p], False, 0, thick * S, cv2.LINE_AA)
    for q in (p[0], p[-1]):                               # extremos redondeados
        cv2.circle(img, tuple(int(v) for v in q), thick * S // 2, 0, -1, cv2.LINE_AA)


def curve(x0, x1, f, n=24):
    return [(x0 + (x1 - x0) * t / n, f(x0 + (x1 - x0) * t / n)) for t in range(n + 1)]


def face(i):
    h, w = (Y1 - Y0) * S, (X1 - X0) * S
    img = np.full((h, w), 255, np.uint8)
    # "Hervor" de línea como en los vídeos originales y escalofrío en dos momentos
    rng = np.random.default_rng(i // 2)
    dx, dy = rng.integers(-1, 2), rng.integers(-1, 2)
    if 30 <= i < 46 or 68 <= i < 80:
        dx += round(2.2 * math.sin(i * 2.4))
    # Cejas bajas hacia el centro (la izquierda más marcada)
    stroke(img, [(578, 194), (589, 198), (601, 203)], 6, dx, dy)
    stroke(img, [(637, 202), (648, 199), (659, 197)], 6, dx, dy)
    # Ojo izquierdo entrecerrado (línea) y ojo derecho abierto
    stroke(img, [(584, 212), (599, 213)], 7, dx, dy)
    cv2.ellipse(img, ((648 + dx - X0) * S, (213 + dy - Y0) * S), (5 * S, 7 * S), 0, 0, 360, 0, -1, cv2.LINE_AA)
    # Nariz arrugada: dos pliegues
    for yy in (222, 227):
        stroke(img, curve(612, 624, lambda x, yy=yy: yy + 1.2 * math.sin((x - 612) / 12 * 2 * math.pi)), 2, dx, dy)
    # Boca hacia abajo y ondulada (mueca)
    mouth = curve(600, 640, lambda x: 243 - 4.5 * math.sin((x - 600) / 40 * math.pi)
                  + 1.6 * math.sin((x - 600) / 40 * 4 * math.pi) + 0.08 * (x - 600))
    stroke(img, mouth, 6, dx, dy)
    # Lengua fuera ("puaj"), que se mueve un poco
    tl = 11 + 2 * math.sin(i * 0.45)
    cx = 612
    cy = 243 - 4.5 * math.sin((cx - 600) / 40 * math.pi) + 1.6 * math.sin((cx - 600) / 40 * 4 * math.pi) + 0.08 * (cx - 600) + 1
    tongue = [(cx - 6.5, cy)] + curve(cx - 6.5, cx + 6.5, lambda x: cy + tl * math.sqrt(max(0, 1 - ((x - cx) / 6.5) ** 2)), 16) + [(cx + 6.5, cy)]
    p = np.array([[(x + dx - X0) * S, (y + dy - Y0) * S] for x, y in tongue], np.int32)
    cv2.fillPoly(img, [p], 255, cv2.LINE_AA)
    cv2.polylines(img, [p], False, 0, 4 * S, cv2.LINE_AA)
    stroke(img, [(cx, cy + 2), (cx, cy + tl * 0.55)], 2, dx, dy)
    return cv2.resize(img, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture(SRC)
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", "1280x720", "-r", "25", "-i", "-", "-c:v", "libx264", "-profile:v", "main",
                       "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", OUT],
                      stdin=subprocess.PIPE)
rng = np.random.default_rng(0)
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    region = frame[Y0:Y1, X0:X1]
    # Borrar la cara neutra con el blanco de la pantalla (con su grano)
    white = np.clip(251 + rng.normal(0, 1.5, region.shape[:2]), 0, 255).astype(np.uint8)
    feat = face(i)
    patch = np.minimum(white, feat)
    patch = cv2.GaussianBlur(patch, (3, 3), 0.6)
    frame[Y0:Y1, X0:X1] = cv2.merge([patch] * 3)
    ff.stdin.write(frame.tobytes())
    i += 1
ff.stdin.close(); ff.wait()
print("fotogramas:", i)
