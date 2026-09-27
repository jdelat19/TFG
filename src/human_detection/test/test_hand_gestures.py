#!/usr/bin/env python3
"""Pruebas de los gestos por forma de la mano con manos sintéticas."""
import math
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import gestures as g

IMAGE_SHAPE = (480, 640)  # (alto, ancho)
WRIST = (320, 300)

# Base de cada dedo relativa a la muñeca (mano vertical, dedos hacia arriba)
KNUCKLES = {"index": (-30, -100), "middle": (-10, -105), "ring": (10, -100), "pinky": (30, -90)}
FINGER_IDS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
              "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def make_hand(extended=(), thumb="folded", angle=0.0, offset=(0, 0)):
    """Devuelve los 21 puntos (px) de una mano.

    extended: dedos extendidos ("index", "middle", "ring", "pinky")
    thumb: "folded", "out" (hacia el lado) o "touch_index" (círculo de OK)
    angle: giro de la mano alrededor de la muñeca (radianes)
    """
    pts = [None] * 21
    pts[0] = (0, 0)
    for name, (mcp, pip, dip, tip) in FINGER_IDS.items():
        kx, ky = KNUCKLES[name]
        pts[mcp] = (kx, ky)
        if name in extended:
            pts[pip], pts[dip], pts[tip] = (kx, ky - 35), (kx, ky - 60), (kx, ky - 80)
        else:
            pts[pip], pts[dip], pts[tip] = (kx, ky - 30), (kx + 3, ky - 10), (kx + 5, ky + 10)

    pts[1], pts[2] = (-30, -20), (-50, -45)
    if thumb == "out":
        pts[3], pts[4] = (-75, -65), (-100, -80)
    elif thumb == "touch_index":
        # Índice doblado hacia el pulgar para cerrar el círculo
        pts[6], pts[7], pts[8] = (-45, -125), (-55, -110), (-58, -95)
        pts[3], pts[4] = (-60, -75), (-60, -90)
    else:
        pts[3], pts[4] = (-35, -65), (-5, -70)

    c, s = math.cos(angle), math.sin(angle)
    return [(WRIST[0] + offset[0] + c * x - s * y, WRIST[1] + offset[1] + s * x + c * y)
            for x, y in pts]


def to_results(pts, side="right"):
    h, w = IMAGE_SHAPE
    lm = SimpleNamespace(landmark=[SimpleNamespace(x=x / w, y=y / h) for x, y in pts])
    return SimpleNamespace(
        left_hand_landmarks=lm if side == "left" else None,
        right_hand_landmarks=lm if side == "right" else None,
        pose_landmarks=None, face_landmarks=None)


def winner(results, gestures):
    """Mismo criterio que GestureDetector.detect_gestures: el de mayor prioridad."""
    detected = [gest for gest in gestures if gest.check(results, IMAGE_SHAPE)]
    return max(detected, key=lambda gest: gest.priority).name if detected else "Ninguno"


def fresh_gestures():
    return [type(gest)() for gest in g.DEFAULT_GESTURES]


ALL = ("index", "middle", "ring", "pinky")
THUMB_DOWN_ANGLE = math.radians(-128)  # gira la mano para que el pulgar apunte abajo

CASES = {
    "puño": dict(),
    "mano_abierta": dict(extended=ALL, thumb="out"),
    "ok": dict(extended=("middle", "ring", "pinky"), thumb="touch_index"),
    "cuernos": dict(extended=("index", "pinky")),
    "llamame": dict(extended=("pinky",), thumb="out"),
    "pulgar_abajo": dict(thumb="out", angle=THUMB_DOWN_ANGLE),
}


def test_finger_states():
    states = g.finger_states(make_hand(extended=("index", "pinky"), thumb="out"))
    assert states == {"index": True, "middle": False, "ring": False, "pinky": True, "thumb": True}
    assert not any(g.finger_states(make_hand()).values())


def test_each_hand_shape_is_recognized():
    for expected, kwargs in CASES.items():
        for side in ("left", "right"):
            got = winner(to_results(make_hand(**kwargs), side), fresh_gestures())
            assert got == expected, f"{expected} ({side}) detectado como {got}"


def test_hand_shapes_work_when_rotated():
    for expected, kwargs in CASES.items():
        if expected == "pulgar_abajo":
            continue  # depende de la orientación a propósito
        for deg in (-30, 25):
            hand = make_hand(**dict(kwargs, angle=math.radians(deg)))
            got = winner(to_results(hand), fresh_gestures())
            assert got == expected, f"{expected} girado {deg}º detectado como {got}"


def test_thumb_sideways_is_not_thumbs_down():
    hand = make_hand(thumb="out")  # pulgar hacia el lado, no hacia abajo
    assert not g.ThumbsDownGesture().check(to_results(hand), IMAGE_SHAPE)


def test_wave_needs_side_to_side_motion():
    gestures = fresh_gestures()
    open_hand = dict(extended=ALL, thumb="out")
    for _ in range(20):
        still = winner(to_results(make_hand(**open_hand)), gestures)
    assert still == "mano_abierta"

    gestures = fresh_gestures()
    for i in range(20):
        dx = 60 * math.sin(i * math.pi / 4)
        got = winner(to_results(make_hand(**open_hand, offset=(dx, 0))), gestures)
    assert got == "saludar"
