#!/usr/bin/env python3
"""Pruebas de scan_utils con un lidar simulado (python3 -m pytest test/)."""
import math
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import scan_utils as su


def raycast(circles, walls, pose=(0.0, 0.0, 0.0), n=360, max_range=3.5):
    """Barrido 2D contra círculos (x, y, r) y segmentos ((x1, y1), (x2, y2))."""
    px, py, yaw = pose
    ranges = []
    for i in range(n):
        a = yaw + 2 * math.pi * i / n
        dx, dy = math.cos(a), math.sin(a)
        best = math.inf
        for cx, cy, r in circles:
            fx, fy = px - cx, py - cy
            b = fx * dx + fy * dy
            disc = b * b - (fx * fx + fy * fy - r * r)
            if disc >= 0:
                t = -b - math.sqrt(disc)
                if t > 0:
                    best = min(best, t)
        for (x1, y1), (x2, y2) in walls:
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            t = ((x1 - px) * ey - (y1 - py) * ex) / den
            u = ((x1 - px) * dy - (y1 - py) * dx) / den
            if t > 0 and 0 <= u <= 1:
                best = min(best, t)
        ranges.append(best if best < max_range else 0.0)
    return ranges


def process(ranges, range_max=3.5):
    ang, r, xs, ys = su.scan_to_points(ranges, 0.0, 2 * math.pi / len(ranges), 0.12, range_max)
    clusters = su.describe_clusters(xs, ys, su.cluster_points(xs, ys))
    return ang, r, xs, ys, clusters


ROOM = [((-3, -3), (3, -3)), ((3, -3), (3, 3)), ((3, 3), (-3, 3)), ((-3, 3), (-3, -3))]
PERSON = (1.2, 0.0, 0.18)
TABLE_LEGS = [(-1.0, 0.6, 0.025), (-1.0, 1.3, 0.025), (-2.0, 0.6, 0.025), (-2.0, 1.3, 0.025)]


def test_invalid_readings_are_dropped():
    ang, r, _, _ = su.scan_to_points([0.0, math.inf, 1.0, float("nan"), 5.0], 0.0, 0.1, 0.12, 3.5)
    assert list(r) == [1.0]
    assert ang[0] == 0.2


def test_detects_table_and_ignores_person():
    ranges = raycast([PERSON] + TABLE_LEGS, ROOM)
    _, _, xs, ys, clusters = process(ranges)

    legs = su.find_legs(clusters)
    assert len(legs) == 4
    for leg in legs:
        assert min(math.hypot(leg["x"] - lx, leg["y"] - ly) for lx, ly, _ in TABLE_LEGS) < 0.05

    tables = su.find_tables(legs, xs, ys)
    assert tables, "debería encontrar al menos una mesa"
    # La más cercana es el lado frontal (-1.0, 0.6)-(-1.0, 1.3)
    assert abs(tables[0]["mid_x"] + 1.0) < 0.05 and abs(tables[0]["mid_y"] - 0.95) < 0.05


def test_blocked_gap_is_not_a_table():
    box_between = [((-1.0, 0.8), (-1.0, 1.1))]
    ranges = raycast(TABLE_LEGS[:2], ROOM + box_between)
    _, _, xs, ys, clusters = process(ranges)
    legs = su.find_legs(clusters)
    assert su.find_tables(legs, xs, ys) == []


def test_wall_seen_sideways_is_not_a_leg():
    # Pasillo de 3 m con la mesa al fondo (mundo emotion_far_table): de lejos la
    # pared se rompe en puntos sueltos que no deben emparejarse con las patas
    corridor = [((-8, 1.5), (3, 1.5)), ((-8, -1.5), (3, -1.5)),
                ((-8, -1.5), (-8, 1.5)), ((3, -1.5), (3, 1.5))]
    legs_xy = [(-4.73, 0.32, 0.025), (-4.73, -0.32, 0.025),
               (-5.67, 0.32, 0.025), (-5.67, -0.32, 0.025)]
    ranges = raycast(legs_xy, corridor, pose=(-2.77, 0.0, 0.0), max_range=12.0)
    _, _, xs, ys, clusters = process(ranges, range_max=12.0)

    legs = su.find_legs(clusters)
    assert all(abs(leg["y"]) < 0.4 for leg in legs), "un punto de la pared cuenta como pata"
    tables = su.find_tables(legs, xs, ys)
    assert tables
    # Lado de la mesa más cercano, en el marco del robot (x = -4.73 + 2.77)
    assert abs(tables[0]["mid_x"] + 1.96) < 0.05 and abs(tables[0]["mid_y"]) < 0.05


def chair_legs(x, y, half=0.2, r=0.015):
    return [(x + sx * half, y + sy * half, r) for sx in (-1, 1) for sy in (-1, 1)]


def test_chairs_are_not_tables():
    # Dos sillas juntas (como en emotion_furniture): sus patas quedan a 0.5 m
    # unas de otras, pero no forman una esquina con lados de al menos 0.5 m
    chairs = chair_legs(-1.5, 1.0) + chair_legs(-2.0, 1.0)
    ranges = raycast(chairs, ROOM)
    _, _, xs, ys, clusters = process(ranges)
    legs = su.find_legs(clusters)
    assert len(legs) >= 3
    assert su.find_tables(legs, xs, ys, min_spacing=0.5) == []


def test_three_legs_without_fourth_is_not_a_table():
    # Tres patas sueltas en ángulo recto (p. ej. de sillas distintas) y suelo
    # libre donde iría la cuarta, (-1.0, 1.3), que no queda tapada por nada
    ranges = raycast([TABLE_LEGS[0], TABLE_LEGS[2], TABLE_LEGS[3]], ROOM)
    _, _, xs, ys, clusters = process(ranges)
    assert su.find_tables(su.find_legs(clusters), xs, ys) == []


def test_hidden_fourth_leg_still_counts():
    # (-2.0, 1.3) queda justo detrás de (-1.0, 0.6) visto desde el robot
    ranges = raycast(TABLE_LEGS[:3], ROOM)
    _, _, xs, ys, clusters = process(ranges)
    assert su.find_tables(su.find_legs(clusters), xs, ys)


def test_table_diagonal_is_not_a_side():
    ranges = raycast(TABLE_LEGS, ROOM)
    _, _, xs, ys, clusters = process(ranges)
    tables = su.find_tables(su.find_legs(clusters), xs, ys)
    for t in tables:
        side = math.hypot(t["a"]["x"] - t["b"]["x"], t["a"]["y"] - t["b"]["y"])
        assert side < 1.1, "la diagonal (1.22 m) no es un lado"


def test_closest_cluster_finds_person_in_front():
    ranges = raycast([PERSON] + TABLE_LEGS, ROOM)
    ang, r, _, _, clusters = process(ranges)
    target = su.closest_cluster(clusters, center=0.0, half_width=0.5)
    assert target is not None
    assert abs(target["dist"] - 1.05) < 0.1
    assert abs(su.min_range_in_sector(ang, r, 0.0, 0.2) - 1.02) < 0.02


def test_person_is_not_confused_with_chairs():
    # Una silla delante del robot, más cerca que la persona (como en emotion_furniture)
    ranges = raycast([PERSON] + chair_legs(0.6, 0.0), ROOM)
    _, _, _, _, clusters = process(ranges)
    target = su.closest_cluster(su.find_people(clusters), center=0.0, half_width=0.5)
    assert target is not None
    assert abs(target["x"] - 1.05) < 0.1 and abs(target["y"]) < 0.1


def test_two_legs_are_one_person():
    # Persona real: dos piernas de 12 cm separadas 25 cm, a 1.5 m
    legs = [(1.5, 0.125, 0.06), (1.5, -0.125, 0.06)]
    ranges = raycast(legs, ROOM)
    _, _, _, _, clusters = process(ranges)
    people = su.find_people(clusters)
    assert any(abs(p["x"] - 1.45) < 0.1 and abs(p["y"]) < 0.05 for p in people)


def test_table_legs_are_not_people():
    ranges = raycast(TABLE_LEGS, ROOM)
    _, _, _, _, clusters = process(ranges)
    assert su.find_people(clusters) == []


def test_frame_round_trip():
    pose = (1.0, -2.0, 0.7)
    x, y = su.robot_to_odom(0.3, -0.4, pose)
    assert np.allclose(su.odom_to_robot(x, y, pose), (0.3, -0.4))
