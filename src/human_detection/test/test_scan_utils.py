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


def test_narrow_sensor_sweep_rebuilds_360_view():
    """Con un sensor de ~63º (cámara de profundidad) la mesa de detrás no se ve
    desde la pose inicial, pero sí en el barrido virtual tras dar una vuelta."""
    fov, n = math.radians(63), 720
    half = fov / 2
    obstacles = [PERSON] + TABLE_LEGS

    def narrow_points(pose):
        ranges = raycast(obstacles, ROOM, pose=pose, n=n)
        ang, r, xs, ys = su.scan_to_points(ranges, 0.0, 2 * math.pi / n, 0.12, 3.5)
        front = np.abs(np.arctan2(np.sin(ang), np.cos(ang))) <= half
        return xs[front], ys[front]

    start = (0.0, 0.0, 0.0)
    xs, ys = narrow_points(start)
    clusters = su.describe_clusters(xs, ys, su.cluster_points(xs, ys))
    assert su.find_tables(su.find_legs(clusters), xs, ys) == []

    all_x, all_y = [], []
    for k in range(24):  # vuelta completa en pasos de 15º
        pose = (0.0, 0.0, k * 2 * math.pi / 24)
        xs, ys = narrow_points(pose)
        ox, oy = su.robot_to_odom(xs, ys, pose)
        all_x.append(ox)
        all_y.append(oy)
    ang, r = su.points_to_scan(np.concatenate(all_x), np.concatenate(all_y), start)
    assert np.all(np.diff(ang) > 0)
    xs, ys = r * np.cos(ang), r * np.sin(ang)
    clusters = su.describe_clusters(xs, ys, su.cluster_points(xs, ys))
    tables = su.find_tables(su.find_legs(clusters), xs, ys)
    assert tables
    assert tables[0]["mid_x"] < -0.9  # la mesa está detrás del robot


def test_points_to_scan_priority_prefers_centered_points():
    # Dos observaciones de la misma dirección: una cerca (vista de lado, prioridad
    # alta) y otra lejos (vista de frente). Con prioridad gana la de frente.
    xs, ys = [1.0, 2.0], [0.0, 0.0]
    _, r = su.points_to_scan(xs, ys, (0.0, 0.0, 0.0))
    assert np.isclose(r[0], 1.0)
    _, r = su.points_to_scan(xs, ys, (0.0, 0.0, 0.0), priority=[0.5, 0.01])
    assert np.isclose(r[0], 2.0)


def test_match_leg_pair_uses_spacing():
    def leg(x, y):
        return {"x": x, "y": y}
    a, b = leg(1.0, 0.45), leg(1.0, -0.45)
    back = leg(1.6, 0.45)   # pata del fondo, más cerca de lo esperado para a
    legs = [back, a, b]
    # La estimación está desplazada 0.4 m hacia el fondo
    pair = su.match_leg_pair(legs, (1.4, 0.45), (1.4, -0.45), spacing=0.9, gate=0.6)
    assert pair == (a, b)
    assert su.match_leg_pair(legs, (1.4, 0.45), (1.4, -0.45), spacing=0.9, gate=0.2) is None
    assert su.match_leg_pair([a], (1.0, 0.45), (1.0, -0.45), spacing=0.9, gate=0.3) is None


def test_segment_clearance():
    xs, ys = np.array([1.0, 0.5]), np.array([0.3, -2.0])
    assert np.isclose(su.segment_clearance(xs, ys, (0.0, 0.0), (2.0, 0.0)), 0.3)
    assert su.segment_clearance([], [], (0.0, 0.0), (1.0, 0.0)) == math.inf


def test_move_with_segment_rotates_and_translates():
    # El lado a-b gira 90º alrededor del origen: la esquina c gira con él
    p = su.move_with_segment((1.0, 1.0), (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5))
    assert np.allclose(p, (-1.0, 1.0))


def test_turn_rate_frena_al_final_y_para_dentro_de_la_tolerancia():
    w = 1.2
    assert su.turn_rate(math.pi, w) == w                       # lejos: velocidad máxima
    assert 0.2 < su.turn_rate(math.radians(10), w) < w          # cerca: frena
    assert su.turn_rate(math.radians(2), w) == 0.2              # nunca por debajo del mínimo
    assert su.turn_rate(math.radians(0.5), w) == 0.0            # dentro de la tolerancia: para
    assert su.turn_rate(-0.1, w) == 0.0                         # pasado de largo: para


def test_fat_legs_are_legs_not_walls():
    # Columnas de 10 cm vistas de cerca con mucha resolución (cámara de
    # profundidad): muchos puntos por pata, pero siguen siendo patas
    columns = [(-0.8, 0.6, 0.05), (-0.8, -0.6, 0.05)]
    ranges = raycast(columns, ROOM, n=2000)
    _, _, _, _, clusters = process(ranges)
    legs = su.find_legs(clusters)
    assert len(legs) == 2
    assert all(leg["n"] > 12 for leg in legs)


def test_two_leg_table_is_found_and_entered_from_robot_side():
    # Mesa de pie central o en T: dos columnas de 8 cm separadas 1.2 m
    columns = [(1.5, 0.6, 0.04), (1.5, -0.6, 0.04)]
    ranges = raycast(columns, ROOM, n=1000)
    _, _, xs, ys, clusters = process(ranges)
    legs = su.find_legs(clusters)
    assert su.find_tables(legs, xs, ys) == []          # no hay esquina
    tables = su.find_two_leg_tables(legs, xs, ys)
    assert len(tables) == 1
    t = tables[0]
    assert abs(t["mid_x"] - 1.5) < 0.05 and abs(t["mid_y"]) < 0.05
    assert t["c"]["x"] > t["mid_x"]                     # el interior queda lejos del robot


def test_two_leg_table_needs_fat_legs_and_free_middle():
    thin = [(1.5, 0.6, 0.012), (1.5, -0.6, 0.012)]      # patas de silla
    _, _, xs, ys, clusters = process(raycast(thin, ROOM, n=1000))
    assert su.find_two_leg_tables(su.find_legs(clusters), xs, ys) == []
    columns = [(1.5, 0.6, 0.04), (1.5, -0.6, 0.04)]
    box = [((1.65, -0.1), (1.65, 0.1))]                 # algo debajo, cerca del centro
    _, _, xs, ys, clusters = process(raycast(columns, ROOM + box, n=1000))
    assert su.find_two_leg_tables(su.find_legs(clusters), xs, ys) == []


def test_person_legs_are_not_a_two_leg_table():
    person = [(1.5, 0.12, 0.06), (1.5, -0.12, 0.06)]
    other = [(1.5, 1.0, 0.06), (1.5, 0.78, 0.06)]
    _, _, xs, ys, clusters = process(raycast(person + other, ROOM, n=1000))
    assert su.find_two_leg_tables(su.find_legs(clusters), xs, ys) == []
