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


def process(ranges):
    ang, r, xs, ys = su.scan_to_points(ranges, 0.0, 2 * math.pi / len(ranges), 0.12, 3.5)
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


def test_closest_cluster_finds_person_in_front():
    ranges = raycast([PERSON] + TABLE_LEGS, ROOM)
    ang, r, _, _, clusters = process(ranges)
    target = su.closest_cluster(clusters, center=0.0, half_width=0.5)
    assert target is not None
    assert abs(target["dist"] - 1.05) < 0.1
    assert abs(su.min_range_in_sector(ang, r, 0.0, 0.2) - 1.02) < 0.02


def test_frame_round_trip():
    pose = (1.0, -2.0, 0.7)
    x, y = su.robot_to_odom(0.3, -0.4, pose)
    assert np.allclose(su.odom_to_robot(x, y, pose), (0.3, -0.4))
