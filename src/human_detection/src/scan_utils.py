"""Procesado del LaserScan del TurtleBot (sin dependencias de ROS).

Convenio: marco del robot con x hacia delante, y hacia la izquierda y
ángulos en radianes en sentido antihorario (0 = delante).
"""
import math

import numpy as np


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def scan_to_points(ranges, angle_min, angle_increment, range_min, range_max):
    """Devuelve (angulos, distancias, xs, ys) solo de las lecturas válidas."""
    r = np.asarray(ranges, dtype=float)
    ang = angle_min + angle_increment * np.arange(len(r))
    # El LDS del TurtleBot3 devuelve 0.0 para lecturas inválidas
    valid = np.isfinite(r) & (r > max(range_min, 0.01)) & (r < range_max)
    r, ang = r[valid], ang[valid]
    return ang, r, r * np.cos(ang), r * np.sin(ang)


def min_range_in_sector(angles, ranges, center, half_width):
    """Distancia mínima dentro del sector [center - half_width, center + half_width]."""
    if len(ranges) == 0:
        return math.inf
    diff = np.abs(np.arctan2(np.sin(angles - center), np.cos(angles - center)))
    in_sector = ranges[diff <= half_width]
    return float(in_sector.min()) if len(in_sector) else math.inf


def cluster_points(xs, ys, jump=0.1):
    """Agrupa puntos consecutivos del barrido separados menos de `jump` metros.

    Devuelve una lista de arrays de índices. Une el primer y el último grupo si
    el barrido de 360º los corta por la mitad.
    """
    n = len(xs)
    if n == 0:
        return []
    gaps = np.hypot(np.diff(xs), np.diff(ys)) > jump
    breaks = np.flatnonzero(gaps) + 1
    clusters = np.split(np.arange(n), breaks)

    if len(clusters) > 1 and math.hypot(xs[0] - xs[-1], ys[0] - ys[-1]) <= jump:
        clusters[0] = np.concatenate([clusters[-1], clusters[0]])
        clusters.pop()
    return clusters


def describe_clusters(xs, ys, clusters):
    """Centroide, anchura (extremo a extremo) y número de puntos de cada grupo."""
    out = []
    for idx in clusters:
        cx, cy = float(xs[idx].mean()), float(ys[idx].mean())
        width = math.hypot(xs[idx[0]] - xs[idx[-1]], ys[idx[0]] - ys[idx[-1]])
        out.append({"x": cx, "y": cy, "width": width, "n": len(idx),
                    "dist": math.hypot(cx, cy), "angle": math.atan2(cy, cx)})
    return out


def find_legs(clusters_info, max_width=0.09, max_dist=3.5, max_points=12):
    """Grupos pequeños y aislados: candidatos a pata de mesa."""
    return [c for c in clusters_info
            if c["width"] <= max_width and c["dist"] <= max_dist and c["n"] <= max_points]


def _segment_is_free(xs, ys, a, b, margin=0.08, end_clearance=0.1):
    """Comprueba que no hay puntos del barrido sobre el segmento a-b (hueco libre)."""
    ax, ay, bx, by = a["x"], a["y"], b["x"], b["y"]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    t = ((xs - ax) * dx + (ys - ay) * dy) / (length ** 2)
    inner = (t * length > end_clearance) & ((1 - t) * length > end_clearance)
    px, py = ax + t * dx, ay + t * dy
    dist = np.hypot(xs - px, ys - py)
    return not np.any(inner & (dist < margin))


def find_tables(legs, xs, ys, min_spacing=0.4, max_spacing=1.6):
    """Parejas de patas con hueco libre entre ellas por donde puede pasar el robot.

    Devuelve una lista de dicts {left, right, mid_x, mid_y, dist}, de la más
    cercana a la más lejana.
    """
    tables = []
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            a, b = legs[i], legs[j]
            spacing = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if not (min_spacing <= spacing <= max_spacing):
                continue
            if not _segment_is_free(xs, ys, a, b):
                continue
            mx, my = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
            tables.append({"a": a, "b": b, "mid_x": mx, "mid_y": my,
                           "dist": math.hypot(mx, my)})
    tables.sort(key=lambda t: t["dist"])
    return tables


def closest_cluster(clusters_info, center, half_width, max_dist=3.0, near=None, near_radius=0.7):
    """Grupo más cercano dentro de un sector frontal.

    Si se da `near` (x, y), solo se consideran los grupos a menos de
    `near_radius` de ese punto (seguimiento de la persona).
    """
    best = None
    for c in clusters_info:
        if c["dist"] > max_dist:
            continue
        if abs(normalize_angle(c["angle"] - center)) > half_width:
            continue
        if near is not None and math.hypot(c["x"] - near[0], c["y"] - near[1]) > near_radius:
            continue
        if best is None or c["dist"] < best["dist"]:
            best = c
    return best


# ---------------------------------------------------------------------------
# Transformaciones entre el marco odom y el marco del robot
# ---------------------------------------------------------------------------
def robot_to_odom(x, y, pose):
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return px + c * x - s * y, py + s * x + c * y


def odom_to_robot(x, y, pose):
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x - px, y - py
    return c * dx + s * dy, -s * dx + c * dy
