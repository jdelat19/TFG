"""Procesado del LaserScan del TurtleBot (sin dependencias de ROS).

Convenio: marco del robot con x hacia delante, y hacia la izquierda y
ángulos en radianes en sentido antihorario (0 = delante).
"""
import math

import numpy as np


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def turn_rate(remaining, w_max, slowdown=math.radians(25), w_min=0.2, tol=math.radians(1)):
    """Velocidad angular (positiva) para un giro medido con odometría.

    `remaining` es lo que falta por girar (rad). Por debajo de `tol` devuelve 0
    (giro terminado); en los últimos `slowdown` rad frena proporcionalmente sin
    bajar de `w_min`, por debajo de la cual el Kobuki puede no moverse. Así no
    se pasa al cortar de golpe a velocidad alta.
    """
    if remaining < tol:
        return 0.0
    return max(w_min, min(w_max, w_max * remaining / slowdown))


def scan_to_points(ranges, angle_min, angle_increment, range_min, range_max):
    """Devuelve (angulos, distancias, xs, ys) solo de las lecturas válidas."""
    r = np.asarray(ranges, dtype=float)
    ang = angle_min + angle_increment * np.arange(len(r))
    # Algunos lidar (RPLidar, LDS) devuelven 0.0 para lecturas inválidas
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


def describe_clusters(xs, ys, clusters, surface_gap=0.5, line_tol=0.05):
    """Centroide, anchura (extremo a extremo) y número de puntos de cada grupo.

    `isolated` es False si el grupo es un trozo de una superficie más grande: una
    pared vista muy de lado se rompe en puntos sueltos, pero cada uno queda
    alineado con sus vecinos del barrido (a menos de `surface_gap` metros).
    """
    n = len(xs)
    out = []
    for idx in clusters:
        cx, cy = float(xs[idx].mean()), float(ys[idx].mean())
        width = math.hypot(xs[idx[0]] - xs[idx[-1]], ys[idx[0]] - ys[idx[-1]])
        prev, nxt = (idx[0] - 1) % n, (idx[-1] + 1) % n
        isolated = True
        if len(clusters) > 1:
            gap_prev = math.hypot(xs[prev] - xs[idx[0]], ys[prev] - ys[idx[0]])
            gap_next = math.hypot(xs[nxt] - xs[idx[-1]], ys[nxt] - ys[idx[-1]])
            if gap_prev < surface_gap and gap_next < surface_gap:
                isolated = _point_segment_dist(cx, cy, xs[prev], ys[prev],
                                               xs[nxt], ys[nxt]) > line_tol
        out.append({"x": cx, "y": cy, "width": width, "n": len(idx),
                    "dist": math.hypot(cx, cy), "angle": math.atan2(cy, cx),
                    "isolated": isolated})
    return out


def _point_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def find_legs(clusters_info, max_width=0.15, max_dist=3.5, max_points=None):
    """Grupos pequeños y aislados: candidatos a pata de mesa.

    El límite es de anchura, no de puntos: con la cámara de profundidad (o la
    vuelta reconstruida a 0.5º) una pata gorda de cerca tiene decenas de puntos
    y no por eso es una pared. `max_points` (None = sin límite) queda por si
    algún sensor lo necesita.
    """
    return [c for c in clusters_info
            if c["width"] <= max_width and c["dist"] <= max_dist
            and (max_points is None or c["n"] <= max_points)
            and c.get("isolated", True)]


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


def _fourth_leg_possible(xs, ys, x, y, own_legs, tol=0.1):
    """La cuarta pata en (x, y) se ve en el barrido o la tapa otra pata de la
    misma mesa (la pata del fondo queda detrás de una de delante)."""
    if np.any(np.hypot(xs - x, ys - y) < tol):
        return True
    dist, ang = math.hypot(x, y), math.atan2(y, x)
    half = max(math.atan2(tol, dist), math.radians(1.5))
    return any(abs(normalize_angle(leg["angle"] - ang)) < half + math.atan2(0.05, leg["dist"])
               and leg["dist"] < dist - tol for leg in own_legs)


def _inside_is_empty(xs, ys, end, other, c, margin=0.08):
    """No hay puntos del barrido dentro del rectángulo de patas end-other-c
    (debajo de una mesa no hay nada a la altura del lidar)."""
    ux, uy = other["x"] - end["x"], other["y"] - end["y"]
    vx, vy = c["x"] - end["x"], c["y"] - end["y"]
    lu, lv = math.hypot(ux, uy), math.hypot(vx, vy)
    s = ((xs - end["x"]) * ux + (ys - end["y"]) * uy) / lu
    t = ((xs - end["x"]) * vx + (ys - end["y"]) * vy) / lv
    inside = (s > margin) & (s < lu - margin) & (t > margin) & (t < lv - margin)
    return not np.any(inside)


def _corner_leg(a, b, legs, xs, ys, min_spacing, max_spacing, angle_tol):
    """Tercera pata que forma un ángulo recto con el lado a-b en a o en b, y
    cuya cuarta pata (la que cierra el rectángulo) no contradice el barrido."""
    for end, other in ((a, b), (b, a)):
        ux, uy = other["x"] - end["x"], other["y"] - end["y"]
        for c in legs:
            if c is a or c is b:
                continue
            vx, vy = c["x"] - end["x"], c["y"] - end["y"]
            side = math.hypot(vx, vy)
            if not (min_spacing <= side <= max_spacing):
                continue
            cos = (ux * vx + uy * vy) / (math.hypot(ux, uy) * side)
            if (abs(cos) < math.sin(angle_tol)
                    and _fourth_leg_possible(xs, ys, other["x"] + vx, other["y"] + vy, (a, b, c))
                    and _inside_is_empty(xs, ys, end, other, c)):
                return c
    return None


def find_tables(legs, xs, ys, min_spacing=0.45, max_spacing=1.6, angle_tol=math.radians(15)):
    """Lados de mesa por donde puede pasar el robot.

    Un lado es una pareja de patas a-b con hueco libre entre ellas y una tercera
    pata c en ángulo recto (una esquina), con los dos lados entre `min_spacing`
    y `max_spacing`; la cuarta pata tiene que verse o quedar tapada por otra
    pata de la misma mesa, y dentro del rectángulo no puede haber nada. Así no cuentan las sillas (lados más
    cortos), patas sueltas de varias sillas ni la diagonal de la mesa. La pata
    c indica hacia dónde está el interior de la mesa.

    Devuelve una lista de dicts {a, b, c, mid_x, mid_y, dist}, de la más
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
            c = _corner_leg(a, b, legs, xs, ys, min_spacing, max_spacing, angle_tol)
            if c is None:
                continue
            mx, my = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
            tables.append({"a": a, "b": b, "c": c, "mid_x": mx, "mid_y": my,
                           "dist": math.hypot(mx, my)})
    tables.sort(key=lambda t: t["dist"])
    return tables


def find_two_leg_tables(legs, xs, ys, min_spacing=0.45, max_spacing=1.8, min_width=0.05,
                        clearance=0.25, person_gap=0.45, depth=0.02):
    """Mesas de dos patas gordas: pie central con base redonda o patas en T.

    Solo tienen dos columnas (6-12 cm) a lo largo del eje de la mesa y el
    tablero sobresale igual por los dos lados, así que el sitio para esconderse
    es el punto medio entre ellas, entrando perpendicular. Cuentan parejas de
    patas de al menos `min_width` de ancho (las de silla son más finas),
    separadas entre `min_spacing` y `max_spacing`, con el hueco entre ellas
    libre y sin nada a `clearance` del punto medio a ningún lado. Una pata con
    otra pata gorda a menos de `person_gap` se descarta (serían las dos piernas
    de una persona).

    Devuelve el mismo formato que `find_tables`, con una pata c ficticia a
    `depth` m de b hacia el lado contrario al robot (así se entra desde donde
    está) y `two_leg` = True.
    """
    fat = [l for l in legs if l["width"] >= min_width]
    fat = [l for l in fat if not any(
        o is not l and math.hypot(o["x"] - l["x"], o["y"] - l["y"]) < person_gap for o in fat)]
    tables = []
    for i in range(len(fat)):
        for j in range(i + 1, len(fat)):
            a, b = fat[i], fat[j]
            spacing = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if not (min_spacing <= spacing <= max_spacing):
                continue
            if not _segment_is_free(xs, ys, a, b):
                continue
            mx, my = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
            nx, ny = -(b["y"] - a["y"]) / spacing, (b["x"] - a["x"]) / spacing
            if nx * mx + ny * my < 0:     # normal hacia el lado contrario al robot
                nx, ny = -nx, -ny
            if len(xs) and np.min(np.hypot(np.asarray(xs) - mx, np.asarray(ys) - my)) < clearance:
                continue
            c = {"x": b["x"] + depth * nx, "y": b["y"] + depth * ny}
            c.update(dist=math.hypot(c["x"], c["y"]), angle=math.atan2(c["y"], c["x"]))
            tables.append({"a": a, "b": b, "c": c, "mid_x": mx, "mid_y": my,
                           "dist": math.hypot(mx, my), "two_leg": True})
    tables.sort(key=lambda t: t["dist"])
    return tables


def find_people(clusters_info, body_width=(0.15, 0.6), leg_width=(0.06, 0.25),
                max_leg_gap=0.45, max_dist=3.5):
    """Objetos con forma de persona vista a la altura del lidar.

    - Un cuerpo: un grupo de `body_width` metros de ancho (en Gazebo la persona
      es un cilindro de 36 cm).
    - Dos piernas: dos grupos de `leg_width` metros a menos de `max_leg_gap`
      entre sí (persona real con el lidar a la altura de las rodillas). Se
      devuelven como un solo candidato en el punto medio.

    Las patas de sillas y mesas son más finas y las paredes más anchas, así
    que no cuentan. Devuelve dicts con las mismas claves que describe_clusters.
    """
    near = [c for c in clusters_info if c["dist"] <= max_dist]
    people = [c for c in near if body_width[0] <= c["width"] <= body_width[1]]
    legs = [c for c in near if leg_width[0] <= c["width"] <= leg_width[1]]
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            a, b = legs[i], legs[j]
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) <= max_leg_gap:
                x, y = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
                people.append({"x": x, "y": y, "width": a["width"] + b["width"],
                               "n": a["n"] + b["n"], "dist": math.hypot(x, y),
                               "angle": math.atan2(y, x), "isolated": True})
    return people


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


def points_to_scan(xs_odom, ys_odom, pose, resolution=math.radians(0.5), max_range=3.5,
                   priority=None):
    """Barrido virtual de 360º desde `pose` a partir de puntos en odom.

    Sirve para un sensor de campo estrecho (p. ej. el /scan de la cámara de
    profundidad): se acumulan los puntos mientras el robot gira y luego se
    reordenan por ángulo en el marco del robot. En cada sector de `resolution`
    se queda el punto más cercano, como haría un lidar real, o, si se da
    `priority`, el de menor prioridad (p. ej. el visto más centrado en la
    cámara: así cada zona sale de uno o dos scans seguidos y un error de
    odometría durante la vuelta no emborrona las patas). Devuelve
    (angulos, distancias) ordenados de -pi a pi.
    """
    xs_odom, ys_odom = np.asarray(xs_odom, dtype=float), np.asarray(ys_odom, dtype=float)
    if len(xs_odom) == 0:
        return np.array([]), np.array([])
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = xs_odom - px, ys_odom - py
    x, y = c * dx + s * dy, -s * dx + c * dy
    r = np.hypot(x, y)
    keep = (r > 0.01) & (r < max_range)
    ang, r = np.arctan2(y[keep], x[keep]), r[keep]
    n_bins = int(round(2 * math.pi / resolution))
    bins = np.floor((ang + math.pi) / resolution).astype(int) % n_bins
    best = np.full(n_bins, np.inf)
    if priority is None:
        np.minimum.at(best, bins, r)
    else:
        # Al asignar con índices repetidos gana el último: se ordena de peor a mejor
        order = np.argsort(-np.asarray(priority, dtype=float)[keep], kind="stable")
        best[bins[order]] = r[order]
    filled = np.isfinite(best)
    centers = -math.pi + (np.arange(n_bins) + 0.5) * resolution
    return centers[filled], best[filled]


def segment_clearance(xs, ys, p0, p1):
    """Distancia del punto del barrido más cercano al segmento p0-p1."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs) == 0:
        return math.inf
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length2 = dx * dx + dy * dy
    t = np.zeros_like(xs) if length2 == 0 else np.clip(
        ((xs - p0[0]) * dx + (ys - p0[1]) * dy) / length2, 0.0, 1.0)
    return float(np.min(np.hypot(xs - (p0[0] + t * dx), ys - (p0[1] + t * dy))))


def move_with_segment(p, a, b, a2, b2):
    """Lleva el punto p con el movimiento rígido que pasa el segmento a-b a a2-b2."""
    ang = math.atan2(b2[1] - a2[1], b2[0] - a2[0]) - math.atan2(b[1] - a[1], b[0] - a[0])
    c, s = math.cos(ang), math.sin(ang)
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    m2x, m2y = (a2[0] + b2[0]) / 2, (a2[1] + b2[1]) / 2
    dx, dy = p[0] - mx, p[1] - my
    return (m2x + c * dx - s * dy, m2y + s * dx + c * dy)


def match_leg_pair(legs, expected_a, expected_b, spacing, gate, spacing_tol=0.1):
    """Pareja de patas (la, lb) que mejor encaja con las posiciones esperadas.

    Solo cuentan parejas separadas `spacing` ± `spacing_tol` y con cada pata a
    menos de `gate` de donde se esperaba. Al exigir la separación, con un `gate`
    grande no se confunde con otras patas de la misma mesa. Devuelve None si no
    hay ninguna.
    """
    best, best_cost = None, math.inf
    for la in legs:
        da = math.hypot(la["x"] - expected_a[0], la["y"] - expected_a[1])
        if da > gate:
            continue
        for lb in legs:
            if lb is la:
                continue
            db = math.hypot(lb["x"] - expected_b[0], lb["y"] - expected_b[1])
            if db > gate:
                continue
            if abs(math.hypot(la["x"] - lb["x"], la["y"] - lb["y"]) - spacing) > spacing_tol:
                continue
            if da + db < best_cost:
                best, best_cost = (la, lb), da + db
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
