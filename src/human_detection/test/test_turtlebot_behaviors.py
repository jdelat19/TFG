#!/usr/bin/env python3
"""Pruebas de turtlebot_behaviors.py con un robot falso (sin roscore ni Gazebo).

FakeRobot pone a mano el estado que normalmente rellena TurtlebotEmotionNode
(pose, scan, persona) y se comprueba el Twist que devuelve cada comportamiento.
"""
import math
import os
import sys

import numpy as np
import pytest

rospy = pytest.importorskip("rospy")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import scan_utils as su
from turtlebot_behaviors import TurtlebotBehaviors, make_twist

# rospy.get_time() sin init_node: usa la hora del reloj
rospy.rostime.set_rostime_initialized(True)


def circle_points(cx, cy, r, n=40):
    """Puntos de la mitad visible (desde el origen) de un círculo."""
    base = math.atan2(-cy, -cx)
    a = base + np.linspace(-math.pi / 2, math.pi / 2, n)
    return cx + r * np.cos(a), cy + r * np.sin(a)


class FakeRobot(TurtlebotBehaviors):
    def __init__(self, objects=(), person=(1.0, 0.0)):
        # Parámetros con los valores por defecto del nodo
        self.max_v, self.max_w, self.rate_hz = 0.18, 1.2, 10.0
        self.robot_radius, self.safety_distance = 0.18, 0.35
        self.retreat_distance, self.acc_linear, self.acc_angular = 1.5, 0.3, 1.5
        self.turn_slowdown, self.turn_min_angular = math.radians(25), 0.2
        self.turn_tolerance = math.radians(1)
        self.blind_front, self.bump_time, self.bump_pending = 0.0, -1e9, False
        self.last_cmd = make_twist()
        self.pose, self.st = (0.0, 0.0, 0.0), {}
        self.person, self.person_confirmed = person, False
        xs = np.concatenate([o[0] for o in objects]) if objects else np.array([])
        ys = np.concatenate([o[1] for o in objects]) if objects else np.array([])
        clusters = su.describe_clusters(xs, ys, su.cluster_points(xs, ys))
        self.scan = {"angles": np.arctan2(ys, xs), "ranges": np.hypot(xs, ys), "xs": xs,
                     "ys": ys, "clusters": clusters, "people": su.find_people(clusters),
                     "pose": self.pose}


def test_approach_moves_towards_a_person_in_front():
    robot = FakeRobot([circle_points(1.2, 0.0, 0.18)])
    cmd = robot.behave_approach(stop_distance=0.6, speed=0.18)
    assert cmd.linear.x > 0 and abs(cmd.angular.z) < 0.2
    assert robot.person_confirmed


def test_approach_stops_at_the_distance():
    robot = FakeRobot([circle_points(0.55, 0.0, 0.18)], person=(0.55, 0.0))
    assert robot.behave_approach(stop_distance=0.6, speed=0.18).linear.x == 0.0


def test_retreat_backs_away_from_a_close_person():
    robot = FakeRobot([circle_points(0.8, 0.0, 0.18)], person=(0.8, 0.0))
    assert robot.behave_retreat().linear.x < 0


def test_idle_turns_to_face_the_person():
    robot = FakeRobot([circle_points(0.0, 1.0, 0.18)], person=(0.0, 1.0))
    cmd = robot.behave_idle()
    assert cmd.linear.x == 0.0 and cmd.angular.z > 0      # persona a la izquierda


def test_safety_filter_blocks_an_obstacle_ahead():
    wall = (np.full(30, 0.25), np.linspace(-0.4, 0.4, 30))
    robot = FakeRobot([wall])
    assert robot.safety_filter(make_twist(0.15, 0.0)).linear.x == 0.0
    # Girar o retroceder sigue permitido
    assert robot.safety_filter(make_twist(-0.1, 0.5)).linear.x == -0.1


def test_safety_filter_lets_it_go_when_free():
    robot = FakeRobot([(np.full(30, 1.5), np.linspace(-0.4, 0.4, 30))])
    assert robot.safety_filter(make_twist(0.15, 0.0)).linear.x == 0.15


def test_ramp_limits_acceleration_but_not_braking():
    robot = FakeRobot()
    assert robot.ramp(make_twist(0.18, 0.0)).linear.x == pytest.approx(0.03)
    robot.last_cmd = make_twist(0.18, 0.0)
    assert robot.ramp(make_twist(0.0, 0.0)).linear.x == 0.0


def test_spin_turns_until_360_degrees():
    robot = FakeRobot()
    assert robot.behave_spin().angular.z > 0
    for k in range(1, 37):                 # simula la odometría girando de 10 en 10º
        robot.pose = (0.0, 0.0, su.normalize_angle(math.radians(10 * k)))
        cmd = robot.behave_spin()
    assert cmd.angular.z == 0.0 and robot.st["spin_done"]


def test_table_geometry():
    (mx, my), (nx, ny), side, depth = FakeRobot.table_geometry((1, -0.4), (1, 0.4), (1.6, -0.4))
    assert (mx, my) == (1, 0) and (nx, ny) == pytest.approx((1, 0))
    assert side == pytest.approx(0.8) and depth == pytest.approx(0.6)
