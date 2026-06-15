"""
幾何工具函式 — 純數學運算，不依賴 ROS。
所有模組共用的 L0 基礎層。
"""

import math
import numpy as np


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rot2d(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]])


def closest_point_on_aabb(point, center, size):
    half = 0.5 * np.array(size[:2], dtype=float)
    center = np.array(center[:2], dtype=float)
    return np.minimum(np.maximum(point, center - half), center + half)
