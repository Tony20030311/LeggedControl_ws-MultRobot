#!/usr/bin/env python3
"""
RViz helper for Formation_manager_unified_qp.py.

Inputs:
  /dogN/ground_truth/state        nav_msgs/Odometry
  /formation/projected_goals       geometry_msgs/PoseArray
  /formation/current_formation     std_msgs/String
  /formation/dogN_astar_path       nav_msgs/Path
  /formation/prediction_markers    visualization_msgs/MarkerArray

Outputs:
  /formation/debug_markers         visualization_msgs/MarkerArray

This node is visualization-only. It never writes cmd_vel or controller state.
"""

import os
import math
import threading
import xml.etree.ElementTree as ET

import rospy
import yaml
from geometry_msgs.msg import Point, PoseArray
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Cbf_params_uqp.yaml"
)
DEFAULT_WORLD_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "legged_gazebo", "worlds", "obstacle_world.world",
))


def load_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class FormationDebugVisualizer:
    def __init__(self):
        rospy.init_node("formation_debug_visualizer")
        self.config_path = rospy.get_param("~config_path", CONFIG_PATH)
        self.cfg = load_config(self.config_path)
        self.frame_id = rospy.get_param(
            "~debug_frame_id", self.cfg.get("debug_frame_id", "map"))
        self.rate_hz = float(rospy.get_param("~rate", 5.0))
        self.world_path = rospy.get_param(
            "~gazebo_world_path",
            self.cfg.get("debug_gazebo_world_path", DEFAULT_WORLD_PATH))
        self.draw_goal_formation_edges = bool(rospy.get_param(
            "~debug_draw_goal_formation_edges",
            self.cfg.get("debug_draw_goal_formation_edges", False)))
        self._world_geometry = self._load_world_geometry(self.world_path)
        self.leader_name = rospy.get_param(
            "~leader_name", self.cfg.get("leader_name", "dog1"))
        self.follower_names = list(rospy.get_param(
            "~follower_names", self.cfg.get("follower_names", ["dog2", "dog3"])))
        self.all_dogs = [self.leader_name] + list(self.follower_names)

        self._lock = threading.Lock()
        self._projected_goals = PoseArray()
        self._current_formation = ""
        self._dog_positions = {}
        self._dog_paths = {name: Path() for name in self.all_dogs}
        self._prediction_markers = []

        rospy.Subscriber(
            "/formation/projected_goals",
            PoseArray,
            self._goals_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            "/formation/current_formation",
            String,
            self._formation_cb,
            queue_size=1,
        )
        for name in self.all_dogs:
            rospy.Subscriber(
                "/%s/ground_truth/state" % name,
                Odometry,
                self._odom_cb,
                callback_args=name,
                queue_size=1,
            )
            rospy.Subscriber(
                "/formation/%s_astar_path" % name,
                Path,
                self._path_cb,
                callback_args=name,
                queue_size=1,
            )
        rospy.Subscriber(
            "/formation/prediction_markers",
            MarkerArray,
            self._prediction_cb,
            queue_size=1,
        )
        self._pub = rospy.Publisher(
            "/formation/debug_markers", MarkerArray, queue_size=1, latch=True)
        rospy.loginfo(
            "[FormationDebugVisualizer] ready, config=%s, gazebo_world=%s, geometry=%d",
            self.config_path,
            self.world_path,
            len(self._world_geometry),
        )

    def _goals_cb(self, msg):
        with self._lock:
            self._projected_goals = msg

    def _formation_cb(self, msg):
        with self._lock:
            self._current_formation = msg.data

    def _odom_cb(self, msg, name):
        pose = msg.pose.pose.position
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        with self._lock:
            self._dog_positions[name] = (pose.x, pose.y, yaw)

    def _path_cb(self, msg, name):
        with self._lock:
            self._dog_paths[name] = msg

    def _prediction_cb(self, msg):
        with self._lock:
            self._prediction_markers = [
                marker for marker in msg.markers
                if marker.action != Marker.DELETEALL
            ]

    def _base_marker(self, ns, marker_id, marker_type):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _delete_all_marker(self):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.DELETEALL
        return marker

    @staticmethod
    def _point(x, y, z):
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

    @staticmethod
    def _dog_color(idx):
        colors = [
            (1.0, 0.08, 0.05),
            (0.0, 0.85, 0.25),
            (0.1, 0.45, 1.0),
        ]
        return colors[idx % len(colors)]

    @staticmethod
    def _parse_float_list(text, default=None):
        if text is None:
            return list(default or [])
        try:
            return [float(value) for value in text.split()]
        except (TypeError, ValueError):
            return list(default or [])

    @classmethod
    def _parse_pose(cls, text):
        values = cls._parse_float_list(text, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        values = (values + [0.0] * 6)[:6]
        return values

    @staticmethod
    def _compose_planar_pose(parent, child):
        yaw = parent[5]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return [
            parent[0] + cos_yaw * child[0] - sin_yaw * child[1],
            parent[1] + sin_yaw * child[0] + cos_yaw * child[1],
            parent[2] + child[2],
            parent[3] + child[3],
            parent[4] + child[4],
            parent[5] + child[5],
        ]

    @staticmethod
    def _set_yaw(marker, yaw):
        marker.pose.orientation.z = math.sin(0.5 * yaw)
        marker.pose.orientation.w = math.cos(0.5 * yaw)

    @classmethod
    def _sdf_material_rgba(cls, visual):
        material = visual.find("material")
        if material is None:
            return None
        rgba = cls._parse_float_list(
            material.findtext("diffuse"),
            cls._parse_float_list(material.findtext("ambient"), []),
        )
        if len(rgba) < 3:
            return None
        rgba = (rgba + [1.0])[:4]
        return rgba

    def _load_world_geometry(self, path):
        if not path or not os.path.exists(path):
            rospy.logwarn(
                "[FormationDebugVisualizer] Gazebo world not found: %s", path)
            return []

        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            rospy.logwarn(
                "[FormationDebugVisualizer] failed to parse Gazebo world %s: %s",
                path,
                exc,
            )
            return []

        geometry = []
        for model_idx, model in enumerate(root.findall(".//model")):
            model_name = model.get("name", "model_%d" % model_idx)
            model_pose = self._parse_pose(model.findtext("pose"))
            for link_idx, link in enumerate(model.findall("link")):
                link_pose = self._compose_planar_pose(
                    model_pose, self._parse_pose(link.findtext("pose")))
                elements = link.findall("visual")
                if not elements:
                    elements = link.findall("collision")
                for visual_idx, visual in enumerate(elements):
                    geom = visual.find("geometry")
                    if geom is None:
                        continue
                    pose = self._compose_planar_pose(
                        link_pose, self._parse_pose(visual.findtext("pose")))
                    rgba = self._sdf_material_rgba(visual)
                    box = geom.find("box")
                    cylinder = geom.find("cylinder")
                    if box is not None:
                        size = self._parse_float_list(box.findtext("size"), [])
                        if len(size) < 3:
                            continue
                        geometry.append({
                            "type": "box",
                            "name": model_name,
                            "id": "%s_%d_%d" % (
                                model_name, link_idx, visual_idx),
                            "pose": pose,
                            "size": size[:3],
                            "rgba": rgba,
                        })
                    elif cylinder is not None:
                        radius = self._parse_float_list(
                            cylinder.findtext("radius"), [])
                        length = self._parse_float_list(
                            cylinder.findtext("length"), [])
                        if not radius or not length:
                            continue
                        geometry.append({
                            "type": "cylinder",
                            "name": model_name,
                            "id": "%s_%d_%d" % (
                                model_name, link_idx, visual_idx),
                            "pose": pose,
                            "radius": radius[0],
                            "length": length[0],
                            "rgba": rgba,
                        })
        return geometry

    def _map_floor_marker(self):
        x_min = float(self.cfg.get("map_x_min", 0.0))
        x_max = float(self.cfg.get("map_x_max", 10.0))
        y_min = float(self.cfg.get("map_y_min", -5.0))
        y_max = float(self.cfg.get("map_y_max", 5.0))
        marker = self._base_marker("gazebo_map_floor", 0, Marker.CUBE)
        marker.pose.position.x = 0.5 * (x_min + x_max)
        marker.pose.position.y = 0.5 * (y_min + y_max)
        marker.pose.position.z = -0.03
        marker.scale.x = max(0.01, x_max - x_min)
        marker.scale.y = max(0.01, y_max - y_min)
        marker.scale.z = 0.01
        marker.color.r = 0.58
        marker.color.g = 0.58
        marker.color.b = 0.58
        marker.color.a = 0.35
        return marker

    @staticmethod
    def _world_marker_color(item):
        if item.get("rgba") is not None:
            rgba = item["rgba"]
            return (rgba[0], rgba[1], rgba[2], min(0.95, rgba[3]))
        name = item.get("name", "").lower()
        if "wall" in name:
            return (0.50, 0.30, 0.12, 0.88)
        return (0.20, 0.20, 0.20, 0.88)

    def _gazebo_world_markers(self):
        markers = [self._map_floor_marker()]
        for idx, item in enumerate(self._world_geometry):
            marker_type = Marker.CUBE if item["type"] == "box" else Marker.CYLINDER
            ns = "gazebo_map_walls" if "wall" in item["name"] else "gazebo_map_objects"
            marker = self._base_marker(ns, idx, marker_type)
            pose = item["pose"]
            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            marker.pose.position.z = pose[2]
            self._set_yaw(marker, pose[5])

            if item["type"] == "box":
                marker.scale.x = item["size"][0]
                marker.scale.y = item["size"][1]
                marker.scale.z = item["size"][2]
            else:
                marker.scale.x = 2.0 * item["radius"]
                marker.scale.y = 2.0 * item["radius"]
                marker.scale.z = item["length"]

            r, g, b, a = self._world_marker_color(item)
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = a
            markers.append(marker)
        return markers

    def _physical_map_markers(self):
        if self._world_geometry:
            return self._gazebo_world_markers()
        markers = [self._map_floor_marker()]
        markers.extend(self._outer_wall_markers())
        markers.extend(self._rect_obstacle_markers())
        markers.extend(self._circular_obstacle_markers())
        return markers

    def _astar_planning_margin(self, formation):
        if formation == "line":
            return float(self.cfg.get(
                "astar_formation_margin_line",
                self.cfg.get("astar_robot_radius", 0.15)))
        return float(self.cfg.get(
            "astar_formation_margin_v",
            self.cfg.get("astar_robot_radius", 0.15)))

    def _rectangle_outline_marker(self, ns, marker_id, center, size,
                                  z, rgba, width=0.035):
        marker = self._base_marker(ns, marker_id, Marker.LINE_LIST)
        marker.scale.x = float(width)
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]

        cx, cy = float(center[0]), float(center[1])
        hx, hy = 0.5 * float(size[0]), 0.5 * float(size[1])
        corners = [
            (cx - hx, cy - hy),
            (cx + hx, cy - hy),
            (cx + hx, cy + hy),
            (cx - hx, cy + hy),
        ]
        for idx in range(4):
            start = corners[idx]
            end = corners[(idx + 1) % 4]
            marker.points.append(self._point(start[0], start[1], z))
            marker.points.append(self._point(end[0], end[1], z))
        return marker

    def _circle_outline_marker(self, ns, marker_id, center, radius,
                               z, rgba, width=0.035, segments=72):
        marker = self._base_marker(ns, marker_id, Marker.LINE_STRIP)
        marker.scale.x = float(width)
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]

        cx, cy = float(center[0]), float(center[1])
        radius = max(0.0, float(radius))
        for idx in range(segments + 1):
            theta = 2.0 * math.pi * float(idx) / float(segments)
            marker.points.append(self._point(
                cx + radius * math.cos(theta),
                cy + radius * math.sin(theta),
                z,
            ))
        return marker

    def _rect_obstacle_markers(self):
        markers = []
        for idx, rect in enumerate(self.cfg.get("rect_obstacles", [])):
            center = rect.get("center", [0.0, 0.0])
            size = rect.get("size", [0.0, 0.0])
            marker = self._base_marker("rect_obstacles", idx, Marker.CUBE)
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = 0.035
            marker.scale.x = float(size[0])
            marker.scale.y = float(size[1])
            marker.scale.z = 0.10
            marker.color.r = 0.02
            marker.color.g = 0.02
            marker.color.b = 0.02
            marker.color.a = 0.70
            markers.append(marker)
        return markers

    def _circular_obstacle_markers(self):
        markers = []
        default_physical_radius = float(
            self.cfg.get("debug_obstacle_physical_radius", 0.20))
        for idx, obs in enumerate(self.cfg.get("obstacles", [])):
            pos = obs.get("pos", [0.0, 0.0])
            radius = float(obs.get("physical_radius", default_physical_radius))

            marker = self._base_marker("circular_obstacles", idx, Marker.CYLINDER)
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = 0.06
            marker.scale.x = 2.0 * radius
            marker.scale.y = 2.0 * radius
            marker.scale.z = 0.12
            marker.color.r = 0.02
            marker.color.g = 0.02
            marker.color.b = 0.02
            marker.color.a = 0.72
            markers.append(marker)

            label = self._base_marker("circular_obstacle_labels", idx, Marker.TEXT_VIEW_FACING)
            label.pose.position.x = float(pos[0])
            label.pose.position.y = float(pos[1])
            label.pose.position.z = 0.38
            label.scale.z = 0.16
            label.color.r = 0.05
            label.color.g = 0.05
            label.color.b = 0.05
            label.color.a = 0.9
            label.text = "obs%d" % (idx + 1)
            markers.append(label)
        return markers

    def _outer_wall_segments(self, right_offset=0.0, top_offset=None, bottom_offset=None):
        x_min = float(self.cfg.get("map_x_min", 0.0))
        x_max = float(self.cfg.get("map_x_max", 10.0))
        y_min = float(self.cfg.get("map_y_min", -5.0))
        y_max = float(self.cfg.get("map_y_max", 5.0))
        door_x = float(self.cfg.get(
            "debug_outer_wall_x_min", self.cfg.get("door_x", x_min)))
        door_x = min(max(door_x, x_min), x_max)

        right_offset = max(0.0, float(right_offset))
        top_offset = right_offset if top_offset is None else max(0.0, float(top_offset))
        bottom_offset = right_offset if bottom_offset is None else max(0.0, float(bottom_offset))

        right_x = max(door_x, x_max - right_offset)
        top_y = y_max - top_offset
        bottom_y = y_min + bottom_offset

        # Match Gazebo's C-shaped outer wall: right wall is full height,
        # top/bottom walls only exist from the doorway x to the right wall.
        return [
            ((right_x, bottom_y), (right_x, top_y)),
            ((door_x, top_y), (right_x, top_y)),
            ((door_x, bottom_y), (right_x, bottom_y)),
        ]

    def _outer_wall_markers(self):
        marker = self._base_marker("outer_walls", 0, Marker.LINE_LIST)
        marker.scale.x = 0.10
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.90
        for start, end in self._outer_wall_segments():
            marker.points.append(self._point(start[0], start[1], 0.08))
            marker.points.append(self._point(end[0], end[1], 0.08))
        return [marker]

    def _forbidden_zone_markers(self):
        markers = []
        if not bool(self.cfg.get("forbidden_zones_enabled", True)):
            return markers
        for idx, zone in enumerate(self.cfg.get("astar_forbidden_zones", [])):
            center = zone.get("center", [0.0, 0.0])
            size = zone.get("size", [0.0, 0.0])
            marker = self._base_marker("forbidden_zones", idx, Marker.CUBE)
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = 0.025
            marker.scale.x = float(size[0])
            marker.scale.y = float(size[1])
            marker.scale.z = 0.05
            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.05
            marker.color.a = 0.28
            markers.append(marker)

            label = self._base_marker("forbidden_zone_labels", idx, Marker.TEXT_VIEW_FACING)
            label.pose.position.x = float(center[0])
            label.pose.position.y = float(center[1])
            label.pose.position.z = 0.25
            label.scale.z = 0.18
            label.color.r = 1.0
            label.color.g = 0.25
            label.color.b = 0.1
            label.color.a = 0.9
            label.text = zone.get("name", "forbidden")
            markers.append(label)
        return markers

    def _astar_inflation_markers(self, formation):
        markers = []
        inflate = self._astar_planning_margin(formation)
        wall_inflate = max(
            float(self.cfg.get("astar_boundary_margin", 0.45)),
            inflate,
        )

        for idx, obs in enumerate(self.cfg.get("obstacles", [])):
            pos = obs.get("pos", [0.0, 0.0])
            radius = float(obs.get("astar_radius", obs.get("radius", 0.0))) + inflate
            markers.append(self._circle_outline_marker(
                "astar_inflation_obstacles",
                idx,
                pos,
                radius,
                0.17,
                (0.1, 0.45, 1.0, 0.95),
                width=0.030,
            ))

        for idx, rect in enumerate(self.cfg.get("rect_obstacles", [])):
            center = rect.get("center", [0.0, 0.0])
            size = rect.get("size", [0.0, 0.0])
            margin = max(float(rect.get("astar_margin", inflate)), inflate)
            inflated_size = [
                float(size[0]) + 2.0 * margin,
                float(size[1]) + 2.0 * margin,
            ]
            markers.append(self._rectangle_outline_marker(
                "astar_inflation_rects",
                idx,
                center,
                inflated_size,
                0.17,
                (0.1, 0.45, 1.0, 0.95),
                width=0.030,
            ))

        wall_marker = self._base_marker("astar_inflation_walls", 0, Marker.LINE_LIST)
        wall_marker.scale.x = 0.045
        wall_marker.color.r = 0.1
        wall_marker.color.g = 0.45
        wall_marker.color.b = 1.0
        wall_marker.color.a = 0.95
        for start, end in self._outer_wall_segments(wall_inflate):
            wall_marker.points.append(self._point(start[0], start[1], 0.17))
            wall_marker.points.append(self._point(end[0], end[1], 0.17))
        markers.append(wall_marker)
        return markers

    def _cbf_boundary_markers(self):
        markers = []

        for idx, obs in enumerate(self.cfg.get("obstacles", [])):
            pos = obs.get("pos", [0.0, 0.0])
            radius = float(obs.get("radius", 0.0)) + float(obs.get("cbf_d_safe", 0.0))
            markers.append(self._circle_outline_marker(
                "cbf_boundaries_obstacles",
                idx,
                pos,
                radius,
                0.21,
                (1.0, 0.55, 0.05, 0.98),
                width=0.040,
            ))

        for idx, rect in enumerate(self.cfg.get("rect_obstacles", [])):
            center = rect.get("center", [0.0, 0.0])
            size = rect.get("size", [0.0, 0.0])
            d_safe = float(rect.get("d_safe", 0.35))
            safe_size = [
                float(size[0]) + 2.0 * d_safe,
                float(size[1]) + 2.0 * d_safe,
            ]
            markers.append(self._rectangle_outline_marker(
                "cbf_boundaries_rects",
                idx,
                center,
                safe_size,
                0.21,
                (1.0, 0.55, 0.05, 0.98),
                width=0.040,
            ))

        wall_marker = self._base_marker("cbf_boundaries_walls", 0, Marker.LINE_LIST)
        wall_marker.scale.x = 0.055
        wall_marker.color.r = 1.0
        wall_marker.color.g = 0.55
        wall_marker.color.b = 0.05
        wall_marker.color.a = 0.98
        right_offset = 0.4
        top_offset = 0.4
        bottom_offset = 0.4
        for wall in self.cfg.get("walls", []):
            normal = wall.get("normal", [0.0, 0.0])
            d_safe = float(wall.get("d_safe", 0.4))
            nx = float(normal[0])
            ny = float(normal[1])
            if abs(nx) > abs(ny) and nx < 0.0:
                right_offset = d_safe
            elif abs(ny) >= abs(nx) and ny < 0.0:
                top_offset = d_safe
            elif abs(ny) >= abs(nx) and ny > 0.0:
                bottom_offset = d_safe

        for start, end in self._outer_wall_segments(
                right_offset, top_offset, bottom_offset):
            wall_marker.points.append(self._point(start[0], start[1], 0.21))
            wall_marker.points.append(self._point(end[0], end[1], 0.21))
        markers.append(wall_marker)
        return markers

    def _dog_footprint_markers(self, dog_positions):
        markers = []
        length = 2.0 * float(self.cfg.get("robot_footprint_half_length", 0.35))
        width = 2.0 * float(self.cfg.get("robot_footprint_half_width", 0.20))
        drift = float(self.cfg.get("robot_footprint_drift_margin", 0.08))
        for idx, name in enumerate(self.all_dogs):
            state = dog_positions.get(name)
            if state is None:
                continue
            x, y, yaw = (list(state) + [0.0])[:3]

            footprint = self._base_marker("dog_footprints", idx, Marker.CUBE)
            footprint.pose.position.x = float(x)
            footprint.pose.position.y = float(y)
            footprint.pose.position.z = 0.03
            self._set_yaw(footprint, yaw)
            footprint.scale.x = max(0.01, length)
            footprint.scale.y = max(0.01, width)
            footprint.scale.z = 0.025
            footprint.color.r = 0.0
            footprint.color.g = 1.0
            footprint.color.b = 0.1
            footprint.color.a = 0.22
            markers.append(footprint)

            drift_margin = self._base_marker(
                "dog_footprint_drift_margins", idx, Marker.CUBE)
            drift_margin.pose = footprint.pose
            drift_margin.scale.x = max(0.01, length + 2.0 * drift)
            drift_margin.scale.y = max(0.01, width + 2.0 * drift)
            drift_margin.scale.z = 0.015
            drift_margin.color.r = 0.0
            drift_margin.color.g = 0.9
            drift_margin.color.b = 1.0
            drift_margin.color.a = 0.12
            markers.append(drift_margin)
        return markers

    def _current_formation_markers(self, dog_positions):
        markers = []
        if not all(name in dog_positions for name in self.all_dogs):
            return markers

        points = []
        for idx, name in enumerate(self.all_dogs):
            x, y, _ = (list(dog_positions[name]) + [0.0])[:3]
            points.append(self._point(x, y, 0.12))

            node = self._base_marker("current_formation_nodes", idx, Marker.SPHERE)
            node.pose.position.x = float(x)
            node.pose.position.y = float(y)
            node.pose.position.z = 0.12
            node.scale.x = 0.16
            node.scale.y = 0.16
            node.scale.z = 0.16
            node.color.r = 0.0
            node.color.g = 1.0
            node.color.b = 0.1
            node.color.a = 0.95
            markers.append(node)

            label = self._base_marker("current_formation_labels", idx, Marker.TEXT_VIEW_FACING)
            label.pose.position.x = float(x)
            label.pose.position.y = float(y)
            label.pose.position.z = 0.38
            label.scale.z = 0.17
            label.color.r = 0.0
            label.color.g = 1.0
            label.color.b = 0.1
            label.color.a = 0.9
            label.text = name
            markers.append(label)

        edges = self._base_marker("current_formation_edges", 0, Marker.LINE_LIST)
        edges.scale.x = 0.045
        edges.color.r = 1.0
        edges.color.g = 1.0
        edges.color.b = 0.0
        edges.color.a = 0.95
        for ia, ib in ((0, 1), (1, 2), (2, 0)):
            edges.points.append(points[ia])
            edges.points.append(points[ib])
        markers.append(edges)
        return markers

    def _goal_formation_edge_markers(self, goals):
        markers = []
        if not self.draw_goal_formation_edges:
            return markers
        if len(goals.poses) < 2:
            return markers

        points = [
            self._point(pose.position.x, pose.position.y, 0.18)
            for pose in goals.poses
        ]
        edges = self._base_marker("goal_formation_edges", 0, Marker.LINE_LIST)
        edges.scale.x = 0.035
        edges.color.r = 1.0
        edges.color.g = 0.78
        edges.color.b = 0.05
        edges.color.a = 0.85
        for idx in range(len(points)):
            edges.points.append(points[idx])
            edges.points.append(points[(idx + 1) % len(points)])
        markers.append(edges)
        return markers

    def _astar_path_markers(self, dog_paths):
        markers = []
        for idx, name in enumerate(self.all_dogs):
            path = dog_paths.get(name)
            if path is None or len(path.poses) < 2:
                continue
            r, g, b = self._dog_color(idx)
            marker = self._base_marker("astar_paths", idx, Marker.LINE_STRIP)
            marker.scale.x = 0.035
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.9
            for pose_stamped in path.poses:
                p = pose_stamped.pose.position
                marker.points.append(self._point(p.x, p.y, 0.055))
            markers.append(marker)
        return markers

    def _projected_goal_markers(self, goals):
        markers = []
        colors = [
            (0.1, 0.45, 1.0),
            (0.1, 0.85, 0.25),
            (1.0, 0.8, 0.1),
        ]
        for idx, pose in enumerate(goals.poses):
            r, g, b = colors[idx % len(colors)]
            marker = self._base_marker("projected_goals", idx, Marker.SPHERE)
            marker.pose = pose
            marker.pose.position.z = 0.12
            marker.scale.x = 0.18
            marker.scale.y = 0.18
            marker.scale.z = 0.18
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.95
            markers.append(marker)

            label = self._base_marker("projected_goal_labels", idx, Marker.TEXT_VIEW_FACING)
            label.pose.position.x = pose.position.x
            label.pose.position.y = pose.position.y
            label.pose.position.z = 0.36
            label.scale.z = 0.18
            label.color.r = r
            label.color.g = g
            label.color.b = b
            label.color.a = 0.95
            label.text = "goal%d" % (idx + 1)
            markers.append(label)
        return markers

    def _formation_marker(self, formation):
        marker = self._base_marker("formation_status", 0, Marker.TEXT_VIEW_FACING)
        marker.pose.position.x = float(rospy.get_param("~status_x", 1.0))
        marker.pose.position.y = float(rospy.get_param("~status_y", 4.5))
        marker.pose.position.z = 0.5
        marker.scale.z = 0.28
        marker.color.r = 0.05
        marker.color.g = 0.9
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = "formation: %s" % (formation or "unknown")
        return marker

    def _safety_legend_marker(self):
        marker = self._base_marker("safety_layer_legend", 0, Marker.TEXT_VIEW_FACING)
        marker.pose.position.x = float(rospy.get_param("~legend_x", 1.0))
        marker.pose.position.y = float(rospy.get_param("~legend_y", 4.1))
        marker.pose.position.z = 0.5
        marker.scale.z = 0.20
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = "black: physical | blue: A* inflation | orange: CBF boundary"
        return marker

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self._lock:
                goals = self._projected_goals
                formation = self._current_formation
                dog_positions = dict(self._dog_positions)
                dog_paths = dict(self._dog_paths)
                prediction_markers = list(self._prediction_markers)

            msg = MarkerArray()
            msg.markers.append(self._delete_all_marker())
            msg.markers.extend(self._physical_map_markers())
            msg.markers.extend(self._astar_inflation_markers(formation))
            msg.markers.extend(self._cbf_boundary_markers())
            msg.markers.extend(self._forbidden_zone_markers())
            msg.markers.extend(self._astar_path_markers(dog_paths))
            msg.markers.extend(self._dog_footprint_markers(dog_positions))
            msg.markers.extend(self._current_formation_markers(dog_positions))
            msg.markers.extend(self._projected_goal_markers(goals))
            msg.markers.extend(self._goal_formation_edge_markers(goals))
            msg.markers.extend(prediction_markers)
            msg.markers.append(self._formation_marker(formation))
            msg.markers.append(self._safety_legend_marker())
            self._pub.publish(msg)
            rate.sleep()


if __name__ == "__main__":
    try:
        FormationDebugVisualizer().spin()
    except rospy.ROSInterruptException:
        pass
