"""
ROS I/O 模組 — 速度限幅 + 各種 Publisher。
不含業務邏輯，只負責「限制 → 發布」。
"""

import rospy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Float32


class VelocityLimiter:
    def __init__(self, max_vx, max_vy, max_wz):
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_wz = max_wz

    def clamp(self, vx, vy, wz):
        vx = max(-self.max_vx, min(vx, self.max_vx))
        vy = max(-self.max_vy, min(vy, self.max_vy))
        wz = max(-self.max_wz, min(wz, self.max_wz))
        return vx, vy, wz


# ═══════════════════════════════════════════════════════════════
# Module 6: CmdVelPublisher（不動）
# ═══════════════════════════════════════════════════════════════

class CmdVelPublisher:
    def __init__(self, all_dog_names):
        self._pubs = {
            name: rospy.Publisher(f"/{name}/cmd_vel", Twist, queue_size=1)
            for name in all_dog_names
        }

    def publish(self, name, vx, vy, wz):
        cmd = Twist()
        cmd.linear.x  = vx
        cmd.linear.y  = vy
        cmd.angular.z = wz
        self._pubs[name].publish(cmd)

    def publish_zero(self, names):
        zero = Twist()
        for name in names:
            self._pubs[name].publish(zero)


class AccelCmdPublisher:
    def __init__(self, all_dog_names, scale=1.0):
        self._pubs = {
            name: rospy.Publisher(f"/{name}/accel_cmd", Vector3, queue_size=1)
            for name in all_dog_names
        }
        # Scales the upper-layer QP acceleration fed to the WBC base-accel task.
        # The published value is the full a_QP* (not a Δa correction); `scale`
        # multiplies that full acceleration. 0.0 disables injection (rely on
        # cmd_vel→MPC only); 1.0 = full a_QP*.
        self._scale = float(scale)

    def publish(self, name, ax, ay):
        msg = Vector3()
        msg.x = float(ax) * self._scale
        msg.y = float(ay) * self._scale
        msg.z = 0.0
        self._pubs[name].publish(msg)

    def publish_zero(self, names):
        zero = Vector3()
        for name in names:
            self._pubs[name].publish(zero)


class CbfDebugPublisher:
    """發布 CBF 監看量到 /cbf_debug/*（Float32），用 rqt_plot 即時看。
    h_min_<kind>: 各類約束最小 h（實際安全距離，<0 = 撞）
    slack_<kind>: ψ2 被違反量（含速度層 _v）
    vcmd_<dog> / vact_<dog>: 命令 vs 實際速度大小（執行落差）
    """

    _KINDS = ("obs", "wall", "pair", "rect")

    def __init__(self, dog_names):
        self._pubs = {}
        for k in self._KINDS:
            self._pubs["h_" + k] = rospy.Publisher(
                "/cbf_debug/h_min_" + k, Float32, queue_size=1)
            self._pubs["s_" + k] = rospy.Publisher(
                "/cbf_debug/slack_" + k, Float32, queue_size=1)
        for name in dog_names:
            self._pubs["vc_" + name] = rospy.Publisher(
                "/cbf_debug/vcmd_" + name, Float32, queue_size=1)
            self._pubs["va_" + name] = rospy.Publisher(
                "/cbf_debug/vact_" + name, Float32, queue_size=1)

    def publish(self, min_h, slack, vcmd, vact):
        for k in self._KINDS:
            self._pubs["h_" + k].publish(Float32(float(min_h.get(k, 9.0))))
            # 取 base 與速度層(_v)的較大 slack
            s = max(float(slack.get(k, 0.0)), float(slack.get(k + "_v", 0.0)))
            self._pubs["s_" + k].publish(Float32(s))
        for name, v in vcmd.items():
            self._pubs["vc_" + name].publish(Float32(float(v)))
        for name, v in vact.items():
            self._pubs["va_" + name].publish(Float32(float(v)))


# ═══════════════════════════════════════════════════════════════
# 主控: FleetManagerUQP
# ═══════════════════════════════════════════════════════════════

