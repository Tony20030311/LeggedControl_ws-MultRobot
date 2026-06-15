"""
CBF 控制器共用介面 — 定義一階/二階 controller 的多型 base class。

FleetManager 透過這個介面操作 controller，不需要知道是一階還是二階：
    controller.set_obstacles(...)
    controller.set_walls(...)
    controller.solve(...)
"""

from abc import ABC, abstractmethod


class CBFControllerBase(ABC):
    """CBF QP 控制器的共用介面。"""

    @abstractmethod
    def set_obstacles(self, obstacles):
        """設定圓形障礙物清單。"""

    @abstractmethod
    def set_rect_obstacles(self, rect_obstacles):
        """設定矩形障礙物清單。"""

    @abstractmethod
    def set_walls(self, walls):
        """設定牆壁清單。"""

    @abstractmethod
    def reset_prediction(self, reason=""):
        """重設 multi-step prediction 狀態。"""

    @abstractmethod
    def solve(self, all_dogs, states, u_nominal, **kwargs):
        """
        求解 QP，回傳安全的速度指令。

        Parameters
        ----------
        all_dogs : list[str]
            所有狗的名字
        states : dict[str, RobotState]
            每隻狗的狀態
        u_nominal : dict[str, tuple]
            每隻狗的 nominal velocity (vx, vy) in body frame
        **kwargs
            子類特有參數：
            - 一階: formation_grad, cbf_enabled, dt
            - 二階: a_desired, formation_grad, cbf_enabled, dt, yaw_rates

        Returns
        -------
        dict[str, tuple]
            每隻狗的安全速度 (vx, vy) in body frame
        """
