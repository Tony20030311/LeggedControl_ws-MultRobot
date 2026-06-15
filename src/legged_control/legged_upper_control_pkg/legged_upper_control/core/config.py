"""
YAML 設定檔讀取。
"""

import os
import yaml


def _resolve_config_path():
    """依序嘗試: rospkg → 相對路徑 fallback。"""
    # 1) 透過 rospkg 找 catkin package 的 config/ 目錄
    try:
        import rospkg
        pkg_path = rospkg.RosPack().get_path("legged_upper_control")
        candidate = os.path.join(pkg_path, "config",
                                 "Cbf_params_twoOrderCBF.yaml")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    # 2) fallback: 舊的相對路徑（放在 scripts/ 裡面跑的情況）
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, os.pardir,
        "Cbf_params_twoOrderCBF.yaml",
    )


_CONFIG_PATH = _resolve_config_path()


def _load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


_CFG = _load_config(_CONFIG_PATH)


def get_config():
    """回傳已載入的 config dict。"""
    return _CFG
