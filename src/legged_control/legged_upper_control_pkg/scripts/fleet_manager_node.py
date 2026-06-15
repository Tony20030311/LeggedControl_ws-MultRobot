#!/usr/bin/env python3
"""
Entry point for the fleet manager node.

Usage:
    rosrun legged_upper_control fleet_manager_node.py
"""
from legged_upper_control.fleet.manager import FleetManagerUQP
import rospy

if __name__ == '__main__':
    try:
        manager = FleetManagerUQP()
        manager.spin()
    except rospy.ROSInterruptException:
        pass
