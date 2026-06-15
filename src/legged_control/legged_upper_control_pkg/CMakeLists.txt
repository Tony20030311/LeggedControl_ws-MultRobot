cmake_minimum_required(VERSION 3.0.2)
project(legged_upper_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)

catkin_python_setup()

catkin_package()

# Install Python entry-point scripts
catkin_install_python(PROGRAMS
  scripts/fleet_manager_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install config / launch
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
