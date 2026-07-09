cmake_minimum_required(VERSION 3.15)
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

# ── ADMM C++ core (pybind11 module; needs OSQP v0.6.3 + pybind11 —
#    run tools/setup_admm_cpp_env.sh inside the SIL container once) ──
find_package(pybind11 QUIET CONFIG)
find_package(osqp QUIET CONFIG)
find_package(Eigen3 QUIET)
if(pybind11_FOUND AND osqp_FOUND AND Eigen3_FOUND)
  # core algorithm library (headers in include/, implementations in src/admm_core/;
  # reused later by the roscpp node)
  add_library(admm_core STATIC
    src/admm_core/constants.cpp
    src/admm_core/qp_common.cpp
    src/admm_core/rti.cpp
    src/admm_core/node_qp.cpp
    src/admm_core/edge_qp.cpp
    src/admm_core/formation.cpp
    src/admm_core/motion_adapter.cpp
    src/admm_core/coordinator.cpp
  )
  target_compile_features(admm_core PUBLIC cxx_std_17)
  target_include_directories(admm_core PUBLIC include)
  target_link_libraries(admm_core PUBLIC osqp::osqp Eigen3::Eigen)
  set_target_properties(admm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

  pybind11_add_module(admm_core_cpp
    src/admm_core/bindings.cpp
  )
  target_link_libraries(admm_core_cpp PRIVATE admm_core)
  set_target_properties(admm_core_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION})
  install(TARGETS admm_core_cpp
    LIBRARY DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION})
else()
  message(WARNING "legged_upper_control: pybind11/OSQP/Eigen3 not found — "
                  "skipping admm_core_cpp (run tools/setup_admm_cpp_env.sh in the container)")
endif()

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
