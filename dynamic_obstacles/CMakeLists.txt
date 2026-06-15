cmake_minimum_required(VERSION 3.0.2)

project(dynamic_obstacles)

find_package(catkin REQUIRED COMPONENTS
  rospy
  gazebo_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/move_cylinder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)