cmake_minimum_required(VERSION 3.0.2)
project(human_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/ros_node.py
  scripts/avatar_node.py
  scripts/voice_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY media
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)