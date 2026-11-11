cmake_minimum_required(VERSION 3.0.2)
project(human_detection)

# Paquete solo de Python: no se compila nada, catkin solo genera los
# lanzadores de los nodos y el setup.bash del workspace

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs nav_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Nodos de ROS (scripts/). Los módulos de src/ los importan con sys.path
catkin_install_python(PROGRAMS
  scripts/human_detector_node.py
  scripts/voice_node.py
  scripts/avatar_node.py
  scripts/turtlebot_emotion_node.py
  scripts/mental_map_node.py
  scripts/launch_menu.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY media
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)