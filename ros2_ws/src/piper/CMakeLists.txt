cmake_minimum_required(VERSION 3.8)
project(piper)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/arm_reader_node.py
  scripts/arm_teleop_node.py
  scripts/master_handle_node.py
  scripts/policy_inference_node.py
  scripts/rerun_viz_node.py
  scripts/multi_camera_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
