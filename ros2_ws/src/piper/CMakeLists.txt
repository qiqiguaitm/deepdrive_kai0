cmake_minimum_required(VERSION 3.8)
project(piper)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/piper_start_ms_node.py
  scripts/piper_start_ms_node_new.py
  scripts/piper_start_slave_node.py
  scripts/piper_start_master_node.py
  scripts/piper_start_master_node_debug.py
  scripts/piper_read_master_node.py
  scripts/policy_inference_node.py
  scripts/rerun_viz_node.py
  scripts/multi_camera_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
