#!/bin/bash
# Activate all four USB-CAN interfaces for dual master + dual slave.
# 由 calibrate_can_mapping.py 自动生成, 2026-05-08 17:21
#
# sim01 bus-info 映射:
#   3-2.2.2:1.0 → can_left_mas (左 master (示教左臂))
#   3-2.2.1:1.0 → can_left_slave (左 slave (执行左臂))
#   3-2.2.3:1.0 → can_right_mas (右 master (示教右臂))
#   3-2.2.4:1.0 → can_right_slave (右 slave (执行右臂))

bash ./can_activate.sh can_left_mas         1000000 "3-2.2.2:1.0"
bash ./can_activate.sh can_left_slave       1000000 "3-2.2.1:1.0"
bash ./can_activate.sh can_right_mas        1000000 "3-2.2.3:1.0"
bash ./can_activate.sh can_right_slave      1000000 "3-2.2.4:1.0"
