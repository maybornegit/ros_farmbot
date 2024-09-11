import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/frc-ag-2/ros2_ws/src/ros_farmbot/install/ros_farmbot'
