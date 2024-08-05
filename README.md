# ros_farmbot - ROS Package for Farmbot Automation

## 1. Overview

### 1.1 Description
This is the foundation for the ROS package used to operate and automate the I2GROW-Oasis prototype. After installing this package into your ROS environment with the requisite equipment and some other localization requirements, a FarmBot should be capable of autonomously photographing specific coordinates, as well as providing a user interface for live updates.

### 1.2 Hardware Utilized
- Webcam: Brio 100
- FarmBot: FarmBot Genesis 1.6
- Depth Camera: Intel Realsense D405 (any D4xx series will likely work)

### 1.3 Software Utilized

- System OS: Ubuntu 22.04
- ROS Distribution: Humble Hawksbill
- Python - 3.10.12
- Python Packages - see requirements.txt

## 2. 🛠️ Installation

1. Ensure that the equipment above is utilized or some parts of the package may throw errors. (If you are not using Ubuntu 22.04, any ROS2 distribution should work, although they are untested.)

2. For the installation of software requirements - 
- First, install the necessary ROS2 distribution (examples given will be based on Humble distribution, but changes will be noted) and build a ROS workspace for necessary packages.
- [ROS2 Installation Instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- [Setting Up ROS2 Workspace](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html)
- Clone this repository inside the src file in the ROS2 workspace. The preferred method of doing this is making an empty directory inside src directory named 'ros_farmbot' and running the git clone command after git initialization.

```
cd ~/ros2_ws/src/ros_farmbot
git init
git clone https://github.com/maybornegit/ros_farmbot.git
```

- Next, deactivating any conda or virtual environment, ensure necessary PyPI packages are installed globally. For this, run:
```
pip install -r /path/to/requirements.txt
```
- Test that the package was download successfully by running 'colcon build' and ensuring no stderrors. Run:
```
cd ~/ros2_ws
colcon build
```

- Lastly, try sourcing the bash files and running a running a ROS node to initialize the '~/ros_farmbot_data' directory in your local home (disconnect from equipment and expect errors). Please copy the 'placeholders' folder in this package into that new directory. Run the initial node by:
```
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ros_farmbot init
```

## 3 Executing the Package

There are two options given for running the package (also option 1 has only been tried at this time): two terminals with separate ROS nodes or a combined launch file.

### 3.1 Separate Terminals (tested)

In your first terminal, run:
```
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ros_farmbot init
```
You should see 'ENTERING CONINUOUS CONTROL:' and then the robot should zero its position.


In your second terminal, run:
```
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ros_farmbot ui
```
You should see 'DISPLAY UPDATE []' and then a matplotlib user interface should appear.

### 3.2 Launch File (untested)

In one terminal, run:
```
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
cd ~/ros2_ws/sc/ros_farmbot/launch
ros2 launch basic.launch
```

This should run both nodes at once, although none of the traditional printed logs will be there.

## 4. Documentation ~ Issues/Future Work

- Test Launch File
- ROS2 Communication / ROSBAG Integration
- PAR Measurement and Logging
- Environmental Controls (Temperature)
- Integration of Live Plant Visualization



