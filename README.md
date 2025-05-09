# ros_farmbot - ROS Package for Farmbot Automation

## 1. Overview

### 1.1 Description
This is the foundation for the ROS package used to operate and automate the I2GROW-Oasis prototype. After installing this package into your ROS environment with the requisite equipment and some other localization requirements, the FarmBot should be capable of autonomously photographing specific coordinates - as well as providing a user interface for live updates.

### 1.2 Hardware Utilized
- Webcam: Brio 100
- FarmBot: FarmBot Genesis 1.6
- Depth Camera: Intel Realsense D405 (any D4xx series will likely work)
- PAR Sensor: Apogee SQ-420X USB PAR Sensor

### 1.3 Software Utilized

- System OS: Ubuntu 22.04
- ROS Distribution: Humble Hawksbill
- Python - 3.10.12
- Python Packages - see requirements.txt

## 2. 🛠️ Installation

1. Ensure that the equipment above is utilized, or some parts of the package may throw errors. (If you are not using Ubuntu 22.04, any ROS2 distribution should work, although they are untested.)

2. For the installation of software requirements - 
- First, install the necessary ROS2 distribution (examples given will be based on Humble distribution, but change the 'humble' in the source commands to your chosen distribution) and build a ROS workspace for necessary packages.
- [ROS2 Installation Instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)

- To add ROS packages to your local computer, make a directory and clone these repositories in src using these commands:

```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/maybornegit/ros_farmbot.git
git clone https://github.com/maybornegit/ros_farmbot_msgs.git
```

- Next, deactivate any conda or virtual environment, ensure necessary PyPI packages are installed globally. For this, run:
```
cd ros_farmbot
pip install -r requirements.txt
sudo apt-get install python3-opencv
```
- Test that the packages was downloaded successfully by running these steps and ensuring no stderrors. Run:
```
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

- Lastly, the package requires an external directory, '~/ros_farmbot_data', be initialized in your local home (disconnect from equipment and expect errors). Execute these lines:
```
mkdir -p ~/ros_farmbot_data/ros_bags
mkdir -p ~/ros_farmbot_data/rgbd_frames
cd ~/ros2_ws/src/ros_farmbot
mv placeholders ~/ros_farmbot_data
mv par_sampled_grid.txt ~/ros_farmbot_data
cd ~/ros_farmbot_data
touch config.txt
```

## 3 Setting up the Configuration File

To ensure that the package is able to connect to the right devices and accounts, a configuration file (config.txt) has been created, which will be read by the package to find relevant items. The organization of this configuration file is:

```
# FarmBot Email
# FarmBot Password
# FarmBot Sequence ID
# Webcam Device Path "/dev/v4l/by-id/..."
# Aranet4 MAC Address (Environmental Readings Sensor)
# CNN Trained Weights File
# Lighting Grid TXT file location
```

The subsections below will define how to find and add these to the config.txt file in '~/ros_farmbot_data'

### 3.1 FarmBot Account and Sequence

#### 3.1.1 Account Set-up

Set-up a FarmBot account and link the account with the FarmBot hardware used at my.farm.bot. (Or use an account already linked with FarmBot equipment.)

#### 3.1.2 Coordinates

Using the web application linked with your FarmBot, there should be a FarmBot position in the upper right hand corner of the web page. Make sure that this number updates as you move the end-effector around the plant bed. For all of the plant channels on the plant bed that you plan to be accessed by the gantry, move the end-effector camera directly overhead the hole location and record the X, Y coordinates of the location.

#### 3.1.3 Logging Plant Points

With all of the X, Y coordinates logged, go to the 'Points' tab on the web application in the top left corner. Manually record all locations (should be a '+' symbol) with the X, Y coordinates. Name as you please but ensure that you know which points are which plants.

#### 3.1.4 Sequence Set-up

To follow our exact routine, we did a raster sequence which went down a channel, then up the next. This sequence continued until the very last channel. Since sharing custom sequences is not presently supported by FarmBot, the next steps will share which custom sequences were created. Sequences are created in the 'Sequences' tab with the '+' symbol and use a library of pre-programmed sequences, combinable with a block programming interface.

#### 3.1.5 Camera Sequence around Plant

Create a 'Camera Sequence' by dragging blocks into this order:

- Wait 7500 ms
- Move: Location 'Offset from current location' x: 20, y: 0, z: 0
- Wait 7500 ms
- Move: Location 'Offset from current location' x: -40, y: 0, z: 0
- Wait 7500 ms

Note) The sequence is three locations around each plant to get different views. If you would like to change the number of images around the plant, you would need to change the python files in the ros_farmbot source code. Waiting 7500 ms is required to ensure the computer has written the RGB-D image locally.

#### 3.1.6 Path Planning

Pre-plan the exact sequence you would like the gantry to reach the certain points. For a raster sequence, we would recommend starting at the top left point in the bed, moving down the channel, and then moving up the subsequent channel. Physically write down the order of the X, Y coordinates, as these will be recorded in important package files.

#### 3.1.7 Channel Raster Sequences

Create individual 'N channel raster' sequences, which control the order of plants accessed within a channel. There should be one for each relevant channel, and should be:

- Move: Location 'Plant Name 1'
- Camera Sequence
- Move: Location 'Plant Name 2'
- Camera Sequence
- Move: Location 'Plant Name 3'
- Camera Sequence
- Move: Location 'Plant Name 4'
- Camera Sequence
- Repeat for as many are in the channel


#### 3.1.8 Full Sequence

Combine all of these sequences to create the main 'Raster Sequence', which should have the sequence:

- Find Home (all)
- Wait 7500 ms
- 1st channel raster
- 2nd channel raster
- ...
- Find Home (all)
- Reboot

#### 3.1.9 Find the Sequence ID

To ensure the package is able to connect to your account and the relevant sequence, start by creating a new sequence. Make the first block the 'Raster Sequence', and in the Settings of the sequences, toggle 'View CeleryScript' on. Read the sequence_id parameter in the 'Raster Sequence' block. 

In the configuration file (~/ros_farmbot_data/config.txt), add the first three lines (email, password, sequence id)

#### 3.1.10 Source Code Edits

The source code for the ros_farmbot package hard codes the locations of the plants, so these need to be updated to match up. The locations are:

- ~/ros2_ws/src/ros_farmbot/central_i2grow_automation.py Line 154
- ~/ros2_ws/src/ros_farmbot/scripts/ui_printing.py Line 101

Change the present list of tuples to a list of tuples with your coordinates in the order of the sequence. Keep the final element (10000,10000) there.

### 3.2 Webcam Device Path

With the relevant webcam plugged in through the USB, run this:

```
ls -l /dev/v4l/by-id
```

Add to the configuration file in line 4 /dev/v4l/by-d/$WEBCAMID$ where $WEBCAMID$ is replaced by the relevant identifier. For example, it should look like (/dev/v4l/by-id/usb-046d_Brio_101_2350AP046D38-video-index0)

### 3.3 Aranet4 MAC Address

To find the MAC address for Bluetooth communication between the computer and the Aranet4 device, run this:

```
bluetoothctl
scan on
```

Find the MAC address starting with the correct OUI - DB:96:37 - and record that in the next line of the config.txt.

### 3.4 CNN Trained Weights

Download this file (https://drive.google.com/file/d/19PKQWIysDWvG2YIM0W8fa6U0QUUJ4cTC/view?usp=sharing) and record the file location in the config.txt in the next line.

### 3.5 Lighting Grid

#### 3.5.1 PAR Sensor Readings

With the PAR sensor plugged into the computer, run:

```
cd ~/ros2_ws/src/ros_farmbot/ros_farmbot/scripts
python par_sensor_reading.py
```

The python file should print out the lighting measurements at some frequency. Move the sensor manually to the plant locations and record the lighting measurement at those points with the lights turned on. 

Inside the ~/ros_farmbot_data/par_sampled_grid.txt, record the lighting measurements by (X, Y, lighting) in each row of the txt. Note that the positions should be in the same order that the FarmBot sequence reaches the points and lighting value should be rounded up to an integer value. Do not worry about the subsequent numbers in the line.

#### 3.5.2 Configuration

In the config.txt, record the location of this lighting grid txt file, which should just be /home/$USER$/ros_farmbot_data/par_sampled_grid.txt, in the last line.

Lastly, there is a hard-coded location of the PAR sensor in the plant  bed in the source code. Go to line 277 in central_i2grow_automation.py in the package (see 3.1.10) and edit this tuple to the rough coordinates of the sensor once attached to the plant bed.

## 4 Executing the Package

There are two options given for running the package through ROS2 launch

### 4.1 No Ros-Bag Package (Basic Launch / Recommended)

In the terminal, run:
```
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ros_farmbot basic.launch
```

### 4.2 Ros-Bag Package (Ros-Record Launch)

In the terminal, run:
```
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash 
cd ~/ros_farmbot_data/ros_bags
ros2 launch ros_farmbot ros-record.launch
```

This will record a rosbag with images and environmental readings.

## 4.3 Documentation ~ Issues/Future Work

- FarmBot Sequence Sharing
- ROSBAG Reading
- Different Mass Prediction Pipelines



