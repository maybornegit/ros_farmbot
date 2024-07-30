## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
#if not found_rgb:
#    print("The demo requires Depth camera with Color sensor")
#    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming

def pull_depth_image(current_loc):
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    #if not found_rgb:
    #    print("The demo requires Depth camera with Color sensor")
    #    exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        pipeline.start(config)
        frames = pipeline.wait_for_frames()

        # Wait for a coherent pair of frames: depth and color
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()).reshape((480, 640, 1))
        color_image = np.asanyarray(color_frame.get_data())

        # print(depth_image.shape, color_image.shape)

        rgbd_frame = np.concatenate((color_image, depth_image), axis=-1)

        timestamp = time.strftime('%Y-%m-%d-%H:%M:%S')
        timestamp_ = time.strftime('%H:%M:%S')
        # np.savetxt('/home/kantor-lab/Documents/i2grow_central_computer/rgbd_frames/depth-image-'+f'{timestamp}'+'_'+f'{current_loc}'+'.csv',depth_image, delimiter=',')
        # print(f'{timestamp}'+' : Depth Image Saved' )

        np.save('/home/frc-ag-2/Downloads/ros_test_non_py/rgbd_frames/rgbd-image-'+f'{timestamp}'+'_'+f'{current_loc}'+'.npy',rgbd_frame)
        print(f'{timestamp}'+' : RGBD Image Saved' )
        pipeline.stop()

        device.hardware_reset()
    
    except Exception as error:
        try:
            pipeline.stop()
            #device.hardware_reset()
        except Exception as error_:
            print('Failed pipeline close', error_)

        #device.hardware_reset()

        timestamp_ = None
        print('Failed image', error)

    return timestamp_

#for i in range(10):
#    pull_depth_image()