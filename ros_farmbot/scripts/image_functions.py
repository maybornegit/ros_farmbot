import torch.nn as nn
import torch
from torchvision import models, transforms
import numpy as np
import cv2, csv, os
import pandas as pd
from tqdm import tqdm
from .cnn_models import EffNetV2640_C, EffNetV2640D, EffNetV2640RGB

my_dir = os.path.expanduser("~/ros_farmbot_data")
est_mass = '/predicted_mass.csv'

def padding_image(image, target_size):
    """
    Pads an image to the target size with equal padding above and below.

    Parameters:
    - image (np.ndarray): The input image to be padded. Shape should be (height, width, channels).
    - target_size (tuple): The target size as (target_height, target_width).

    Returns:
    - np.ndarray: The padded image.
    """
    height, width = image.shape[:2]
    target_height, target_width = target_size

    # Calculate padding amounts
    pad_y = max(target_height - height, 0)
    pad_x = max(target_width - width, 0)
    # Calculate padding for top, bottom, left, and right
    top_pad = pad_y // 2
    bottom_pad = pad_y - top_pad
    left_pad = pad_x // 2
    right_pad = pad_x - left_pad

    # Pad the image
    padded_image = np.pad(
        image,
        ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return padded_image

def labelled_image(image, weight, idx):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 75)
    fontScale = 2
    color = (0, 0, 255)
    thickness = 4
    text = str(round(weight,1))+'g'
    if weight == 0.0:
        text = '<1g'
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, fontScale, thickness)
    rectangle_top_left = (org[0] - 10, org[1] - text_height - 10)
    rectangle_bottom_right = (org[0] + text_width + 10, org[1] + 10)

    cv2.rectangle(image, rectangle_top_left, rectangle_bottom_right, (0, 0, 0), -1)  # -1 to fill the rectangle
    cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    org = (300,600)
    color = (0,255,0)
    with open(my_dir+est_mass, mode='r') as f:
        reader = csv.reader(f)
        read_list = list(reader)
        if read_list[0][idx] != 'N/A':
            text = str(round(float(read_list[0][idx]),1))+'g'
        else:
            text = 'N/A'

    (text_width, text_height), _ = cv2.getTextSize(text, font, fontScale, thickness)
    rectangle_top_left = (org[0] - 10, org[1] - text_height - 10)
    rectangle_bottom_right = (org[0] + text_width + 10, org[1] + 10)

    cv2.rectangle(image, rectangle_top_left, rectangle_bottom_right, (0, 0, 0), -1)  # -1 to fill the rectangle
    cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    return image

def prep_image(filename):
    rgbd = np.load(filename)
    target_size = (640, 640)
    depth_image = rgbd[:, :, 3].astype(np.float32)
    mask = (depth_image == 0).astype(np.uint8)
    smoothed_depth = cv2.inpaint(depth_image.astype(np.float32), mask, inpaintRadius=10,
                                    flags=cv2.INPAINT_TELEA)
    rgbd = np.concatenate((rgbd[:, :, :3], smoothed_depth.reshape((480, 640, 1))), axis=2)
    rgbd = padding_image(rgbd, target_size)
    rgb = rgbd[:, :, :3].astype(np.uint8)
    rgb_ = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb_, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 35, 35])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel_dilation = np.ones((5, 5), np.uint8)
    mask_opened = cv2.dilate(mask_opened, kernel_dilation, iterations=1)

    result = cv2.bitwise_and(rgb_, rgb_, mask=mask_opened)
    filter_depth = 65535
    depth = rgbd[:, :, 3]
    depth[mask_opened == 0] = filter_depth

    rgbd = np.concatenate((cv2.cvtColor(result, cv2.COLOR_RGB2BGR), depth.reshape((640, 640, 1))), axis=2)
    rgbd = np.transpose(rgbd, (2, 0, 1))
    rgbd = torch.from_numpy(rgbd).float().unsqueeze(0)
    return rgbd

def get_wgt_estimate(net,rgbd):
    with torch.no_grad():
        return net(rgbd).item()

def load_net(file_loc):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on the GPU")
    else:
        device = torch.device("cpu")
        print("running on the CPU")

    torch.set_num_threads(1)

    rgb_model = EffNetV2640RGB()
    depth_model = EffNetV2640D()

    net = EffNetV2640_C(rgb_model,depth_model).to(device)

    net.load_state_dict(torch.load(file_loc,map_location=torch.device('cpu')))
    return net
