import sys
sys.path.append("/home/frc-ag-2/ros2_ws/src/ros_farmbot/ros_farmbot")

from opencv_rgbd_framegrab import *
from measure_env import *
import time, csv
from farmbot import Farmbot
import threading
import datetime as dt

'''
def display_image():
    test_window = cv2.namedWindow("code_output", cv2.WINDOW_NORMAL)
    cv2.imshow('code_output', fullimage)
    cv2.waitKey(5000)
'''

def ui_update_writer(timecode_batch,ee_loc, prev_imagebatch, iter_):
    in_path = '/home/frc-ag-2/Downloads/ros_test_non_py/ui_update_variables.csv'
    with open(in_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(timecode_batch)
        writer.writerow(ee_loc)
        writer.writerow(prev_imagebatch)
        writer.writerow([iter_])
    return None

class MyHandler:
    def __init__(self, bot):
        self.bot = bot

    def on_connect(self, bot, mqtt_client):
        #bot.emergency_unlock()
        #bot.find_home()
        bot._do_cs("execute",{"sequence_id": 165495})
        pass

    def on_change(self, bot, state):
        global position, kill, pin_read
        position = (state['location_data']['position']['x'],state['location_data']['position']['y'],state['location_data']['position']['z'])
        pin_read = state["pins"]['18']['value']
        #print(position, kill)
        if kill:
            sys.exit()

    def on_log(self, bot, log):
        global pin_read
        #print("New message from FarmBot: " + log['message'])
        if log['message'] == 'Take a picture.':
            print(log['message'])
            pin_read = 1.0

    def on_response(self, bot, response):
        #print("ID of successful request: " + response.id)
        #print(response)
        pass

    def on_error(self, bot, response):
        print("ID of failed request: " + response.id)
        print("Reason(s) for failure: " + str(response.errors))
    def move10_x(self):
        
        pass

def main():
    global position
    global kill

    # Setting up locations of interest
    position = (0,0,0)
    delta_camera_seq = 20
    locs_ = [(135,685),(135,480),(135,280),(135,80),(365,80),(365,280),(365,480),(365,685),(565,775),(565,575),(565,375),(565,175),(765,80),(765,280),(765,480),(765,685),(960,175),(960,375),(960,575),(100000,100000)]
    locs = []
    for loc in locs_:
        locs.append((loc[0],loc[1]))
        locs.append((loc[0],loc[1]+delta_camera_seq))
        locs.append((loc[0],loc[1]-delta_camera_seq))
        locs.append((loc[0]+delta_camera_seq,loc[1]))
        locs.append((loc[0]-delta_camera_seq,loc[1]))

    # Initialize Variables
    kill = False
    iter_ = 0
    current_loc = 0
    
    # Initialize Farmbot and UI
    timeimages_batch = []
    previmage_batch = []
    ui_update_writer(timeimages_batch,position, previmage_batch, 0)
    fb = Farmbot.login("kantorlab.farmbot@gmail.com", "field123")
    handler = MyHandler(fb)
    t1 = threading.Thread(target=fb.connect, name="foo", args=[handler])
    t1.start()

    # Initialize Timing Objects
    t = time.time()
    time_envreadings = time.time()
    time_image = time.time()
    time_ui = time.time()
    csv_timecodes = None

    # Enter Continuous While Loop
    print("ENTERING CONTINUOUS CONTROL:")
    while True:
        # If it's time for the next sequence ~ note sequence takes about 15 minutes each
        if time.time() - time_ui > 10:
            ui_update_writer(timeimages_batch,position, previmage_batch, iter_)
            time_ui = time.time()

        if time.time() - t > 30*60:
            print('REINITIALIZE SEQUENCE')
            kill = True
            t1.join()
            time.sleep(5)

            t1 = threading.Thread(target=fb.connect, name="foo", args=[handler])
            t1.start()
            kill = False
            pics = True
            current_loc = 0
            t = time.time()

            iter_ += 1
            previmage_batch = timeimages_batch[:]
            timeimages_batch = []
            if time.time() - time_ui > 10:
                ui_update_writer(timeimages_batch,position, previmage_batch, iter_)
                time_ui = time.time()
        
        #Is it at a picture location? If so take picture. If not and its time, take measurements.
        plug_near_X = position[0] <= 1.03*locs[current_loc][0] and position[0] >= 0.97*locs[current_loc][0]
        plug_near_Y = position[1] <= 1.03*locs[current_loc][1] and position[1] >= 0.97*locs[current_loc][1]
        if plug_near_X and plug_near_Y and (time.time() - time_image > 6.5) and (time.time() - t > 20):
            print("position reached:",current_loc)
            time_image = time.time()
            timestamp = pull_depth_image(current_loc)
            if timestamp != None:
                print('IMAGE TAKEN')
            else:
                print('IMAGE ERROR')
            if current_loc % 5 == 0:
                timeimages_batch.append(timestamp)
            
            if time.time() - time_ui > 10:
                ui_update_writer(timeimages_batch,position, previmage_batch, iter_)
                time_ui = time.time()
            current_loc += 1
        elif time.time() - time_envreadings > 60:
            with open('/home/frc-ag-2/Downloads/ros_test_non_py/sensor-readings.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                try:
                    m, _ = measure_env()
                    if len(m) != 0:
                        writer.writerow(m[0:5])
                        print('MEASUREMENT TAKEN')
                except:
                    pass
            time_envreadings = time.time()
    return None
	
if __name__ == '__main__':
	main()
