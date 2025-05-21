import sys
import time, csv, subprocess,os, cv2
from datetime import datetime
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float64,Float64MultiArray

from ros_farmbot.scripts.ui_printing import netwgts_file

ui_var = '/ui_update_variables.csv'
wgt_timeline = '/wgt_estimates.csv'
sensor = '/sensor-readings.csv'
my_dir = os.path.expanduser("~/ros_farmbot_data")
rgbd_loc = '/rgbd_frames'
est_mass = '/predicted_mass.csv'
if not os.path.exists(my_dir):
   os.mkdir(my_dir)
   os.mkdir(my_dir+'/rgbd_frames')
if not os.path.exists(my_dir+ui_var):
   with open(my_dir+ui_var, 'w') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow(['Test','Test','Test', 'Test'])
if not os.path.exists(my_dir+sensor):
   with open(my_dir+sensor, 'w') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow(['Timestamp','Temperature [in degrees C]','Relative Humidity [%]','CO2 [in ppm]','Pressure [in hPa]'])
if not os.path.exists(my_dir+wgt_timeline):
   with open(my_dir+wgt_timeline, 'w') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow(['Mass History'])

try:
    with open(my_dir+'/config.txt') as f:
        lines = [line for line in f]
    email = lines[0][:-1]
    password = lines[1][:-1]
    seq_id = lines[2][:-1]
    netwgts_file = lines[5][:-1]
except:
    print("Need to update config.txt file")
    raise NotImplementedError

try:
    with open(my_dir+'/coords.txt') as f:
        coords = [line for line in f]
except:
    print("Need to update coords.txt file")
    raise NotImplementedError

from farmbot import Farmbot
import threading
import datetime as dt

from ros_farmbot_msgs.msg import Env 
from .scripts.opencv_rgbd_framegrab import *
from .scripts.measure_env import *
from .scripts.par_sensor_reading import *
from .scripts.cnn_models import *
from .scripts.image_functions import *
from .scripts.prediction_helper import *

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
        
'''
def display_image():
    test_window = cv2.namedWindow("code_output", cv2.WINDOW_NORMAL)
    cv2.imshow('code_output', fullimage)
    cv2.waitKey(5000)
'''

def ui_update_writer(timecode_batch,ee_loc, prev_imagebatch, iter_):
    in_path = my_dir+ui_var
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
        bot._do_cs("execute",{"sequence_id": seq_id})
        pass

    def on_change(self, bot, state):
        global position, kill, pin_read
        position = (state['location_data']['position']['x'],state['location_data']['position']['y'],state['location_data']['position']['z'])
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
    
    rclpy.init(args=None)
    br = CvBridge()
    par_sensor = Quantum()

    node_env = rclpy.create_node('publisher_env')
    node_rgb = rclpy.create_node('rgbd_image')
    node_depth = rclpy.create_node('depth_image')
    node_position = rclpy.create_node('position_image')
    node_t = rclpy.create_node('timing')

    publisher_rgb = node_rgb.create_publisher(Image, '/rgbd/color', 10)
    publisher_depth = node_depth.create_publisher(Float64MultiArray, '/rgbd/depth', 10)
    publisher_env = node_env.create_publisher(Env, '/env/aranet', 10)
    publisher_pos = node_position.create_publisher(Float64MultiArray, '/position', 10)
    #publisher_par = self.create_publisher(Float64MultiArray, '/env/par', 10)
    
    
    # Setting up locations of interest
    position = (0,0,0)
    delta_camera_seq = 20
    locs_ = []
    for c in coords:
        c_edit = c.replace('(', '').replace(')', '').replace('\n', '')
        c_edit = c_edit.split(',')
        locs_.append((int(c_edit[0]), int(c_edit[1])))
    locs_.append((100000, 100000))
    # locs_ = [(150,670),(150,460),(150,260),(150,60),(360,70),(360,260),(360,470),(360,665),(565,770),(565,570),(565,375),(565,165),(770,70),(770,270),(770,475),(770,680),(970,175),(970,370),(970,575),(100000,100000)]
    locs = []
    for loc in locs_:
        locs.append((loc[0],loc[1]))
        # locs.append((loc[0]+delta_camera_seq,loc[1]))
        # locs.append((loc[0]-delta_camera_seq,loc[1]))

    # netwgts_file = "/home/frc-ag-2/Downloads/old_cnn_results/2025_0203_test-1-effnet-c.pt"
    net = load_net(netwgts_file)
    net.eval()

    # Initialize Variables
    kill = False
    iter_ = 0
    current_loc = 0
    locs_per_plant = 1
    
    wgttime = datetime.now()
    wgttime = wgttime.strftime("%Y-%m-%d %H")
    # Initialize Farmbot and UI
    timeimages_batch = []
    previmage_batch = []
    wgtimages_batch = [wgttime]
    estmasses = ['N/A' for _ in range(len(locs_)-1)]
    with open(my_dir + est_mass, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(estmasses)
    if position != '':
        ui_update_writer(timeimages_batch,position, previmage_batch, 0)
    fb = Farmbot.login(email, password)
    handler = MyHandler(fb)
    t1 = threading.Thread(target=fb.connect, name="foo", args=[handler])
    t1.start()

    # Initialize Timing Objects
    t = time.time()
    time_envreadings = time.time()
    time_image = time.time()
    time_ui = time.time()
    time_pos = time.time()
    csv_timecodes = None

    # Enter Continuous While Loop
    print("ENTERING CONTINUOUS CONTROL:")
    while True:
        # If it's time for the next sequence ~ note sequence takes about 20 minutes each
        if time.time() - time_pos > 4 and position[0] != None:
            msg_pos = Float64MultiArray()
            msg_pos.data = [float(pos) for pos in position]
            publisher_pos.publish(msg_pos)
            time_pos = time.time()
        if time.time() - time_ui > 10 and position[0] != None:
            ui_update_writer(timeimages_batch,position, previmage_batch, iter_)
            time_ui = time.time()

        ### Stop when the lighting turns off and only run on the hour
        if time.time() - t > 1*60*60 and datetime.now().hour >= 10:
            with open(my_dir+wgt_timeline, mode='a') as f:
                writer = csv.writer(f)
                writer.writerow(wgtimages_batch)

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
            wgttime = datetime.now()
            wgttime = wgttime.strftime("%Y-%m-%d %H")
            wgtimages_batch = [wgttime]
            if time.time() - time_ui > 10:
                ui_update_writer(timeimages_batch,position, previmage_batch, iter_)
                time_ui = time.time()

        #Is it at a picture location? If so take picture. If not and its time, take measurements.
        try:
            plug_near_X = position[0] <= 1.03*locs[current_loc][0] and position[0] >= 0.97*locs[current_loc][0]
            plug_near_Y = position[1] <= 1.03*locs[current_loc][1] and position[1] >= 0.97*locs[current_loc][1]
        except:
            continue
        if plug_near_X and plug_near_Y and (time.time() - time_image > 6.5) and (time.time() - t > 20):
            print("position reached:",current_loc)
            time_image = time.time()
            timestamp, color_image, depth_image = pull_depth_image(current_loc)

            if current_loc % locs_per_plant == 0:
                date = datetime.today().strftime("%Y-%m-%d")

                pathloc = my_dir+rgbd_loc+'/rgbd-image-'+ date + '-' + timestamp +'_'+str(current_loc)+ '.npy'
                rgbd = prep_image(pathloc)
                wgt = get_wgt_estimate(net, rgbd)
                if wgt <= 1:
                    wgt = 0.0
                wgtimages_batch.append(wgt)

                ### Mass Prediction
                try:
                    idx = current_loc // locs_per_plant
                    with open(my_dir+wgt_timeline, mode='r') as f:
                        reader = csv.reader(f)
                        read_list = list(reader)[1:]
                        prev_masses = [(datetime.strptime(r[0],"%Y-%m-%d %H"),float(r[idx+1])) for r in read_list]
                    if len(prev_masses) < 5 or wgt == 0.0:
                        raise NotImplementedError

                    for i in range(len(prev_masses)-5):
                        wind_masses = [(prev_masses[i+j][1] >= 1) for j in range(5)]
                        if sum(wind_masses) == 5:
                            break
                    prev_masses = prev_masses[i:]
                    traj = [[],[]]
                    for m in prev_masses:
                        traj[1].append(m[1])
                        traj[0].append(8+int((m[0]-prev_masses[0][0]).total_seconds()/3600)/24)

                    par_filename_ = my_dir + '/par_sampled_grid.txt'
                    if par_sensor.get_sensor_status():
                        single_meas = float(par_sensor.get_micromoles())
                        sensor_loc = (860, 175)
                        sample_par = run_grid_approx(single_meas, sensor_loc, [locs_[current_loc//locs_per_plant]], par_filename_)
                        light = sample_par[0][2]
                    else:
                        est_par = default_par_meas(par_filename_)
                        light = est_par[current_loc//locs_per_plant][2]

                    mass_predictions = projected_mass(traj, light, [1,2,4,7,10], params = [2.34e-7,22.4,0.5])
                    node_rgb.get_logger().info('Mass Prediction for 4 days: '+str(mass_predictions[2]))

                    estmasses[current_loc//locs_per_plant] = mass_predictions[-2]
                    with open(my_dir+est_mass, mode='w') as f:
                        writer = csv.writer(f)
                        writer.writerow(estmasses)

                except Exception as error:
                    # ideally make the guess for this one None if nothing is available
                    print(error)
                    pass
                    
            color_image = color_image.astype(np.uint8)
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            depth_image = np.array(depth_image, dtype=np.float64)

            node_rgb.get_logger().info('Publishing Image...')
            node_rgb.get_logger().info('Plant Weight: '+str(wgt))
            ### Create Msg
            msg_depth = Float64MultiArray()

            ### Write Msg
            msg_rgb = br.cv2_to_imgmsg(color_image, encoding="passthrough")  # Use "bgr8" if the image is in BGR format
            msg_depth.data = depth_image.flatten().tolist()

            # msg_rgb.header.stamp = node_rgb.get_clock().now().to_msg()
            # msg_depth.header.stamp = node_depth.get_clock().now().to_msg()

            ### Publish Msg
            publisher_rgb.publish(msg_rgb)
            publisher_depth.publish(msg_depth)

            if timestamp != None:
                print('IMAGE TAKEN')
            else:
                print('IMAGE ERROR')
            if current_loc % locs_per_plant == 0:
                timeimages_batch.append(timestamp)

            if time.time() - time_ui > 10 and position != '':
                ui_update_writer(timeimages_batch,position, previmage_batch, iter_)
                time_ui = time.time()
            current_loc += 1
        elif time.time() - time_envreadings > 60:
            with open(my_dir+sensor, 'a', newline='') as file:
                writer = csv.writer(file)
                try:
                    m, _ = measure_env()
                    par_filename_ = my_dir + '/par_sampled_grid.txt'
                    if par_sensor.get_sensor_status():
                        single_meas = float(par_sensor.get_micromoles())
                        sensor_loc = (860, 175)
                        est_par = run_grid_approx(single_meas, sensor_loc, locs_[:-1], par_filename_)
                    else:
                        est_par = default_par_meas(par_filename_)
                    if len(m) != 0 and rclpy.ok():
                        writer.writerow(m[0:5])
                        print('MEASUREMENT TAKEN')

                        ### Declare and Initialize Msg
                        node_env.get_logger().info('Publishing Environmental Data...')
                        msg_env = Env()

                        ### Write Msg
                        msg_env.pressure = float(m[4])
                        msg_env.co2 = float(m[3])
                        msg_env.temp = float(m[1])
                        msg_env.rh = float(m[2])
                        par_msg = Float64MultiArray()
                        par_msg.data = [float(x[2]) for x in est_par]
                        msg_env.par = par_msg
                        msg_env.header.stamp = node_env.get_clock().now().to_msg()

                        ### Publish Msg
                        publisher_env.publish(msg_env)
                except Exception as error:
                    print(error)
                    pass
            time_envreadings = time.time()
        # except Exception as error:
        #     print(error, 'Position Error', (time.time() - t)/60)
        #     time.sleep(10)
        #     pass
    return None

if __name__ == '__main__':
    main()
