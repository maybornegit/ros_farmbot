# import the opencv library 
import cv2, time, csv, sys, os
import datetime as dt
import re
# import rclpy
# from rclpy.node import Node

from .scripts.ui_printing import *

ui_var = '/ui_update_variables.csv'
sensor = '/sensor-readings.csv'
rgbd = '/rgbd_frames'
placeholders = '/placeholders'
my_dir = os.path.expanduser("~/ros_farmbot_data")
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

try:
    with open(my_dir+'/config.txt') as f:
        lines = [line for line in f]
    id_ = lines[3][:-1]
except:
    print("Need to update config.txt file")
    raise NotImplementedError

def read_ui_updates():
	in_path = my_dir+ui_var
	try:
		with open(in_path, 'r', newline='') as file:
			reader = csv.reader(file)
			read_list = list(reader)
			timeimages_batch = read_list[0]
			if read_list[1][0] != '':
				position = (float(read_list[1][0]),float(read_list[1][1]),float(read_list[1][2]))
				prev_images = read_list[2]
			else:
				raise ValueError
			iter_ = int(read_list[3][0])
	except:
		time.sleep(10)
		with open(in_path, 'r', newline='') as file:
			reader = csv.reader(file)
			read_list = list(reader)
			timeimages_batch = read_list[0]
			if read_list[1][0] != '':
				position = (float(read_list[1][0]),float(read_list[1][1]),float(read_list[1][2]))
				prev_images = read_list[2]
			else:
				position = None
				prev_images = None
			iter_ = read_list[3][0]

	return timeimages_batch, position, prev_images, iter_

def display_update(time_ui, csv_timecodes,fullimage,override=False):
	if time.time() - time_ui > 10 or override:
		timeimages_batch, position, previmage_batch, iter_ = read_ui_updates()
		
		if dt.datetime.now().hour < 10:
			date = dt.datetime.today()-dt.timedelta(days=1)
		else:
			date = dt.datetime.today()
		
		print("DISPLAY UPDATE", timeimages_batch)
		if iter_ != '':
			if len(timeimages_batch) == 0 and iter_ == 0:
				fullimage, csv_timecodes = ui_output(None, date, position) #put this env readings
			elif len(timeimages_batch) == 0 and iter_ > 0:
				fullimage, csv_timecodes = ui_output(previmage_batch, date, position,csv_timecodes) #put this env readings
			elif len(timeimages_batch) > 0:
				fullimage, csv_timecodes = ui_output(timeimages_batch, date, position, csv_timecodes)

			time_ui = time.time()
	return fullimage, csv_timecodes, time_ui
	
def find_webcam(id_):
    device_num = 0
    if os.path.exists(id_):
        device_path = os.path.realpath(id_)
        device_re = re.compile("\/dev\/video(\d+)")
        info = device_re.match(device_path)
        if info:
            device_num = int(info.group(1))
            #print("Using default video capture device on /dev/video" + str(device_num))

    return device_num
	

def main():
	# define a video capture object 
	# id_ = '/dev/v4l/by-id/usb-046d_Brio_101_2350AP046D38-video-index0'
	device = find_webcam(id_)
	
	vid = cv2.VideoCapture(device)
	time_ui = time.time()
	fullimage, csv_timecodes,time_ui = display_update(time_ui,None,None, override=True)
	time_restart = time.time()

	while(True): 
		# rclpy.init(args=None)

		# node_disp = rclpy.create_node('display_logging')
		# Capture the video frame 
		# by frame 
		try:
			ret, frame = vid.read()

			frame = cv2.resize(frame, (1860-654,510))    ######
			fullimage[:510,654:,:] = frame
			# Display the resulting frame
			cv2.imshow('I2GROW Automation Updates', fullimage)

			# the 'q' button is set as the
			# quitting button you may use any
			# desired button of your choice
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			fullimage, csv_timecodes, time_ui = display_update(time_ui,csv_timecodes,fullimage)
			# node_disp.get_logger().info('Display Updated.')
		except Exception as error:
			print(error)
			exit()
		

	# After the loop release the cap object 
	vid.release() 
	# Destroy all the windows 
	cv2.destroyAllWindows()
	return None

if __name__ == '__main__':
	main()
