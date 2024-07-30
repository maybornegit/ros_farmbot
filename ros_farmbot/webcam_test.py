# import the opencv library 
import cv2, time, csv, sys
import datetime as dt

sys.path.append("/home/frc-ag-2/ros2_ws/src/ros_farmbot/ros_farmbot")
from ui_printing import *

def read_ui_updates():
	in_path = '/home/frc-ag-2/Downloads/ros_test_non_py/ui_update_variables.csv'
	try:
		with open(in_path, 'r', newline='') as file:
			reader = csv.reader(file)
			read_list = list(reader)
			timeimages_batch = read_list[0]
			position = (float(read_list[1][0]),float(read_list[1][1]),float(read_list[1][2]))
			prev_images = read_list[2]
			iter_ = int(read_list[3][0])
	except:
		time.sleep(1)
		with open(in_path, 'r', newline='') as file:
			reader = csv.reader(file)
			read_list = list(reader)
			timeimages_batch = read_list[0]
			position = (float(read_list[1][0]),float(read_list[1][1]),float(read_list[1][2]))
			prev_images = read_list[2]
			iter_ = read_list[3][0]

	return timeimages_batch, position, prev_images, iter_

def display_update(time_ui, csv_timecodes,fullimage,override=False):
	if time.time() - time_ui > 10 or override:
		timeimages_batch, position, previmage_batch, iter_ = read_ui_updates()

		print("DISPLAY UPDATE", timeimages_batch)
		if len(timeimages_batch) == 0 and iter_ == 0:
			fullimage, csv_timecodes = ui_output(None, dt.datetime.today(), position) #put this env readings
		elif len(timeimages_batch) == 0 and iter_ > 0:
			fullimage, csv_timecodes = ui_output(previmage_batch, dt.datetime.today(), position,csv_timecodes) #put this env readings
		elif len(timeimages_batch) > 0:
			fullimage, csv_timecodes = ui_output(timeimages_batch, dt.datetime.today(), position, csv_timecodes)

		time_ui = time.time()
	return fullimage, csv_timecodes, time_ui

def main():
	# define a video capture object 
	
	vid = cv2.VideoCapture(8)
	time_ui = time.time()
	fullimage, csv_timecodes,time_ui = display_update(time_ui,None,None, override=True)
	time_restart = time.time()

	while(True): 
		
		# Capture the video frame 
		# by frame 

		ret, frame = vid.read() 

		frame = cv2.resize(frame, (930,510))
		fullimage[510:,:930,:] = frame
		# Display the resulting frame 
		cv2.imshow('I2GROW Automation Updates', fullimage) 
		
		# the 'q' button is set as the 
		# quitting button you may use any 
		# desired button of your choice 
		if cv2.waitKey(1) & 0xFF == ord('q'): 
			break

		fullimage, csv_timecodes, time_ui = display_update(time_ui,csv_timecodes,fullimage)
		

	# After the loop release the cap object 
	vid.release() 
	# Destroy all the windows 
	cv2.destroyAllWindows()
	return None

if __name__ == '__main__':
	main()
