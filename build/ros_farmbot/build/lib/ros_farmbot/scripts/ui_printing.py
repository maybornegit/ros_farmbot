import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd

from .cnn_models import *
from .image_functions import *

ui_var = '/ui_update_variables.csv'
sensor = '/sensor-readings.csv'
rgbd_loc = '/rgbd_frames'
placeholders = '/placeholders'
my_dir = os.path.expanduser("~/ros_farmbot_data")

try:
    with open(my_dir+'/config.txt') as f:
        lines = [line for line in f]
    netwgts_file = lines[5][:-1]
except:
    print("Need to update config.txt file")
    raise NotImplementedError

try:
    with open(my_dir+'/ui_order.txt') as f:
        order_lines = [line for line in f]
    default_order = False
except:
    default_order = True
    print("Need to update ui_order.txt file - default order")

# netwgts_file = "/home/frc-ag-2/Downloads/old_cnn_results/2025_0203_test-1-effnet-c.pt"
net = load_net(netwgts_file)
net.eval()

# Necessary inputs: timecodes of prev batch, current date, current ee loc, gopro photo
# Outputs: The complete image array with the concatenated pieces
def ui_output(timecode_batch=None, date=None, ee_loc=None, csv_indxs=None):
    img_base = my_dir+placeholders

    img_locations = ['/previous_rgb_placeholder.jpg','/raster_seq_placeholder.jpg','/go_pro_placeholder.jpg','/time_Series_placeholder.jpg']

    im1 = cv2.imread(img_base+img_locations[0])
    im2 = cv2.imread(img_base+img_locations[1])
    im3 = cv2.imread(img_base+img_locations[2])
    im4 = cv2.imread(img_base+img_locations[3])

    natural_shape = im3.shape[0], im3.shape[1]
    new_shape1 = 768,675
    new_shape2 = 312,675
    new_shape3 = 540,1245
    new_shape4 = 540,1245

    ###### Previous RGB Images
    # Each image is 135 x 192 in a 5x4 grid
    # Necessary inputs: timecodes of prev batch and current date
    try:
        if timecode_batch != None and date != None and default_order:
            #test_timecodes = ['10:48:29','10:48:38','10:48:46','10:48:55','10:49:02','10:49:10','10:49:19','10:49:28','10:49:36','10:49:45','10:49:54','10:50:00','10:50:09','10:50:17','10:50:26','10:50:46','10:50:54','10:51:03','10:51:12']
            test_date = date.strftime("%Y-%m-%d")

            im_list = []
            locs_per_plant = 1
            for i in range(len(timecode_batch)):
                if timecode_batch[i] != None and timecode_batch[i] != '':
                    pathloc = my_dir+rgbd_loc+'/rgbd-image-'+ test_date + '-' + timecode_batch[i] +'_'+str(i*locs_per_plant)+ '.npy'
                    rgbd = prep_image(pathloc)
                    wgt = get_wgt_estimate(net, rgbd)
                    if wgt < 1.0:
                        wgt = 0.0
                    
                    picture_npy = np.load(pathloc)[:,:,:3].astype(np.uint8)
                    im = cv2.cvtColor(picture_npy, cv2.COLOR_BGR2RGB)
                    im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)  ######
                    im = labelled_image(im, wgt, i)
                    im = cv2.resize(im, (135,192),interpolation=cv2.INTER_CUBIC)      ######
                    im_list.append(im)
                else:
                    im_list.append(np.zeros((192,135,3),np.uint8)) ######
            for i in range(20-len(timecode_batch)):
                im_list.append(np.zeros((192,135,3),np.uint8))   #######


            # row1_pics = cv2.hconcat(im_list[:5])
            # row2_pics = cv2.hconcat(im_list[5:10])
            # row3_pics = cv2.hconcat(im_list[10:15])
            # row4_pics = cv2.hconcat(im_list[15:])
            col1_pics = cv2.vconcat(im_list[:4])
            col2_pics = cv2.vconcat(im_list[4:8][::-1])
            col3_pics = cv2.vconcat(im_list[8:12])
            col4_pics = cv2.vconcat(im_list[12:16][::-1])
            col5_pics = cv2.vconcat(im_list[16:][::-1])
            full_image_prev_images = cv2.hconcat((col1_pics, col2_pics, col3_pics, col4_pics, col5_pics))
            # full_image_prev_images = cv2.vconcat((row1_pics, row2_pics,row3_pics,row4_pics))
        elif timecode_batch != None and date != None and not default_order:
            grid_dims = order_lines[0].replace('(', '').replace(')', '').replace('\n', '')
            grid_dims = grid_dims.split(',')
            image_sizes = (new_shape1[0]//int(grid_dims[0]), new_shape1[1]//int(grid_dims[1]))
            im_list = [[np.zeros((image_sizes[0],image_sizes[1],3),np.uint8) for __ in range(int(grid_dims[0]))] for _ in range(int(grid_dims[1]))]
            print(grid_dims, image_sizes)

            locs_per_plant = 1
            test_date = date.strftime("%Y-%m-%d")
            for i in range(len(timecode_batch)):
                if timecode_batch[i] != None and timecode_batch[i] != '':
                    pathloc = my_dir + rgbd_loc + '/rgbd-image-' + test_date + '-' + timecode_batch[i] + '_' + str(
                        i * locs_per_plant) + '.npy'
                    rgbd = prep_image(pathloc)
                    wgt = get_wgt_estimate(net, rgbd)
                    if wgt < 1.0:
                        wgt = 0.0

                    picture_npy = np.load(pathloc)[:, :, :3].astype(np.uint8)
                    im = cv2.cvtColor(picture_npy, cv2.COLOR_BGR2RGB)
                    im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)  ######
                    im = labelled_image(im, wgt, i)
                    im = cv2.resize(im, (image_sizes[1],image_sizes[0]), interpolation=cv2.INTER_CUBIC)  ######

                    idx = order_lines[2+i].replace('(', '').replace(')', '').replace('\n', '')
                    print(idx)
                    idx = idx.split(',')
                    im_list[int(idx[1])-1][int(idx[0])-1] = im
            print("J")
            col_pics = [cv2.vconcat(im_list[i][::-1]) for i in range(len(im_list))]
            print(len(col_pics))
            full_image_prev_images = cv2.hconcat(col_pics)
            print("K")
        else:
            full_image_prev_images = im1.copy()
        full_image_prev_images = cv2.resize(full_image_prev_images, (new_shape1[1], new_shape1[0]),interpolation=cv2.INTER_CUBIC)  #####
    except Exception as e:
        print("Error with Previous Images: ",e)
        full_image_prev_images = im1.copy()
        full_image_prev_images = cv2.resize(full_image_prev_images, (new_shape1[1], new_shape1[0]),
                                      interpolation=cv2.INTER_CUBIC)  #####

    ###### Raster Sequence Placeholder
    # Necessary inputs: current ee location
    try:
        if ee_loc != None:
            #test_rasterlocation = (100,200)
            locs_ = [(150,670),(150,460),(150,260),(150,60),(360,70),(360,260),(360,470),(360,665),(565,770),(565,570),(565,375),(565,165),(770,70),(770,270),(770,475),(770,680),(970,175),(970,370),(970,575),(100000,100000)]

            plug_x = [loc[0] for loc in locs_]
            plug_y = [loc[1] for loc in locs_]

            fig = plt.figure(1)
            plt.scatter(plug_x, plug_y, label="Plants", color='green', s=300)
            plt.scatter(ee_loc[0],ee_loc[1], label='Robot', color='red')
            plt.xlim(-50,1250)
            plt.ylim(-50,850)
            plt.title('Robot Location')
            plt.legend()
            plt.xlabel('X')
            plt.xlabel('Y')

            fig.canvas.draw()
            fullimage_raster = np.array(fig.canvas.renderer.buffer_rgba())
            plt.clf()
            fullimage_raster = cv2.cvtColor(fullimage_raster, cv2.COLOR_BGR2RGB)
        else:
            fullimage_raster = im2.copy()
    except:
        print("Error with Raster")
        fullimage_raster = im2.copy()
    fullimage_raster = cv2.resize(fullimage_raster, (new_shape2[1], new_shape2[0]),interpolation=cv2.INTER_CUBIC)  #####

    ###### Go Pro
    
    fullimage_gopro = im3.copy()
    fullimage_gopro = cv2.resize(fullimage_gopro, (new_shape4[1], new_shape4[0]), interpolation=cv2.INTER_CUBIC)

    ###### Time Series Placeholder
    # Necessary inputs: current date
    try:
        if date != None:
            #initialtime_test = dt.datetime(2024,2,4)
            #date = dt.datetime(date.year, date.month, date.day)
            lasthour = dt.timedelta(hours=1)
            data = pd.read_csv(my_dir+sensor)
            #print(csv_indxs)
            if csv_indxs == None:
                idxs = []
                timestamps = []
                for i in range(len(data.index)):
                    testtime = dt.datetime.strptime(data.loc[i]["Timestamp"],"%Y-%m-%d %H:%M:%S")
                    if (testtime < date) and (testtime +lasthour > date):
                        idxs.append(i)
                        timestamps.append(testtime)
                    if (testtime > date):
                        break

                if idxs == []:
                    raise ValueError
            else:
                idxs = []
                timestamps = []
                for i in range(min(csv_indxs),len(data.index)):
                    testtime = dt.datetime.strptime(data.loc[i]["Timestamp"],"%Y-%m-%d %H:%M:%S")
                    if (testtime < date) and (testtime +lasthour > date):
                        idxs.append(i)
                        timestamps.append(testtime)
                    if (testtime > date):
                        break

                if idxs == []:
                    raise ValueError
            
            temp = data.loc[idxs]["Temperature [in degrees C]"].to_numpy()
            rh = data.loc[idxs]["Relative Humidity [%]"].to_numpy()
            co2 = data.loc[idxs]["CO2 [in ppm]"].to_numpy()
            pres = data.loc[idxs]["Pressure [in hPa]"].to_numpy()
            timestamps = [(date - np.array(timestamps))[i].total_seconds()/60 for i in range(len(timestamps))]

            data_list = [temp, rh, co2,pres]
            data_titles = ['Temperature (in the last hour)','RH (in the last hour)','CO2 (in the last hour)','Pressure (in the last hour)']
            data_ylabels = ['Temp. (in C)', 'RH (in %)', 'CO2 (in ppm)', 'Pressure (in hPa)']
            im_list = []
            for i in range(len(data_list)):
                fig = plt.figure(i)
                plt.plot(timestamps, data_list[i])
                plt.title(data_titles[i])
                plt.xlabel('Minutes')
                plt.ylabel(data_ylabels[i])
                
                fig.canvas.draw()
                plot = np.array(fig.canvas.renderer.buffer_rgba())
                plt.clf()
                plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
                plot = cv2.resize(plot, (natural_shape[1]//2, natural_shape[0]//2),interpolation = cv2.INTER_CUBIC)
                im_list.append(plot)

            row1_pics = cv2.hconcat(im_list[:2])
            row2_pics = cv2.hconcat(im_list[2:])
            full_image_env = cv2.vconcat((row1_pics, row2_pics))
        else:
            full_image_env = im4.copy()
    except Exception as error:
        print("Error with Environmental Charts", error)
        idxs = None
        full_image_env = im4.copy()
    full_image_env = cv2.resize(full_image_env, (new_shape3[1], new_shape3[0]), interpolation=cv2.INTER_CUBIC)  ########

    ###### Final Concatenation

    # row1 = cv2.hconcat((full_image_prev_images, fullimage_raster))
    # row2 = cv2.hconcat((fullimage_gopro, full_image_env))
    print(full_image_prev_images.shape, fullimage_raster.shape)
    col1 = cv2.vconcat((full_image_prev_images, fullimage_raster))
    col2 = cv2.vconcat((fullimage_gopro, full_image_env))
    # full_image = cv2.vconcat((row1, row2))
    full_image = cv2.hconcat((col1, col2))
    full_image = cv2.resize(full_image, (1860,1020),interpolation = cv2.INTER_CUBIC)
    return full_image, idxs

if __name__ == "__main__":
    fullimage, csv_timecodes = ui_output(None, dt.datetime.today(), (0,0,0), None)
    #
    # #test_timecodes = ['10:48:29','10:48:38','10:48:46','10:48:55','10:49:02','10:49:10','10:49:19','10:49:28','10:49:36','10:49:45','10:49:54','10:50:00','10:50:09','10:50:17','10:50:26','10:50:46','10:50:54','10:51:03','10:51:12']
    # test_timecodes = ['16:44:50','16:49:04','16:51:18','16:51:27','16:51:36','16:51:56']
    # testimage_location = '/home/kantor-lab/Documents/i2grow_central_computer/testphoto_gopro.JPG'
    # test_rasterlocation = (100,200)
    # #initialtime_test = dt.datetime(2024,2,4,10,50,0)
    # initialtime_test = dt.datetime(2024,3,1,16,57,0)
    # full_image = ui_output(test_timecodes, initialtime_test, test_rasterlocation, testimage_location)
    '''full_image = cv2.resize(fullimage, (1850,1020),interpolation=cv2.INTER_CUBIC)
    while True:
        cv2.imshow("Code Output", full_image)
        cv2.waitKey(0)
        sys.exit()'''
