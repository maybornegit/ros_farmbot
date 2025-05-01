import subprocess, time, os

my_dir = os.path.expanduser("~/ros_farmbot_data")
try:
    with open(my_dir+'/config.txt') as f:
        lines = [line for line in f]
    mac_Address = lines[4][:-1]
except:
    print("Need to update config.txt file")
    raise NotImplementedError

def measure_env():
    def measure_check(mask, check_char, text, offset, i):
        measure = []
        j = offset
        while text[i+j] != check_char:
            if text[i+j] in mask:
                measure.append(text[i+j])
            j += 1
        return float(''.join(measure))

    mac_Address = 'DB:96:37:C7:C3:93'

    time_ = 0
    measurements = []
    t = time.time()
    text = subprocess.run(['aranetctl', mac_Address], capture_output=True, text=True).stdout

    mask = ['.','0','1','2','3','4','5','6','7','8','9']
    for i in range(len(text)):
        if text[i:i+3] == 'CO2':
            co2 = measure_check(mask, 'p',text, 3, i)
        if text[i:i+11] == 'Temperature':
            temp = measure_check(mask, 'C',text, 11, i)
        if text[i:i+8] == 'Humidity':
            rh = measure_check(mask, '%',text, 8, i)
        if text[i:i+8] == 'Pressure':
            pres = measure_check(mask, 'h',text, 8, i)
        if text[i:i+7] == 'Battery':
            battery = measure_check(mask, '%',text, 7, i)
    try:
        measurements = [time.strftime('%Y-%m-%d %H:%M:%S'),temp, rh, co2, pres, battery]
        time_ = time.time() - t
        return measurements, time_
    except:
        return [], 0

if __name__ == '__main__':
    m,t = measure_env()
    print([str(m[i]) for i in range(len(m))], t)

        
