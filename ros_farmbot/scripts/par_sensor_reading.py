#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:48:08 2024

@author: morganmayborne
"""

from serial import Serial
from time import sleep
import struct, time
import serial.tools.list_ports

GET_VOLT = b'\x55!'
READ_CALIBRATION = b'\x83!'
SET_CALIBRATION = lambda x, y: b'\x84' + x + y + b'!'
READ_SERIAL_NUM = b'\x87!'
GET_LOGGING_COUNT = b'\xf3!'
GET_LOGGED_ENTRY = lambda x: b'\xf2' + x + b'!'
ERASE_LOGGED_DATA = b'\xf4!'

class Quantum:
    def __init__(self):
        """Initializes class variables, and attempts to connect to device"""
        self.quantum = None
        self.offset = 0.0
        self.multiplier = 0.0
        self.connect_to_device()

    def connect_to_device(self):
        """This function creates a Serial connection with the defined comport
        and attempts to read the calibration values"""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if 'SQ-420' in str(p):
                port = str(p).replace(' - SQ-420 - SQ-420', '')
        # port = "/dev/ttyACM0" # you'll have to check your device manager and put the actual com port here
        self.quantum = Serial(port, 115200, timeout=0.5)
        try:
            self.quantum.write(READ_CALIBRATION)
            multiplier = self.quantum.read(5)[1:]
            offset = self.quantum.read(4)
            self.multiplier = struct.unpack('<f', multiplier)[0]
            self.offset = struct.unpack('<f', offset)[0]
        except (IOError, struct.error) as e:
            print(e)
            self.quantum = None

    def get_micromoles(self):
        """This function converts the voltage to micromoles"""
        voltage = self.read_voltage()
        if voltage == 9999:
            # you could raise some sort of Exception here if you wanted to
            return
        # this next line converts volts to micromoles
        micromoles = (voltage - self.offset) * self.multiplier * 1000
        if micromoles < 0:
            micromoles = 0
        return micromoles

    def read_voltage(self):
        """This function averages 5 readings over 1 second and returns
        the result."""
        if self.quantum is None:
            try:
                self.connect_to_device()
            except IOError:
                # you can raise some sort of exception here if you need to
                return 9999

        # store the responses to average
        response_list = []

        # change to average more or less samples over the given time period
        number_to_average = 5

        # change to shorten or extend the time duration for each measurement
        # be sure to leave as floating point to avoid truncation
        number_of_seconds = 1.0

        for i in range(number_to_average):
            try:
                self.quantum.write(GET_VOLT)
                response = self.quantum.read(5)[1:]
            except IOError as e:
                print(e)
                # dummy value to know something went wrong. could raise an
                # exception here alternatively
                return 9999
            else:
                if not response:
                    continue
                # if the response is not 4 bytes long, this line will raise
                # an exception
                voltage = struct.unpack('<f', response)[0]
                response_list.append(voltage)
                sleep(number_of_seconds / number_to_average)

        if response_list:
            return sum(response_list) / len(response_list)

        return 0.0
    
if  __name__ == '__main__':
    ### Change port depending on Linux port
    test_sensor = Quantum()
    while True:
        time.sleep(1)
        print(test_sensor.get_micromoles())
    
