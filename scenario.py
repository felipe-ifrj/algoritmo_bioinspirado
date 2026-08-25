import random
from device import Device 
import logging
import csv

logging.basicConfig(level=logging.INFO, format='%(message)s')

class Scenario:
    #def __init__(self, devices=None, number_of_sensor_types=9, number_of_devices_per_type=10): #scenario original
    def __init__(self, devices=None, number_of_sensor_types=9, number_of_devices_per_type=10): #scenario 1
    #def __init__(self, devices=None, number_of_sensor_types=9, number_of_devices_per_type=50): #scenario 2
   #def __init__(self, devices=None, number_of_sensor_types=9, number_of_devices_per_type=100): #scenario 3
        self.devices = devices
        self.number_of_sensor_types = number_of_sensor_types
        self.number_of_devices_per_type = number_of_devices_per_type
        logging.info(f'Number of devices simulated: {self.number_of_devices_per_type * self.number_of_sensor_types}')

    def max_clusters(self):
        #return self.number_of_devices_per_type * self.number_of_sensor_types
        return 36

