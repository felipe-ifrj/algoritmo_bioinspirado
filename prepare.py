import pandas as pd
from snapshot import Snapshot
import os
import logging
from device import Device
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

NUMBER_OF_DEVICES = 9
NUMBER_OF_SIMULATIONS = 10

beta_coefficients = {
    "1": 0.01,   # Danmini_Doorbell
    "2": 0.05,   # Ecobee_Thermostat
    "3": 0.01,   # Ennio_Doorbell
    "4": 0.08,   # Philips_Baby_Monitor
    "5": 0.09,   # Provision_PT_737E
    "6": 0.09,   # Provision_PT_838
    "7": 0.09,   # Samsung_Webcam
    "8": 0.1,    # SimpleHome_XCS7_1002
    "9": 0.1     # SimpleHome_XCS7_1003
}

simulation_variance = {
    "1": 0.1,   # Danmini_Doorbell
    "2": 0.2,   # Ecobee_Thermostat
    "3": 0.1,   # Ennio_Doorbell
    "4": 0.9,   # Philips_Baby_Monitor
    "5": 0.8,   # Provision_PT_737E
    "6": 0.7,   # Provision_PT_838
    "7": 0.7,   # Samsung_Webcam
    "8": 0.7,   # SimpleHome_XCS7_1002
    "9": 0.7    # SimpleHome_XCS7_1003
}


def prepare():
    all_devices = {}
    
    plots = {}
    max_cpu = min_cpu = 0
    max_memory = min_memory = 0
    max_bandwidth = min_bandwidth = 0
    cpus = []
    memories = []
    bandwidths = []

    for i in range(NUMBER_OF_DEVICES):
        logging.info(f"Processing device file {i+1}")
        file_name = f'dataset_original//{i+1}.benign.csv'
        df = pd.read_csv(file_name)
        t= 0 
        for row_tuple in df.itertuples(index=True, name='PandasRow'):
            try:
                devices = all_devices[t]
            except:
                devices = {}
                all_devices[t] = devices
            w = getattr(row_tuple, 'MI_dir_L5_weight')
            m = getattr(row_tuple, 'MI_dir_L5_mean')
            v = getattr(row_tuple, 'MI_dir_L5_variance')
            snapshot = Snapshot(f'{i+1}_0', i+1, w, m, v)
            device = Device()
            device.from_snapshot(snapshot, beta_coefficients[str(i+1)])
            #devices [snapshot.id] = device
            cpus.append(device.cpu)
            memories.append(device.memory)
            bandwidths.append(device.bandwidth)
            for j in range(NUMBER_OF_SIMULATIONS):
                simulated_snapshot = snapshot.clone_and_simulate(j+1, simulation_variance=simulation_variance[str(i+1)])
                simulated_device = Device()
                simulated_device.from_snapshot(simulated_snapshot,beta_coefficients[str(i+1)])
                devices [simulated_device.id] = simulated_device
                cpus.append(simulated_device.cpu)
                memories.append(simulated_device.memory)
                bandwidths.append(simulated_device.bandwidth)
         
            t = t + 1
            if row_tuple.Index >= 13112:
                logging.info(f"Reached row limit for device {i+1}, stopping at index {row_tuple.Index}.")
                break

    last_batteries = {}
    min_cpu = min(cpus)
    max_cpu = max(cpus)
    min_memory = min(memories)
    max_memory = max(memories)
    min_bandwidth = min(bandwidths)
    max_bandwidth = max(bandwidths)
    
    for t, devices in all_devices.items():
        for id, device in devices.items():
            try:
                last_battery = last_batteries[id]
            except:
                last_battery = 1
            try:
                data = plots[device.id]
            except:
                data = ([], [], [], [], [])
                plots[device.id] = data

            if device.battery < 0.05:
                device.cpu = 0
                device.memory = 0
                device.bandwidth = 0
                device.battery = 0
                logging.info(f"Device {device.id} battery depleted at time {t}.")
            else:
                plot_cpu = (device.cpu - min_cpu) / (max_cpu - min_cpu)
                if plot_cpu == 0:
                    plot_cpu = 0.01
                plot_memory = (device.memory - min_memory) / (max_memory - min_memory)
                if plot_memory == 0:
                    plot_memory = 0.01
                plot_bandwidth = (device.bandwidth - min_bandwidth) / (max_bandwidth - min_bandwidth)
                if plot_bandwidth == 0:
                    plot_bandwidth = 0.01
                                                
                device.cpu = plot_cpu
                device.memory = plot_memory
                device.bandwidth = plot_bandwidth
                device.update_battery(last_battery)
                last_batteries[id] = device.battery
                plot_battery = device.battery
                
    
            data[0].append(t)
            data[1].append(plot_cpu)
            data[2].append(plot_memory)
            data[3].append(plot_bandwidth)
            data[4].append(plot_battery)
          
    save_to_csv(t, all_devices)
    save_plots(plots)

def save_plots(plots):
    for id, device_plot in plots.items():
        file_name = f'dataset_transformed//graphic_files//device_{id}.png'
        if os.path.exists(file_name):
           # logging.info("Removing plot file.")
            os.remove(file_name) 
        logging.info(f"Plotting device {id}")
        plt.figure()

        plt.plot(device_plot[0], device_plot[1], label='CPU')
        plt.plot(device_plot[0], device_plot[2], label='Memory')
        plt.plot(device_plot[0], device_plot[3], label='Bandwidth')
        plt.plot(device_plot[0], device_plot[4], label='Battery', linewidth=1)
        
        plt.xlabel('Time')
        plt.ylabel('Normalized Values')
        plt.title(f'Device {id}')
        plt.legend()
        plt.savefig(file_name, dpi=300)
        plt.close()

def save_to_csv(t, all_devices):
    for t, devices in all_devices.items():
        logging.info(f"Saving devices at time {t} to CSV.")
        file_name = f'dataset_transformed//csv_files//t_{t}.csv'
        if os.path.exists(file_name):
            os.remove(file_name)   
    
        with open(file_name, 'a') as f:
            header = 'id,type,cpu,memory,battery,bandwidth\n'
            f.write(header)
            for id, device in devices.items():
                f.write(device.to_csv())

if __name__ == '__main__':
    prepare()