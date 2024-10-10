import os
import json
import matplotlib.pyplot as plt
import glob


# Define the directory and file pattern
base_dir = 'Savings/Actuation_Decoup'
file_pattern = 'Actuation_Decoup*.txt'
files = sorted(glob.glob(os.path.join(base_dir, file_pattern)))

# Iterate over each file and generate plots
for file_path in files:
    # Initialize lists to hold data
    timestamps = []
    sensor_data_S1_x = []
    sensor_data_S1_y = []
    sensor_data_S1_z = []
    sensor_data_S2_x = []
    sensor_data_S2_y = []
    sensor_data_S2_z = []
    prediction_fx_S1 = []
    prediction_fy_S1 = []
    prediction_fz_S1 = []
    prediction_fx_S2 = []
    prediction_fy_S2 = []
    prediction_fz_S2 = []
    pressure_reading1 = []
    pressure_reading2 = []

    # Read and parse the data file
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            timestamps.append(data["timestamp"])
            sensor_data_S1_x.append(data["sensor_data_S1_x"])
            sensor_data_S1_y.append(data["sensor_data_S1_y"])
            sensor_data_S1_z.append(data["sensor_data_S1_z"])
            sensor_data_S2_x.append(data["sensor_data_S2_x"])
            sensor_data_S2_y.append(data["sensor_data_S2_y"])
            sensor_data_S2_z.append(data["sensor_data_S2_z"])
            prediction_fx_S1.append(data["prediction_fx_S1"])
            prediction_fy_S1.append(data["prediction_fy_S1"])
            prediction_fz_S1.append(data["prediction_fz_S1"])
            prediction_fx_S2.append(data["prediction_fx_S2"])
            prediction_fy_S2.append(data["prediction_fy_S2"])
            prediction_fz_S2.append(data["prediction_fz_S2"])
            pressure_reading1.append(data["pressure_reading1"])
            pressure_reading2.append(data["pressure_reading2"])

    # Generate plots for this file
    file_title = os.path.basename(file_path)

    # 1. Flux Plot for both S1 and S2
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, sensor_data_S1_x, 'r-', label='S1 Bx')
    plt.plot(timestamps, sensor_data_S1_y, 'g-', label='S1 By')
    plt.plot(timestamps, sensor_data_S1_z, 'b-', label='S1 Bz')
    plt.plot(timestamps, sensor_data_S2_x, 'r--', label='S2 Bx')
    plt.plot(timestamps, sensor_data_S2_y, 'g--', label='S2 By')
    plt.plot(timestamps, sensor_data_S2_z, 'b--', label='S2 Bz')
    plt.title(f'Flux Plot for S1 and S2 - {file_title}')
    plt.xlabel('Timestamp')
    plt.ylabel('Flux (T)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Force Plot for both S1 and S2
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, prediction_fx_S1, 'r-', label='S1 Fx')
    plt.plot(timestamps, prediction_fy_S1, 'g-', label='S1 Fy')
    plt.plot(timestamps, prediction_fz_S1, 'b-', label='S1 Fz')
    plt.plot(timestamps, prediction_fx_S2, 'r--', label='S2 Fx')
    plt.plot(timestamps, prediction_fy_S2, 'g--', label='S2 Fy')
    plt.plot(timestamps, prediction_fz_S2, 'b--', label='S2 Fz')
    plt.title(f'Force Plot for S1 and S2 - {file_title}')
    plt.xlabel('Timestamp')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Pressure Plot for both S1 and S2
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, pressure_reading1, 'm-', label='S1 Pressure')
    plt.plot(timestamps, pressure_reading2, 'c-', label='S2 Pressure')
    plt.title(f'Pressure Plot for S1 and S2 - {file_title}')
    plt.xlabel('Timestamp')
    plt.ylabel('Pressure (kPa)')
    plt.legend()
    plt.grid(True)
    plt.show()

debug = 1