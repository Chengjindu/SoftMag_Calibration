import os
import json
import matplotlib.pyplot as plt
import glob
from scipy.signal import cheby1, lfilter, lfilter_zi
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib

# ----------------------------------------------Data Importing-------------------------------------------------------

# Define the directory and file pattern
base_dir = 'Savings/Actuation_Decoup'
file_pattern = 'Actuation_Decoup*.txt'
files = sorted(glob.glob(os.path.join(base_dir, file_pattern)))

# Initialize lists to hold combined data from all files
combined_timestamps = []
combined_sensor_data_S1_x = []
combined_sensor_data_S1_y = []
combined_sensor_data_S1_z = []
combined_sensor_data_S2_x = []
combined_sensor_data_S2_y = []
combined_sensor_data_S2_z = []
combined_pressure_reading1 = []
combined_pressure_reading2 = []

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
            pressure_reading1.append(data["pressure_reading1"])
            pressure_reading2.append(data["pressure_reading2"])

            combined_timestamps.append(data["timestamp"])
            combined_sensor_data_S1_x.append(data["sensor_data_S1_x"])
            combined_sensor_data_S1_y.append(data["sensor_data_S1_y"])
            combined_sensor_data_S1_z.append(data["sensor_data_S1_z"])
            combined_sensor_data_S2_x.append(data["sensor_data_S2_x"])
            combined_sensor_data_S2_y.append(data["sensor_data_S2_y"])
            combined_sensor_data_S2_z.append(data["sensor_data_S2_z"])
            combined_pressure_reading1.append(data["pressure_reading1"])
            combined_pressure_reading2.append(data["pressure_reading2"])

    # Generate plots for this file
    file_title = os.path.basename(file_path)

    # 1. Flux Plot for both S1 and S2
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, sensor_data_S1_x, 'r-', label='S1 Bx')
    plt.plot(timestamps, sensor_data_S1_y, 'g-', label='S1 By')
    plt.plot(timestamps, sensor_data_S1_z, 'b-', label='S1 Bz')
    plt.plot(timestamps, sensor_data_S2_x, 'm--', label='S2 Bx')
    plt.plot(timestamps, sensor_data_S2_y, 'c--', label='S2 By')
    plt.plot(timestamps, sensor_data_S2_z, 'g--', label='S2 Bz')
    plt.title(f'Flux Plot for S1 and S2 - {file_title}')
    plt.xlabel('Timestamp')
    plt.ylabel('Flux (T)')
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

# ----------------------------------------------Filtering-------------------------------------------------------

# Chebyshev filter initialization function (as from previous code)
def initialize_pressure_filter(sampling_freq=60, passband_freq=1, order=1, passband_ripple=0.5):
    nyquist_freq = sampling_freq / 2
    normalized_passband_freq = passband_freq / nyquist_freq

    if not (0 < normalized_passband_freq < 1):
        raise ValueError("Normalized passband frequency is out of range.")

    # Design the Chebyshev type I low-pass filter
    b, a = cheby1(N=order, rp=passband_ripple, Wn=normalized_passband_freq, btype='low')
    zi = lfilter_zi(b, a)
    return b, a, zi

# Apply filtering to the data
def apply_filter(data, b, a, zi):
    filtered_data, _ = lfilter(b, a, data, zi=zi * data[0])
    return filtered_data

# Initialize the filter with the given frequency and parameters
sampling_frequency = 60  # as specified
b, a, zi = initialize_pressure_filter(sampling_freq=sampling_frequency, passband_freq=0.5, order=1, passband_ripple=2)

# Apply filtering to the combined pressure readings
filtered_pressure1 = apply_filter(combined_pressure_reading1, b, a, zi)
filtered_pressure2 = apply_filter(combined_pressure_reading2, b, a, zi)

# Generate plots for this file
file_title = os.path.basename(file_path)

# Pressure plot for sensor 1 (before and after filtering)
plt.figure(figsize=(10, 6))
plt.plot(combined_timestamps, combined_pressure_reading1, 'm-', label='Original S1 Pressure')
plt.plot(combined_timestamps, filtered_pressure1, 'b--', label='Filtered S1 Pressure')
plt.title(f'Pressure Plot for S1 - {file_title}')
plt.xlabel('Timestamp')
plt.ylabel('Pressure (kPa)')
plt.legend()
plt.grid(True)
plt.show()

# Pressure plot for sensor 2 (before and after filtering)
plt.figure(figsize=(10, 6))
plt.plot(combined_timestamps, combined_pressure_reading2, 'c-', label='Original S2 Pressure')
plt.plot(combined_timestamps, filtered_pressure2, 'g--', label='Filtered S2 Pressure')
plt.title(f'Pressure Plot for S2 - {file_title}')
plt.xlabel('Timestamp')
plt.ylabel('Pressure (kPa)')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------Training-------------------------------------------------------

# Prepare data for training
X_S1 = np.array(filtered_pressure1).reshape(-1, 1)  # Only using pressure_reading1 for S1
X_S2 = np.array(filtered_pressure2).reshape(-1, 1)  # Only using pressure_reading2 for S2

y_S1 = np.array(list(zip(combined_sensor_data_S1_x, combined_sensor_data_S1_y, combined_sensor_data_S1_z)))  # Targets for S1
y_S2 = np.array(list(zip(combined_sensor_data_S2_x, combined_sensor_data_S2_y, combined_sensor_data_S2_z)))  # Targets for S2

# Split the data into training and testing sets for both models
X_S1_train, X_S1_test, y_S1_train, y_S1_test = train_test_split(X_S1, y_S1, test_size=0.2, random_state=42)
X_S2_train, X_S2_test, y_S2_train, y_S2_test = train_test_split(X_S2, y_S2, test_size=0.2, random_state=42)

random_seed = 10

mlp_S1 = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=4,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=random_seed,  # for reproducibility
    verbose=True      # enables verbose output
)

mlp_S2 = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=8,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=random_seed,  # for reproducibility
    verbose=True      # enables verbose output
)

print("Training model for Sensor S1...")
mlp_S1.fit(X_S1_train, y_S1_train)

print("Training model for Sensor S2...")
mlp_S2.fit(X_S2_train, y_S2_train)

# Save the trained models
joblib.dump(mlp_S1, 'actuation_sensor_model_S1.pkl')
joblib.dump(mlp_S2, 'actuation_sensor_model_S2.pkl')

print("Models saved to mlp_S1_model.pkl and mlp_S2_model.pkl")

# Make predictions on the test set
y_S1_pred = mlp_S1.predict(X_S1_test)
y_S2_pred = mlp_S2.predict(X_S2_test)

# Evaluate the models using Mean Squared Error (MSE)
mse_S1 = mean_squared_error(y_S1_test, y_S1_pred)
mse_S2 = mean_squared_error(y_S2_test, y_S2_pred)

print(f"Mean Squared Error for S1 Model: {mse_S1}")
print(f"Mean Squared Error for S2 Model: {mse_S2}")

# ----------------------------------------------Prediction Validation-------------------------------------------------------

# Plot predictions vs actual values for Sensor S1 Bx
plt.figure(figsize=(10, 6))
plt.plot(y_S1_test[:, 0], 'b-', label='Actual Bx')  # Blue for Actual Bx
plt.plot(y_S1_pred[:, 0], 'c--', label='Predicted Bx')  # Cyan for Predicted Bx
plt.title('Sensor S1 - Actual vs Predicted Bx')
plt.xlabel('Sample Index')
plt.ylabel('Flux (T)')
plt.legend()
plt.grid(True)
plt.show()

# Plot predictions vs actual values for Sensor S1 By
plt.figure(figsize=(10, 6))
plt.plot(y_S1_test[:, 1], 'g-', label='Actual By')  # Green for Actual By
plt.plot(y_S1_pred[:, 1], 'y--', label='Predicted By')  # Yellow for Predicted By
plt.title('Sensor S1 - Actual vs Predicted By')
plt.xlabel('Sample Index')
plt.ylabel('Flux (T)')
plt.legend()
plt.grid(True)
plt.show()

# Plot predictions vs actual values for Sensor S1 Bz
plt.figure(figsize=(10, 6))
plt.plot(y_S1_test[:, 2], 'r-', label='Actual Bz')  # Red for Actual Bz
plt.plot(y_S1_pred[:, 2], 'm--', label='Predicted Bz')  # Magenta for Predicted Bz
plt.title('Sensor S1 - Actual vs Predicted Bz')
plt.xlabel('Sample Index')
plt.ylabel('Flux (T)')
plt.legend()
plt.grid(True)
plt.show()


# Plot predictions vs actual values for Sensor S2 Bx
plt.figure(figsize=(10, 6))
plt.plot(y_S2_test[:, 0], 'b-', label='Actual Bx')  # Blue for Actual Bx
plt.plot(y_S2_pred[:, 0], 'c--', label='Predicted Bx')  # Cyan for Predicted Bx
plt.title('Sensor S2 - Actual vs Predicted Bx')
plt.xlabel('Sample Index')
plt.ylabel('Flux (T)')
plt.legend()
plt.grid(True)
plt.show()

# Plot predictions vs actual values for Sensor S2 By
plt.figure(figsize=(10, 6))
plt.plot(y_S2_test[:, 1], 'g-', label='Actual By')  # Green for Actual By
plt.plot(y_S2_pred[:, 1], 'y--', label='Predicted By')  # Yellow for Predicted By
plt.title('Sensor S2 - Actual vs Predicted By')
plt.xlabel('Sample Index')
plt.ylabel('Flux (T)')
plt.legend()
plt.grid(True)
plt.show()

# Plot predictions vs actual values for Sensor S2 Bz
plt.figure(figsize=(10, 6))
plt.plot(y_S2_test[:, 2], 'r-', label='Actual Bz')  # Red for Actual Bz
plt.plot(y_S2_pred[:, 2], 'm--', label='Predicted Bz')  # Magenta for Predicted Bz
plt.title('Sensor S2 - Actual vs Predicted Bz')
plt.xlabel('Sample Index')
plt.ylabel('Flux (T)')
plt.legend()
plt.grid(True)
plt.show()



# Making predictions
predictions_S1 = mlp_S1.predict(np.array(filtered_pressure1).reshape(-1, 1))
predictions_S2 = mlp_S2.predict(np.array(filtered_pressure2).reshape(-1, 1))

# Actual sensor data arrays should be provided here
actual_S1_x, actual_S1_y, actual_S1_z = sensor_data_S1_x, sensor_data_S1_y, sensor_data_S1_z
actual_S2_x, actual_S2_y, actual_S2_z = sensor_data_S2_x, sensor_data_S2_y, sensor_data_S2_z

# Define a function to plot data
def plot_sensor_data(predictions, actuals, title_prefix):
    axes = ['Bx', 'By', 'Bz']
    colors = ['b', 'g', 'r']  # Colors for Bx, By, Bz respectively
    for i in range(3):
        plt.figure(figsize=(10, 6))
        plt.plot(predictions[:, i], f'{colors[i]}--', label=f'Predicted {axes[i]}')
        plt.plot(actuals[i], f'{colors[i]}-', label=f'Actual {axes[i]}')
        plt.title(f'{title_prefix} - Actual vs Predicted {axes[i]}')
        plt.xlabel('Sample Index')
        plt.ylabel('Flux (T)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Plot for Sensor S1
plot_sensor_data(predictions_S1, [actual_S1_x, actual_S1_y, actual_S1_z], 'Sensor S1')

# Plot for Sensor S2
plot_sensor_data(predictions_S2, [actual_S2_x, actual_S2_y, actual_S2_z], 'Sensor S2')


debug = 1