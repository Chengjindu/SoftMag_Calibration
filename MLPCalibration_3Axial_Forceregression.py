import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import pickle
import os

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def check_cuda():
    if tf.config.list_physical_devices('GPU'):
        print("CUDA is available")
    else:
        print("CUDA is not available")

    print("TensorFlow was built with CUDA:", tf.test.is_built_with_cuda())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Custom callback for dynamic plotting
class PlotLearning(Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x and x not in ['lr']]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


# ------------------------------Shear data import and preprocessing------------------------------------------

# Load the shear datasets from all positions
base_dir = 'Savings/Grippermap/1_G1'
positions = ['1.1', '1.2', '1.3', '2.1', '2.2', '2.3', '3.1', '3.2', '3.3']

# Initialize empty lists to hold data
shear_flux_data_list = []
shear_fx_fy_data_list = []
shear_fz_data_list = []

# Loop through each position and load the corresponding files
for pos in positions:
    shear_flux_path = os.path.join(base_dir, f'Shear_Flux_data_{pos}.csv')
    shear_fx_fy_path = os.path.join(base_dir, f'Shear_Fx_Fy_data_{pos}.csv')
    shear_fz_path = os.path.join(base_dir, f'Shear_Fz_data_{pos}.csv')

    shear_flux_data = pd.read_csv(shear_flux_path)
    shear_fx_fy_data = pd.read_csv(shear_fx_fy_path)
    shear_fz_data = pd.read_csv(shear_fz_path)

    shear_flux_data_list.append(shear_flux_data)
    shear_fx_fy_data_list.append(shear_fx_fy_data)
    shear_fz_data_list.append(shear_fz_data)

# Concatenate all data into single DataFrames
shear_flux_data = pd.concat(shear_flux_data_list, ignore_index=True)
shear_xyforce_data = pd.concat(shear_fx_fy_data_list, ignore_index=True)
shear_normal_force_data = pd.concat(shear_fz_data_list, ignore_index=True)


# Observation
fig, axes = plt.subplots(2, 1, figsize=(15, 15))

# Plot shear flux data
axes[0].plot(shear_flux_data.index, shear_flux_data['Bx'], label='Bx')
axes[0].plot(shear_flux_data.index, shear_flux_data['By'], label='By')
axes[0].plot(shear_flux_data.index, shear_flux_data['Bz'], label='Bz')
axes[0].set_title('Shear Flux Data (Bx, By, Bz)')
axes[0].set_xlabel('Sample Count')
axes[0].set_ylabel('Flux Value')
axes[0].legend()

# Plot shear force data
axes[1].plot(shear_xyforce_data.index, shear_xyforce_data['Fx'], label='Fx')
axes[1].plot(shear_xyforce_data.index, shear_xyforce_data['Fy'], label='Fy')
axes[1].set_title('Shear Force Data (Fx, Fy)')
axes[1].set_xlabel('Sample Count')
axes[1].set_ylabel('Force Value')
axes[1].legend()

plt.tight_layout()
plt.show()


# Normalize inputs flux data within (-1, 1)
scaler_shear_flux = MinMaxScaler(feature_range=(-1, 1))
x_shearflux = scaler_shear_flux.fit_transform(shear_flux_data[['Bx', 'By']])  # Select only Bx and By

# Normalize shear force outputs within (-1, 1)
scaler_shear_force = MinMaxScaler(feature_range=(-1, 1))
y_shearforce = scaler_shear_force.fit_transform(shear_xyforce_data)

# Check the minimum and maximum values used by the scaler
fx_min, fx_max = scaler_shear_force.data_min_[0], scaler_shear_force.data_max_[0]
fy_min, fy_max = scaler_shear_force.data_min_[1], scaler_shear_force.data_max_[1]
print("Fx Min:", fx_min)
print("Fx Max:", fx_max)
print("Fy Min:", fy_min)
print("Fy Max:", fy_max)


# Convert normalized data back to DataFrames for plotting
x_shearflux_df = pd.DataFrame(x_shearflux, columns=['Bx', 'By'])
y_shearforce_df = pd.DataFrame(y_shearforce, columns=['Fx', 'Fy'])


# Observation
fig, axes = plt.subplots(2, 1, figsize=(15, 15))

# Plot normalized shear flux data
axes[0].plot(x_shearflux_df.index, x_shearflux_df['Bx'], label='Bx')
axes[0].plot(x_shearflux_df.index, x_shearflux_df['By'], label='By')
axes[0].set_title('Normalized Shear Flux Data (Bx, By, Bz)')
axes[0].set_xlabel('Sample Count')
axes[0].set_ylabel('Normalized Flux Value')
axes[0].legend()

# Plot normalized shear force data
axes[1].plot(y_shearforce_df.index, y_shearforce_df['Fx'], label='Fx')
axes[1].plot(y_shearforce_df.index, y_shearforce_df['Fy'], label='Fy')
axes[1].set_title('Normalized Shear Force Data (Fx, Fy)')
axes[1].set_xlabel('Sample Count')
axes[1].set_ylabel('Normalized Force Value')
axes[1].legend()

plt.tight_layout()
plt.show()


# ------------------------------Normal data import and preprocessing------------------------------------------

# Load the normal force data with headers
normal_flux_data = pd.read_csv('Savings/Grippermap/1_G1/Normal_Flux_data.csv')
normal_indentation_data = pd.read_csv('Savings/Grippermap/1_G1/Normal_Indentation_data.csv')

# Convert all values to numeric, forcing errors to NaN
normal_flux_data = normal_flux_data.apply(pd.to_numeric, errors='coerce')
normal_indentation_data = normal_indentation_data.apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN values that could not be converted to numeric
normal_flux_data = normal_flux_data.dropna()
normal_indentation_data = normal_indentation_data.dropna()

# Set appropriate column names if needed (assuming columns names are known)
normal_indentation_data.columns = ['Fx', 'Fy', 'Fz']
normal_flux_data.columns = ['Bx', 'By', 'Bz', 'P_Label']

# Get the normal force values
normal_force_data = abs(normal_indentation_data['Fz']).to_frame(name='Fz')


# Observation
fig, axes = plt.subplots(1, 1, figsize=(15, 15))

# Plot normal flux data
axes.plot(normal_flux_data.index, normal_flux_data['Bx'], label='Bx')
axes.plot(normal_flux_data.index, normal_flux_data['By'], label='By')
axes.plot(normal_flux_data.index, normal_flux_data['Bz'], label='Bz')
axes.set_title('Normal Flux Data (Bx, By, Bz)')
axes.set_xlabel('Sample Count')
axes.set_ylabel('Flux Value')
axes.legend()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 1, figsize=(15, 15))

# Plot normal force data
axes.plot(normal_force_data.index, normal_force_data['Fz'], label='Fz')
axes.set_title('Normal Force Data (Fz)')
axes.set_xlabel('Sample Count')
axes.set_ylabel('Force Value')
axes.legend()

plt.tight_layout()
plt.show()


# Combine the datasets for normal force model
# normal_flux_training = pd.concat([normal_flux_data, shear_flux_data], ignore_index=True)
# normal_force_training = pd.concat([normal_force_data, shear_normal_force_data], ignore_index=True)
normal_flux_training = normal_flux_data
normal_force_training = normal_force_data

# Fit on normal data (Bx, By, Bz) without 'P_Label'
scaler_normal_flux = MinMaxScaler(feature_range=(-1, 1))
x_normalflux = scaler_normal_flux.fit_transform(normal_flux_training[['Bx', 'By', 'Bz']])

# Normalize normal force outputs within (0, 1)
scaler_normal_force = MinMaxScaler(feature_range=(0, 1))
normal_force_training = normal_force_training.values.reshape(-1, 1)
y_normalforce = scaler_normal_force.fit_transform(normal_force_training)

# Check the minimum and maximum values used by the scaler
fz_min, fz_max = scaler_normal_force.data_min_[0], scaler_normal_force.data_max_[0]
print("Fz Min:", fz_min)
print("Fz Max:", fz_max)

# Convert normalized data back to DataFrames for plotting
x_normalflux_df = pd.DataFrame(x_normalflux, columns=['Bx', 'By', 'Bz'])
y_normalforce_df = pd.DataFrame(y_normalforce, columns=['Fz'])


# Plotting after normalization
fig, axes = plt.subplots(2, 1, figsize=(15, 15))

# Plot normalized normal flux data
axes[0].plot(x_normalflux_df.index, x_normalflux_df['Bx'], label='Bx')
axes[0].plot(x_normalflux_df.index, x_normalflux_df['By'], label='By')
axes[0].plot(x_normalflux_df.index, x_normalflux_df['Bz'], label='Bz')
axes[0].set_title('Normalized Normal Flux Data (Bx, By, Bz)')
axes[0].set_xlabel('Sample Count')
axes[0].set_ylabel('Normalized Flux Value')
axes[0].legend()

# Plot normalized normal force data
axes[1].plot(y_normalforce_df.index, y_normalforce_df['Fz'], label='Fz')
axes[1].set_title('Normalized Normal Force Data (Fz)')
axes[1].set_xlabel('Sample Count')
axes[1].set_ylabel('Normalized Force Value')
axes[1].legend()

plt.tight_layout()
plt.show()


# Save normalization parameters
normalization_force_params = {
    'normal_flux': {
        'Bx_min': normal_flux_data.min()[0],
        'Bx_max': normal_flux_data.max()[0],
        'By_min': normal_flux_data.min()[1],
        'By_max': normal_flux_data.max()[1],
        'Bz_min': normal_flux_data.min()[2],
        'Bz_max': normal_flux_data.max()[2]
    },
    'shear_flux': {
        'Bx_min': shear_flux_data.min()[0],
        'Bx_max': shear_flux_data.max()[0],
        'By_min': shear_flux_data.min()[1],
        'By_max': shear_flux_data.max()[1],
    },
    'force': {
        'Fx_min': fx_min,
        'Fx_max': fx_max,
        'Fy_min': fy_min,
        'Fy_max': fy_max,
        'Fz_min': fz_min,
        'Fz_max': fz_max
    }
}

# Save to a pickle file
with open('Savings/normalization_force_params.pkl', 'wb') as file:
    pickle.dump(normalization_force_params, file)


# ------------------------------Pressure-free data import and preprocessing------------------------------------------

# Load the data from the text file
pressure_free_data_path = 'Savings/Grippermap/1_G1/Pressure_free_data.txt'

# Read the file using the method we discussed earlier
column_names = [
    'Time [ms]', 'Displacement [mm]', 'Force X [N]', 'Force Y [N]', 'Force Z [N]',
    'Torque X [Nm]', 'Torque Y [Nm]', 'Torque Z [Nm]', 'Bx [Gauss]', 'By [Gauss]', 'Bz [Gauss]'
]

# Load the data, skipping the first two lines and applying the correct column names
pressure_free_data = pd.read_csv(pressure_free_data_path, sep="\t", skiprows=2, names=column_names)

# Convert all values to numeric, forcing errors to NaN, Drop rows with any NaN values
pressure_free_data = pressure_free_data.apply(pd.to_numeric, errors='coerce')
pressure_free_data = pressure_free_data.dropna()

# Ensure column names are stripped of any extra spaces
pressure_free_data.columns = pressure_free_data.columns.str.strip()
print("Columns in pressure_free_data after stripping spaces:", pressure_free_data.columns)

# Rename the columns temporarily to match those used during fitting
pressure_free_data_renamed = pressure_free_data.rename(columns={
    'Bx [Gauss]': 'Bx',
    'By [Gauss]': 'By',
    'Bz [Gauss]': 'Bz',
    'Force X [N]': 'Fx',
    'Force Y [N]': 'Fy',
    'Force Z [N]': 'Fz'
})

# Select relevant columns for shear force model from the renamed DataFrame
shear_flux_data_pressure_free = pressure_free_data_renamed[['Bx', 'By']]
shear_force_data_pressure_free = pressure_free_data_renamed[['Fx', 'Fy']]

# Normalize the data using the existing scalers
x_shearflux_pressure_free = scaler_shear_flux.transform(shear_flux_data_pressure_free)
y_shearforce_pressure_free = scaler_shear_force.transform(shear_force_data_pressure_free)

# Select relevant columns for normal force model
normal_flux_data_pressure_free = pressure_free_data_renamed[['Bx', 'By', 'Bz']]
normal_force_data_pressure_free = abs(pressure_free_data_renamed[['Fz']])

# Ensure the pressure-free data has the same columns (Bx, By, Bz)
x_normalflux_pressure_free = scaler_normal_flux.transform(normal_flux_data_pressure_free[['Bx', 'By', 'Bz']])
y_normalforce_pressure_free = scaler_normal_force.transform(normal_force_data_pressure_free)


# -----------------------------------Shear force model Training------------------------------------------

# Initialize empty lists to hold the final combined shear training and testing data
x_train_shear_combined = []
x_test_shear_combined = []
y_train_shear_combined = []
y_test_shear_combined = []

# Split the shear data into training and testing sets
x_train_shear, x_test_shear, y_train_shear, y_test_shear = train_test_split(x_shearflux, y_shearforce, test_size=0.2, random_state=42)

# Split the shear pressure free data into training and testing sets
x_train_shear_pressure_free, x_test_shear_pressure_free, y_train_shear_pressure_free, y_test_shear_pressure_free = (
    train_test_split(x_shearflux_pressure_free, y_shearforce_pressure_free, test_size=0.2, random_state=42))

# Append the pressure-free data to the shear model datasets
x_train_shear_combined = np.concatenate([x_train_shear, x_train_shear_pressure_free], axis=0)
y_train_shear_combined = np.concatenate([y_train_shear, y_train_shear_pressure_free], axis=0)
x_test_shear_combined = np.concatenate([x_test_shear, x_test_shear_pressure_free], axis=0)
y_test_shear_combined = np.concatenate([y_test_shear, y_test_shear_pressure_free], axis=0)

check_cuda()

# model_shearforce = Sequential([     # Define and compile the shear force model
#     Dense(96, activation='relu', kernel_regularizer=l2(1e-4), input_shape=(2,)),
#     # Dropout(0.5),
#     Dense(48, activation='relu', kernel_regularizer=l2(1e-4)),
#     # Dropout(0.5),
#     Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
#     Dense(2, activation='linear')  # Predicting Fx and Fy
# ])
#
# model_shearforce.compile(optimizer=Adam(learning_rate=5e-4), loss='mean_squared_error', metrics=['mae'])
#
# # Initialize early stopping callback with patience
# early_stopping = EarlyStopping(
#     monitor='val_loss',  # The metric to monitor
#     patience=5,          # Number of epochs with no improvement after which training will be stopped
#     verbose=1,           # To enable verbose output
#     restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric
#     mode='min'
# )
#
def lr_schedule(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.8
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)        # Dynamically reducing the learning rate
#
# print("Training shear force model...")
# history_shearforce = model_shearforce.fit(x_train_shear, y_train_shear, validation_split=0.2,
#                                           epochs=30, batch_size=32,
#                                           callbacks=[early_stopping, lr_scheduler])
#
# model_shearforce.save('Savings/shearforce_model.h5')
#
# loss_shear, _ = model_shearforce.evaluate(x_test_shear, y_test_shear, verbose=1)
# rmse_shear = np.sqrt(loss_shear)
# print(f"Shear Force Model - RMSE: {rmse_shear}")


# -----------------------------------Normal force model Training------------------------------------------

x_train_normal_combined = []
y_train_normal_combined = []
x_test_normal_combined = []
y_test_normal_combined = []

# Split the normal data into training and testing sets
x_train_normal, x_test_normal, y_train_normal, y_test_normal = train_test_split(x_normalflux, y_normalforce, test_size=0.2, random_state=42)

# Split the normal pressure free data into training and testing sets
x_train_normal_pressure_free, x_test_normal_pressure_free, y_train_normal_pressure_free, y_test_normal_pressure_free = (
    train_test_split(x_normalflux_pressure_free, y_normalforce_pressure_free, test_size=0.2, random_state=42))

# Append the normal pressure-free data to the combined normal model datasets
x_train_normal_combined = np.concatenate([x_train_normal, x_train_normal_pressure_free], axis=0)
y_train_normal_combined = np.concatenate([y_train_normal, y_train_normal_pressure_free], axis=0)
x_test_normal_combined = np.concatenate([x_test_normal, x_test_normal_pressure_free], axis=0)
y_test_normal_combined = np.concatenate([y_test_normal, y_test_normal_pressure_free], axis=0)


model_normalforce = Sequential([        # Define and compile the normal force model
    Dense(64, activation='relu', kernel_regularizer=l2(1e-4), input_shape=(x_normalflux.shape[1],)),
    # Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
    # Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(1, activation='linear')  # For regression, output is linear
])

model_normalforce.compile(optimizer=Adam(learning_rate=5e-4), loss='mean_squared_error', metrics=['mae'])  # 'mae' for mean absolute error

# Initialize early stopping callback with patience
early_stopping = EarlyStopping(
    monitor='val_loss',  # The metric to monitor
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # To enable verbose output
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric
    mode='min'
)

print("Training normal force model...")
history_normalforce = model_normalforce.fit(x_train_normal, y_train_normal, validation_split=0.2,
                                            epochs=30, batch_size=32,
                                            callbacks=[early_stopping, lr_scheduler])

# Save the trained models as h5 files
model_normalforce.save('Savings/normalforce_model.h5')

# Evaluate the regression models
loss_normal, _ = model_normalforce.evaluate(x_test_normal, y_test_normal, verbose=1)
rmse_normal = np.sqrt(loss_normal)
print(f"Normal Force Model - RMSE: {rmse_normal}")



# Debug flag
debugflag = 1
