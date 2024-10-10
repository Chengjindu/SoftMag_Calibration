import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

# Set column names for normal indentation data and normal flux data
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

# Normalize inputs flux data within (-1, 1)
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

# -----------------------------------Time-step window training on shear data------------------------------------------

# Initialize empty lists to hold the final combined training, validation, and testing data
x_train_shear = []
x_val_shear = []
x_test_shear = []
y_train_shear = []
y_val_shear = []
y_test_shear = []

# Split the shear data into training, validation, and testing sets
x_train_shear, x_val_test_shear, y_train_shear, y_val_test_shear = train_test_split(x_shearflux, y_shearforce, test_size=0.3, random_state=42)
x_val_shear, x_test_shear, y_val_shear, y_test_shear = train_test_split(x_val_test_shear, y_val_test_shear, test_size=0.5, random_state=42)

def create_time_step_window_data(data, labels, time_steps):     # Function to create dataset with time step windows
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(x), np.array(y)

# Define the time steps to evaluate
time_steps_list = [1, 2, 3, 5, 8, 10]

def lr_schedule(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.8
    return lr

# for dynamically reducing the learning rate when the validation loss plateaus, helping to fine-tune the model.
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

# Store results for comparison
results_shear = {}
history_dict_shear = {}

for time_steps in time_steps_list:
    print(f"Training shear model with {time_steps} time steps...")

    # Create time-step window data for each set
    x_train_windowed_shear, y_train_windowed_shear = create_time_step_window_data(x_train_shear, y_train_shear, time_steps)
    x_val_windowed_shear, y_val_windowed_shear = create_time_step_window_data(x_val_shear, y_val_shear, time_steps)
    x_test_windowed_shear, y_test_windowed_shear = create_time_step_window_data(x_test_shear, y_test_shear, time_steps)

    # Reshape the data to flatten the time-step dimension
    x_train_windowed_shear = x_train_windowed_shear.reshape((x_train_windowed_shear.shape[0], -1))
    x_val_windowed_shear = x_val_windowed_shear.reshape((x_val_windowed_shear.shape[0], -1))
    x_test_windowed_shear = x_test_windowed_shear.reshape((x_test_windowed_shear.shape[0], -1))

    # Define and compile the shear force model using Dense layers
    model_shearforce = Sequential([
        Dense(96, activation='relu', kernel_regularizer=l2(1e-4), input_shape=(time_steps * 2,)),
        Dense(48, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(2, activation='linear')  # Predicting Fx and Fy
    ])

    model_shearforce.compile(optimizer=Adam(learning_rate=5e-4), loss='mean_squared_error', metrics=['mae'])

    # Initialize early stopping callback with patience
    early_stopping = EarlyStopping(
        monitor='val_loss',  # The metric to monitor
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,  # To enable verbose output
        restore_best_weights=True, # restore model weights from the epoch with the best value of the monitored metric
        mode='min'
    )

    # Train the shear force model
    history_shearforce = model_shearforce.fit(x_train_windowed_shear, y_train_windowed_shear,
                                              validation_data=(x_val_windowed_shear, y_val_windowed_shear),
                                              epochs=30, batch_size=32,
                                              callbacks=[early_stopping, LearningRateScheduler(lr_schedule, verbose=1)])

    # Save history for plotting
    history_dict_shear[time_steps] = history_shearforce.history

    val_loss_shear, _ = model_shearforce.evaluate(x_val_windowed_shear, y_val_windowed_shear, verbose=1)
    rmse_shear_val = np.sqrt(val_loss_shear)
    print(f"Shear Force Model with {time_steps} time steps - Validation RMSE: {rmse_shear_val}")

    # Save model and validation results
    model_shearforce.save(f'Savings/shearforce_model_{time_steps}_steps.h5')
    results_shear[time_steps] = rmse_shear_val

# Compare results for different time steps and find the best time step
best_time_steps_shear = min(results_shear, key=results_shear.get)
print(f"Best time steps (Shear): {best_time_steps_shear} with RMSE: {results_shear[best_time_steps_shear]}")

# Plot validation loss for each time step window (Shear)
plt.figure(figsize=(10, 8))
for time_steps in time_steps_list:
    plt.plot(history_dict_shear[time_steps]['val_loss'], label=f'{time_steps} steps')
plt.title('Validation Loss for Different Time Step Windows (Shear Data)')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------Evaluation on the Time-step shear models------------------------------------------

# Final evaluation on the test set with the best time steps
print(f"Evaluating the best shear model with {best_time_steps_shear} time steps on the test set...")
x_train_windowed_shear, y_train_windowed_shear = create_time_step_window_data(x_train_shear, y_train_shear, best_time_steps_shear)
x_test_windowed_shear, y_test_windowed_shear = create_time_step_window_data(x_test_shear, y_test_shear, best_time_steps_shear)

# Load the best model
best_model_shear = tf.keras.models.load_model(f'Savings/shearforce_model_{best_time_steps_shear}_steps.h5')

# Evaluate on the test set
test_loss_shear, _ = best_model_shear.evaluate(x_test_windowed_shear, y_test_windowed_shear, verbose=1)
rmse_shear_test = np.sqrt(test_loss_shear)
print(f"Shear Force Model - Test RMSE with {best_time_steps_shear} time steps: {rmse_shear_test}")

# Final model save
best_model_shear.save('Savings/timewindow_shearforce_model.h5')


# -----------------------------------Time-step window training on normal data------------------------------------------

# Split the normal data into training, validation, and testing sets
x_train_normal, x_val_test_normal, y_train_normal, y_val_test_normal = train_test_split(x_normalflux, y_normalforce, test_size=0.3, random_state=42)
x_val_normal, x_test_normal, y_val_normal, y_test_normal = train_test_split(x_val_test_normal, y_val_test_normal, test_size=0.5, random_state=42)

# Store results for comparison
results_normal = {}
history_dict_normal = {}

for time_steps in time_steps_list:
    print(f"Training normal model with {time_steps} time steps...")

    # Create time-step window data for each set
    x_train_windowed_normal, y_train_windowed_normal = create_time_step_window_data(x_train_normal, y_train_normal, time_steps)
    x_val_windowed_normal, y_val_windowed_normal = create_time_step_window_data(x_val_normal, y_val_normal, time_steps)
    x_test_windowed_normal, y_test_windowed_normal = create_time_step_window_data(x_test_normal, y_test_normal, time_steps)

    # Reshape the data to flatten the time-step dimension
    x_train_windowed_normal = x_train_windowed_normal.reshape(x_train_windowed_normal.shape[0], -1)
    x_val_windowed_normal = x_val_windowed_normal.reshape(x_val_windowed_normal.shape[0], -1)
    x_test_windowed_normal = x_test_windowed_normal.reshape(x_test_windowed_normal.shape[0], -1)

    # Define and compile the normal force model using Dense layers
    model_normalforce = Sequential([
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4), input_shape=(time_steps * 3,)),
        # 3 because Bx, By, Bz
        Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(1, activation='linear')  # Predicting Fz
    ])

    model_normalforce.compile(optimizer=Adam(learning_rate=5e-4), loss='mean_squared_error', metrics=['mae'])

    # Initialize early stopping callback with patience
    early_stopping = EarlyStopping(
        monitor='val_loss',  # The metric to monitor
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,  # To enable verbose output
        restore_best_weights=True, # restore model weights from the epoch with the best value of the monitored metric
        mode='min'
    )

    # Train the normal force model
    history_normalforce = model_normalforce.fit(x_train_windowed_normal, y_train_windowed_normal,
                                                validation_data=(x_val_windowed_normal, y_val_windowed_normal),
                                                epochs=25, batch_size=32,
                                                callbacks=[early_stopping,
                                                           LearningRateScheduler(lr_schedule, verbose=1)])

    # Save history for plotting
    history_dict_normal[time_steps] = history_normalforce.history

    val_loss_normal, _ = model_normalforce.evaluate(x_val_windowed_normal, y_val_windowed_normal, verbose=1)
    rmse_normal_val = np.sqrt(val_loss_normal)
    print(f"Normal Force Model with {time_steps} time steps - Validation RMSE: {rmse_normal_val}")

    # Save model and validation results
    model_normalforce.save(f'Savings/normalforce_model_{time_steps}_steps.h5')
    results_normal[time_steps] = rmse_normal_val

# Compare results for different time steps and find the best time step (Normal)
best_time_steps_normal = min(results_normal, key=results_normal.get)
print(f"Best time steps (Normal): {best_time_steps_normal} with RMSE: {results_normal[best_time_steps_normal]}")

# Plot validation loss for each time step window (Normal)
plt.figure(figsize=(10, 8))
for time_steps in time_steps_list:
    plt.plot(history_dict_normal[time_steps]['val_loss'], label=f'{time_steps} steps')
plt.title('Validation Loss for Different Time Step Windows (Normal Data)')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------------Evaluation on the Time-step normal models------------------------------------------

# Final evaluation on the test set with the best time steps
print(f"Evaluating the best normal model with {best_time_steps_normal} time steps on the test set...")
x_train_windowed_normal, y_train_windowed_normal = create_time_step_window_data(x_train_normal, y_train_normal, best_time_steps_normal)
x_test_windowed_normal, y_test_windowed_normal = create_time_step_window_data(x_test_normal, y_test_normal, best_time_steps_normal)

# Reshape the test data
x_test_windowed_normal = x_test_windowed_normal.reshape(x_test_windowed_normal.shape[0], -1)

# Load the best model
best_model_normal = tf.keras.models.load_model(f'Savings/normalforce_model_{best_time_steps_normal}_steps.h5')

# Evaluate on the test set
test_loss_normal, _ = best_model_normal.evaluate(x_test_windowed_normal, y_test_windowed_normal, verbose=1)
rmse_normal_test = np.sqrt(test_loss_normal)
print(f"Normal Force Model - Test RMSE with {best_time_steps_normal} time steps: {rmse_normal_test}")

# Final model save
best_model_normal.save('Savings/timewindow_normalforce_model.h5')




# Debug flag
debugflag = 1
