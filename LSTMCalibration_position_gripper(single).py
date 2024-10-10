import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import pickle

# Check for GPU and CUDA availability
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

# ---------------------------------Data importing and preprocessing--------------------------------------

# Load the normal force data with headers
normal_flux_data = pd.read_csv('Savings/Grippermap/1_G1/Normal_Flux_data.csv')
normal_flux_data.columns = ['Bx', 'By', 'Bz', 'P_Label']

# Separate the P_Label into x and y coordinates
normal_flux_data['x'] = normal_flux_data['P_Label'].apply(lambda p: int(str(p).split('.')[0]))
normal_flux_data['y'] = normal_flux_data['P_Label'].apply(lambda p: int(str(p).split('.')[1]))

# Define the new position labels based on the 2x2 grid
def assign_position_label(x, y):
    if x > 5 and y < 5:
        return 0  # P1
    elif x < 5 and y < 5:
        return 1  # P2
    elif x < 5 and y > 5:
        return 2  # P3
    elif x > 5 and y > 5:
        return 3  # P4
    else:
        return 4

normal_flux_data['Position_2x2'] = normal_flux_data.apply(lambda row: assign_position_label(row['x'], row['y']), axis=1)

# # Plotting the raw data distribution
# fig, axes = plt.subplots(2, 1, figsize=(15, 10))
# axes[0].plot(normal_flux_data.index, normal_flux_data['Bx'], label='Bx')
# axes[0].plot(normal_flux_data.index, normal_flux_data['By'], label='By')
# axes[0].plot(normal_flux_data.index, normal_flux_data['Bz'], label='Bz')
# axes[0].set_title('Raw Flux Data (Bx, By, Bz)')
# axes[0].set_xlabel('Sample Index')
# axes[0].set_ylabel('Flux Value')
# axes[0].legend()
#
# axes[1].hist(normal_flux_data['Position_2x2'], bins=4, edgecolor='k')
# axes[1].set_title('Distribution of 2x2 Position Labels')
# axes[1].set_xlabel('Position')
# axes[1].set_ylabel('Frequency')
#
# plt.tight_layout()
# plt.show()

# Save normalization parameters for 2x2 position
normalization_position_params = {
    'Bx_min': normal_flux_data['Bx'].min(),
    'Bx_max': normal_flux_data['Bx'].max(),
    'By_min': normal_flux_data['By'].min(),
    'By_max': normal_flux_data['By'].max(),
    'Bz_min': normal_flux_data['Bz'].min(),
    'Bz_max': normal_flux_data['Bz'].max()
}

# Save to a pickle file
with open('Savings/normalization_position_params.pkl', 'wb') as file:
    pickle.dump(normalization_position_params, file)

# Filter out the rows with Position_2x2 label as 4
filtered_data = normal_flux_data[normal_flux_data['Position_2x2'] != 4].reset_index(drop=True)

# Normalize the flux data to the range (-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_filtered_flux_data = scaler.fit_transform(filtered_data[['Bx', 'By', 'Bz']])


# Plotting the normalized data distribution
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(normal_flux_data.index, normalized_filtered_flux_data[:, 0], label='Bx')
ax.plot(normal_flux_data.index, normalized_filtered_flux_data[:, 1], label='By')
ax.plot(normal_flux_data.index, normalized_filtered_flux_data[:, 2], label='Bz')
ax.set_title('Normalized Flux Data (Bx, By, Bz)')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Normalized Flux Value')
ax.legend()

plt.tight_layout()
plt.show()


# ---------------------------------One-hot encoding and training preparation--------------------------------------

# One-Hot Encode the 2x2 positions
y_lstm_onehot_2x2 = to_categorical(filtered_data['Position_2x2'], num_classes=5)

# One-Hot Encode the 9x9 x and y coordinates
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
x_onehot_9x9 = onehot_encoder.fit_transform(normal_flux_data[['x']])
y_onehot_9x9 = onehot_encoder.fit_transform(normal_flux_data[['y']])

# Combine the one-hot encoded x and y into a single target array for 9x9 positions
y_coordinates_onehot_9x9 = np.concatenate([x_onehot_9x9, y_onehot_9x9], axis=1)

x_flux_values = normalized_filtered_flux_data  # # Prepare the input features

# ---------------------------------Sequence Preparation with Sliding Window Method-----------------------------------

def create_time_step_window_data(X, y, time_steps):     # Function to create time-step window data
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10  # Number of time steps to look back

# Prepare LSTM input data for both 2x2 and 9x9 position models
x_lstm_2x2, y_lstm_2x2 = create_time_step_window_data(x_flux_values, y_lstm_onehot_2x2, time_steps)

# Split the data into training and testing sets for 2x2 model
x_train_lstm_2x2, x_test_lstm_2x2, y_train_lstm_2x2, y_test_lstm_2x2 = train_test_split(
    x_lstm_2x2, y_lstm_2x2, test_size=0.2, random_state=42
)

x_lstm_9x9, y_lstm_9x9 = create_time_step_window_data(x_flux_values, y_coordinates_onehot_9x9, time_steps)

# Split the data into training and testing sets for 9x9 model
x_train_lstm_9x9, x_test_lstm_9x9, y_train_lstm_9x9, y_test_lstm_9x9 = train_test_split(
    x_lstm_9x9, y_lstm_9x9, test_size=0.2, random_state=42
)

# Plotting the first few 2*2 sequences and comparing with the corresponding original data array
num_sequences_to_plot = 5  # Number of sequences to plot
plt.figure(figsize=(15, 10))

for i in range(num_sequences_to_plot):
    plt.subplot(num_sequences_to_plot, 1, i + 1)

    # Plot the corresponding original data for the current sequence
    start_idx = i
    end_idx = start_idx + time_steps

    plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 0], 'o-',
             label='Bx - Original')
    plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 1], 'o-',
             label='By - Original')
    plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 2], 'o-',
             label='Bz - Original')

    # Plot the sequence data
    plt.plot(range(start_idx, end_idx), x_lstm_2x2[i, :, 0], 'x--', label='Bx - Sequence')
    plt.plot(range(start_idx, end_idx), x_lstm_2x2[i, :, 1], 'x--', label='By - Sequence')
    plt.plot(range(start_idx, end_idx), x_lstm_2x2[i, :, 2], 'x--', label='Bz - Sequence')

    plt.title(f'Sequence {i + 1}')
    plt.xlabel('Time Step')
    plt.ylabel('Flux Value')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))

for i in range(num_sequences_to_plot):
    plt.subplot(num_sequences_to_plot, 1, i + 1)

    # Plot the corresponding original data for the current sequence
    start_idx = i
    end_idx = start_idx + time_steps

    plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 0], 'o-',
             label='Bx - Original')
    plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 1], 'o-',
             label='By - Original')
    plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 2], 'o-',
             label='Bz - Original')

    # Plot the sequence data
    plt.plot(range(start_idx, end_idx), x_lstm_9x9[i, :, 0], 'x--', label='Bx - Sequence')
    plt.plot(range(start_idx, end_idx), x_lstm_9x9[i, :, 1], 'x--', label='By - Sequence')
    plt.plot(range(start_idx, end_idx), x_lstm_9x9[i, :, 2], 'x--', label='Bz - Sequence')

    plt.title(f'Sequence {i + 1}')
    plt.xlabel('Time Step')
    plt.ylabel('Flux Value')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------Sequence Preparation with Non-Overlapping Method--------------------------------------

# # Number of timesteps in each sample
# time_steps = 10
#
# # Prepare the input features
# x_flux_values = normalized_flux_data
#
# # Calculate the number of sequences that can be created with non-overlapping windows
# n_sequences = len(x_flux_values) // time_steps
#
# # Reshape the data into a 3D array of shape [n_sequences, time_steps, n_features]
# x_lstm_2x2 = x_flux_values[:n_sequences * time_steps].reshape((n_sequences, time_steps, x_flux_values.shape[1]))
# x_lstm_9x9 = x_flux_values[:n_sequences * time_steps].reshape((n_sequences, time_steps, x_flux_values.shape[1]))
#
# # Prepare the corresponding labels for the sequences
# y_lstm_onehot_2x2_trimmed = y_lstm_onehot_2x2[:n_sequences * time_steps:time_steps]
# y_coordinates_onehot_9x9_trimmed = y_coordinates_onehot_9x9[:n_sequences * time_steps:time_steps]
#
# # Split the data into training and testing sets for 2x2 model
# x_train_lstm_2x2, x_test_lstm_2x2, y_train_lstm_2x2, y_test_lstm_2x2 = train_test_split(
#     x_lstm_2x2, y_lstm_onehot_2x2_trimmed, test_size=0.2, random_state=42
# )
#
# # Split the data into training and testing sets for 9x9 model
# x_train_lstm_9x9, x_test_lstm_9x9, y_train_lstm_9x9, y_test_lstm_9x9 = train_test_split(
#     x_lstm_9x9, y_coordinates_onehot_9x9_trimmed, test_size=0.2, random_state=42
# )
#
#
# num_sequences_to_plot = 5  # Number of sequences to plot
# plt.figure(figsize=(15, 10))
#
# for i in range(num_sequences_to_plot):
#     plt.subplot(num_sequences_to_plot, 1, i + 1)
#
#     # Plot the corresponding original data for the current sequence
#     start_idx = i * time_steps
#     end_idx = start_idx + time_steps
#
#     plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 0], 'o-',
#              label='Bx - Original')
#     plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 1], 'o-',
#              label='By - Original')
#     plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 2], 'o-',
#              label='Bz - Original')
#
#     # Plot the sequence data for 2x2 model
#     plt.plot(range(start_idx, end_idx), x_lstm_2x2[i, :, 0], 'x--', label='Bx - Sequence 2x2')
#     plt.plot(range(start_idx, end_idx), x_lstm_2x2[i, :, 1], 'x--', label='By - Sequence 2x2')
#     plt.plot(range(start_idx, end_idx), x_lstm_2x2[i, :, 2], 'x--', label='Bz - Sequence 2x2')
#
#     plt.title(f'Sequence {i + 1}')
#     plt.xlabel('Time Step')
#     plt.ylabel('Flux Value')
#     plt.legend()
#     plt.grid(True)
#
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(15, 10))
#
# for i in range(num_sequences_to_plot):
#     plt.subplot(num_sequences_to_plot, 1, i + 1)
#
#     # Plot the corresponding original data for the current sequence
#     start_idx = i * time_steps
#     end_idx = start_idx + time_steps
#
#     plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 0], 'o-',
#              label='Bx - Original')
#     plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 1], 'o-',
#              label='By - Original')
#     plt.plot(range(start_idx, end_idx), x_flux_values[start_idx:end_idx, 2], 'o-',
#              label='Bz - Original')
#
#     # Plot the sequence data for 9x9 model
#     plt.plot(range(start_idx, end_idx), x_lstm_9x9[i, :, 0], 'x--', label='Bx - Sequence 9x9')
#     plt.plot(range(start_idx, end_idx), x_lstm_9x9[i, :, 1], 'x--', label='By - Sequence 9x9')
#     plt.plot(range(start_idx, end_idx), x_lstm_9x9[i, :, 2], 'x--', label='Bz - Sequence 9x9')
#
#     plt.title(f'Sequence {i + 1}')
#     plt.xlabel('Time Step')
#     plt.ylabel('Flux Value')
#     plt.legend()
#     plt.grid(True)
#
# plt.tight_layout()
# plt.show()


# ---------------------------------Training 2*2--------------------------------------

check_cuda()

# Define the LSTM-based neural network architecture for the 2x2 model
model_lstm_2x2 = Sequential([
    LSTM(96, input_shape=(time_steps, 3), return_sequences=False),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  # Output layer with 4 outputs (P1, P2, P3, P4)
])

model_lstm_2x2.compile(optimizer=Adam(learning_rate=1e-3),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Early stopping callbacks
early_stopping_2x2 = EarlyStopping(
    monitor='val_loss',  # The metric to monitor
    patience=8,          # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # To enable verbose output
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric
    mode='min'
)

# Train the LSTM models
history_2x2 = model_lstm_2x2.fit(x_train_lstm_2x2, y_train_lstm_2x2,
                         validation_data=(x_test_lstm_2x2, y_test_lstm_2x2),
                         epochs=200,
                         batch_size=64,
                         shuffle=True,
                         callbacks=[early_stopping_2x2],
                         verbose=1)

# Evaluate the LSTM models
loss_2x2, accuracy_2x2 = model_lstm_2x2.evaluate(x_test_lstm_2x2, y_test_lstm_2x2, verbose=1)
print(f"2x2 Model - Test Loss: {loss_2x2}")
print(f"2x2 Model - Test Accuracy: {accuracy_2x2}")

# Save the trained LSTM models as h5 files
model_lstm_2x2.save('Savings/lstm_2x2_model.h5')

# ---------------------------------Training 9*9--------------------------------------

# Define the LSTM-based neural network architecture for the 9x9 model
model_lstm_9x9 = Sequential([
    LSTM(128, input_shape=(time_steps, 3), return_sequences=False),
    Dense(96, activation='relu'),
    Dense(18, activation='softmax')  # Output layer with 18 outputs (9 for x, 9 for y)
])

model_lstm_9x9.compile(optimizer=Adam(learning_rate=1e-3),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

early_stopping_9x9 = EarlyStopping(
    monitor='val_loss',  # The metric to monitor
    patience=8,          # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # To enable verbose output
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric
    mode='min'
)

history_9x9 = model_lstm_9x9.fit(x_train_lstm_9x9, y_train_lstm_9x9,
                         validation_data=(x_test_lstm_9x9, y_test_lstm_9x9),
                         epochs=50,
                         batch_size=64,
                         shuffle=True,
                         callbacks=[early_stopping_9x9, PlotLearning()],
                         verbose=1)

loss_9x9, accuracy_9x9 = model_lstm_9x9.evaluate(x_test_lstm_9x9, y_test_lstm_9x9, verbose=1)
print(f"9x9 Model - Test Loss: {loss_9x9}")
print(f"9x9 Model - Test Accuracy: {accuracy_9x9}")

# Save the trained LSTM models as h5 files
model_lstm_9x9.save('Savings/lstm_9x9_model.h5')

# Debug flag
test = 1
