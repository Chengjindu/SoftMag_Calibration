import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import pickle

def check_cuda():
    if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        print("CUDA is available")
    else:
        print("CUDA is not available")
    if tf.test.is_built_with_cuda():
        print("TensorFlow was built with CUDA")
    else:
        print("TensorFlow was not built with CUDA")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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


# Load datasets and assign labels
folder_path = 'Data_onehotposition'
positions = ['P1.txt', 'P2.txt', 'P3.txt', 'P4.txt']
dataframes = []

for i, position in enumerate(positions, start=1):
    file_path = os.path.join(folder_path, position)
    # Read the file, assuming there might be a header and whitespace to ignore
    df = pd.read_csv(file_path, sep=r"\s*,\s*", header=0, engine='python')  # regex separator to handle whitespace
    df['Position'] = i  # Assigning label based on file order
    dataframes.append(df)

# Combine all data into a single DataFrame
training_data = pd.concat(dataframes, ignore_index=True)

# Splitting data into features (X) and labels (y) for classification
x_class = training_data.iloc[:, :3]  # Assume first 3 columns are Bx, By, Bz
y_class = training_data['Position'] - 1  # Subtract 1 to have labels start from 0


# Plotting the raw data distribution
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
axes[0].plot(x_class.index, x_class['Bx'], label='Bx')
axes[0].plot(x_class.index, x_class['By'], label='By')
axes[0].plot(x_class.index, x_class['Bz'], label='Bz')
axes[0].set_title('Raw Flux Data (Bx, By, Bz)')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Flux Value')
axes[0].legend()

axes[1].hist(y_class, bins=4, edgecolor='k')
axes[1].set_title('Distribution of Position Labels')
axes[1].set_xlabel('Position')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


with open('Savings/normalization_position_params.pkl', 'rb') as f:
    normalization_params = pickle.load(f)

# Check if normalization_params contains the expected keys
expected_keys = ['Bx_max', 'Bx_min', 'By_max', 'By_min', 'Bz_max', 'Bz_min']
if not all(key in normalization_params for key in expected_keys):
    raise ValueError("Normalization parameters are missing some expected keys.")

# Assuming 'normalization_params' is a dictionary that contains 'Bx_min', 'Bx_max', 'By_min', 'By_max', 'Bz_min', 'Bz_max'
Bx_min = normalization_params['Bx_min']
Bx_max = normalization_params['Bx_max']
By_min = normalization_params['By_min']
By_max = normalization_params['By_max']
Bz_min = normalization_params['Bz_min']
Bz_max = normalization_params['Bz_max']

# Function to normalize data
def normalize_feature(data, min_value, max_value):
    return 2 * ((data - min_value) / (max_value - min_value)) - 1

# Normalize your features
x_class['Bx'] = normalize_feature(x_class['Bx'], normalization_params['Bx_min'], normalization_params['Bx_max'])  # Normalizing Bx
x_class['By'] = normalize_feature(x_class['By'], normalization_params['By_min'], normalization_params['By_max'])  # Normalizing By
x_class['Bz'] = normalize_feature(x_class['Bz'], normalization_params['Bz_min'], normalization_params['Bz_max'])  # Normalizing Bz


# Plotting the normalized data distribution
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(x_class.index, x_class['Bx'], label='Bx')
ax.plot(x_class.index, x_class['By'], label='By')
ax.plot(x_class.index, x_class['Bz'], label='Bz')
ax.set_title('Normalized Flux Data (Bx, By, Bz)')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Normalized Flux Value')
ax.legend()

plt.tight_layout()
plt.show()

# -----------------------------------Direct Non-overlap Reshaping-------------------------------------

# # Prepared sequential data
# n_steps = 10  # number of timesteps in each sample
# n_features = 3
#
# data_array = x_class.values # Convert the DataFrame into a numpy array
#
# # Calculate the number of sequences that can be created given the sequence length
# n_sequences = len(data_array) // n_steps
#
# # Reshape the data into a 3D array of shape [n_sequences, n_steps, n_features]
# x_lstm = data_array[:n_sequences * n_steps].reshape((n_sequences, n_steps, n_features))
#
# # Plotting the first few sequences and comparing with the original data array
# num_sequences_to_plot = 3  # Number of sequences to plot
# plt.figure(figsize=(15, 10))
#
# for i in range(num_sequences_to_plot):
#     plt.subplot(num_sequences_to_plot, 1, i + 1)
#     plt.plot(range(i * n_steps, (i + 1) * n_steps), data_array[i * n_steps:(i + 1) * n_steps, 0], 'o-',
#              label='Bx - Original')
#     plt.plot(range(i * n_steps, (i + 1) * n_steps), data_array[i * n_steps:(i + 1) * n_steps, 1], 'o-',
#              label='By - Original')
#     plt.plot(range(i * n_steps, (i + 1) * n_steps), data_array[i * n_steps:(i + 1) * n_steps, 2], 'o-',
#              label='Bz - Original')
#
#     plt.plot(range(i * n_steps, (i + 1) * n_steps), x_lstm[i, :, 0], 'x--', label='Bx - Sequence')
#     plt.plot(range(i * n_steps, (i + 1) * n_steps), x_lstm[i, :, 1], 'x--', label='By - Sequence')
#     plt.plot(range(i * n_steps, (i + 1) * n_steps), x_lstm[i, :, 2], 'x--', label='Bz - Sequence')
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
# # Ensure y_class is trimmed to match the number of sequences in x_lstm
# # This is done by taking the first label of each sequence
# y_sequences = y_class.values[:n_sequences * n_steps: n_steps]
# y_class_onehot = to_categorical(y_sequences, num_classes=4)
#
# # Now both x_lstm and y_class_onehot should have the same number of sequences
# assert len(x_lstm) == len(y_class_onehot), "The number of samples should match."

# -----------------------------------Sliding Window Reshaping-------------------------------------

# Function to create time-step window data with overlapping sequences
def create_time_step_window_data(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

n_steps = 10  # Number of timesteps in each sample
n_features = 3

# Create LSTM input data using sliding window method
x_lstm, y_lstm = create_time_step_window_data(x_class.values, y_class.values, n_steps)

# Convert y_lstm to one-hot encoding
y_class_onehot = to_categorical(y_lstm, num_classes=4)

# Plotting the first few sequences and comparing with the original data array
num_sequences_to_plot = 3  # Number of sequences to plot
plt.figure(figsize=(15, 10))

for i in range(num_sequences_to_plot):
    plt.subplot(num_sequences_to_plot, 1, i + 1)
    start_idx = i  # Start at the ith sequence
    end_idx = start_idx + n_steps
    plt.plot(range(n_steps), x_class.values[start_idx:end_idx, 0], 'o-',
             label='Bx - Original')
    plt.plot(range(n_steps), x_class.values[start_idx:end_idx, 1], 'o-',
             label='By - Original')
    plt.plot(range(n_steps), x_class.values[start_idx:end_idx, 2], 'o-',
             label='Bz - Original')

    plt.plot(range(n_steps), x_lstm[i, :, 0], 'x--', label='Bx - Sequence')
    plt.plot(range(n_steps), x_lstm[i, :, 1], 'x--', label='By - Sequence')
    plt.plot(range(n_steps), x_lstm[i, :, 2], 'x--', label='Bz - Sequence')

    plt.title(f'Sequence {i + 1}')
    plt.xlabel('Time Step')
    plt.ylabel('Flux Value')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Split the data into training and testing sets
x_train_lstm, x_test_lstm, y_train_lstm, y_test_lstm = train_test_split(x_lstm, y_class_onehot, test_size=0.2)

check_cuda()
model_lstm = Sequential([
    LSTM(64, activation='tanh', recurrent_activation='sigmoid',
         # dropout=0.25, recurrent_dropout=0.25,
         kernel_regularizer=l2(0.0001),  # L2 regularization on LSTM layer
         input_shape=(n_steps, n_features)),
    # Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),  # L2 regularization on Dense layer
    # Dropout(0.5),
    Dense(4, activation='softmax')  # Assuming 4 classes for the single-touch classification, with L2 regularization
])

model_lstm.compile(optimizer=Adam(learning_rate=1e-3),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

history = model_lstm.fit(x_train_lstm, y_train_lstm,
                         validation_data=(x_test_lstm, y_test_lstm),
                         epochs=200,
                         batch_size=32,
                         callbacks=[early_stopping],
                         verbose=1)

# def convert_tflite():
#     # Convert the trained LSTM model to TensorFlow Lite format
#     lstm_converter = tf.lite.TFLiteConverter.from_keras_model(model_lstm)
#
#     # Enable the use of select TensorFlow ops
#     lstm_converter.target_spec.supported_ops = [
#         tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
#         tf.lite.OpsSet.SELECT_TF_OPS  # Enable select TensorFlow ops.
#     ]
#
#     # Disable the experimental_lower_tensor_list_ops
#     lstm_converter._experimental_lower_tensor_list_ops = False
#
#     # Apply optimizations to the model to support the default set of ops
#     lstm_converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
#     # Convert the model
#     tflite_lstm_model = lstm_converter.convert()
#
#     # Save the TensorFlow Lite model
#     tflite_model_path = '/home/chengjin/Projects/SoftMag/Savings/lstm_classification_model.tflite'
#     with open(tflite_model_path, 'wb') as f:
#         f.write(tflite_lstm_model)
#
#     # Save the Keras model
#     keras_model_path = '/home/chengjin/Projects/SoftMag/Savings/lstm_classification_model.h5'
#     model_lstm.save(keras_model_path)
#
#     print(f"LSTM Model saved to {keras_model_path}")
#     # print(f"LSTM Model saved to {tflite_model_path} and {keras_model_path}")


# Debug flag

# Save the trained lstm model as h5 files
model_lstm.save('Savings/lstm_classification_model.h5')

test = 1