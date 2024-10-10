import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

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

def get_labels_from_filename(filename):
    # This function will need to parse the filename to determine which positions are touched
    # and return a binary vector representing the multitouch state
    labels = [0, 0, 0, 0]
    if 'P1' in filename: labels[0] = 1
    if 'P2' in filename: labels[1] = 1
    if 'P3' in filename: labels[2] = 1
    if 'P4' in filename: labels[3] = 1
    if 'P12' in filename: labels[0] = 1; labels[1] = 1
    if 'P14' in filename: labels[0] = 1; labels[3] = 1
    if 'P23' in filename: labels[1] = 1; labels[2] = 1
    if 'P34' in filename: labels[2] = 1; labels[3] = 1
    if 'P1234' in filename: labels[0] = 1; labels[1] = 1; labels[2] = 1; labels[3] = 1

    return labels

# Load datasets and assign labels
folder_path = 'Data_onehotposition'
filenames = ['P1.txt', 'P2.txt', 'P3.txt', 'P4.txt', 'P12.txt', 'P14.txt', 'P23.txt', 'P34.txt', 'P1234.txt']
dataframes = [] # Empty list for storing flux data
labels_list = [] # Empty list for storing labels

for i, filename in enumerate(filenames, start=1):
    file_path = os.path.join(folder_path, filename)
    # Read the file, assuming there might be a header and whitespace to ignore
    df = pd.read_csv(file_path, sep=r"\s*,\s*", header=0, engine='python')  # regex separator to handle whitespace
    labels = get_labels_from_filename(filename)
    # Replicate labels for the length of the dataframe
    replicated_labels = [labels for _ in range(len(df))]
    dataframes.append(df)
    labels_list.extend(replicated_labels)  # Extend, not append, to flatten the list

# Combine all data into a single DataFrame
training_data = pd.concat(dataframes, ignore_index=True)


x_class = training_data.iloc[:, :3]  # Assume first 3 columns are Bx, By, Bz
y_multitouch = np.array(labels_list)

with open('Savings/normalization_position_params.pkl', 'rb') as f:
    normalization_params = pickle.load(f)

# Check if normalization_params contains the expected keys
expected_keys = ['Bx_min', 'Bx_max', 'By_min', 'By_max', 'Bz_min', 'Bz_max']
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

# Prepared sequential data
n_steps = 10  # number of timesteps in each sample
n_features = 3
n_outputs = 4

# Convert the DataFrame into a numpy array
data_array = x_class.values

# Calculate the number of sequences that can be created given the sequence length
n_sequences = len(data_array) // n_steps

# Reshape the data into a 3D array of shape [n_sequences, n_steps, n_features]
x_lstm = data_array[:n_sequences*n_steps].reshape((n_sequences, n_steps, n_features))
y_lstm = y_multitouch[:n_sequences * n_steps:n_steps]

# Split the data into training and testing sets
x_train_lstm, x_test_lstm, y_train_lstm, y_test_lstm = train_test_split(x_lstm, y_lstm, test_size=0.2)

# model_multitouch = Sequential([
#     LSTM(64, activation='tanh', recurrent_activation='sigmoid',
#          # dropout=0.25, recurrent_dropout=0.25,
#          # kernel_regularizer=l2(0.0001),  # Apply L2 regularization
#          input_shape=(n_steps, n_features)),
#     # Dropout(0.5),
#     Dense(64, activation='relu',
#           kernel_regularizer=l2(0.0001)),  # Apply L2 regularization
#     # Dropout(0.5),
#     Dense(n_outputs, activation='sigmoid')  # Use sigmoid for multi-label classification
# ])
#
# model_multitouch.compile(optimizer=Adam(learning_rate=1e-3),
#                    loss='binary_crossentropy',  # Use binary crossentropy for multi-label classification
#                    metrics=['accuracy'])

# Initialize early stopping callback to avoid overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5, verbose=1)

# history_multitouch = model_multitouch.fit(x_train_lstm, y_train_lstm,
#                                           epochs=1000, batch_size=32,
#                                           validation_data=(x_test_lstm, y_test_lstm),
#                                           callbacks=[early_stopping, reduce_lr],
#                                           verbose=1)

check_cuda()
# Bidirectional LSTM
model_multitouch = Sequential([
    # Bidirectional LSTM layer
    Bidirectional(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                       # dropout=0.25, recurrent_dropout=0.25,  # Enable dropout
                       kernel_regularizer=l2(0.0001),  # Adjusted L2 regularization
                       return_sequences=True),  # Only if adding another LSTM layer after this one
                       input_shape=(n_steps, n_features)),
    # Additional LSTM layer, remove 'return_sequences=True' if this is the last LSTM layer
    LSTM(128, activation='tanh', recurrent_activation='sigmoid',
         # dropout=0.25, recurrent_dropout=0.25,  # Enable dropout
         kernel_regularizer=l2(0.0001)),  # Adjusted L2 regularization
    # Dropout layer
    # Dropout(0.5),
    # Dense layer with regularization
    Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
    # Dropout(0.5),  # Additional dropout layer before the final output layer
    # Output layer for multi-label classification
    Dense(n_outputs, activation='sigmoid')
])


# Compile the model with an optimizer and learning rate adjustment
model_multitouch.compile(optimizer=Adam(learning_rate=1e-3),  # Adjusted learning rate
                         loss='binary_crossentropy',  # Assuming binary crossentropy is appropriate for your task
                         metrics=['accuracy'])

history_multitouch = model_multitouch.fit(x_train_lstm, y_train_lstm,
                                          epochs=150, batch_size=32,
                                          validation_data=(x_test_lstm, y_test_lstm),
                                          callbacks=None,
                                          verbose=1)

# def convert_tflite():
#     # Convert the trained LSTM model to TensorFlow Lite format
#     lstm_converter = tf.lite.TFLiteConverter.from_keras_model(model_multitouch)
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
#     keras_model_path = '/home/chengjin/Projects/SoftMag/Savings/lstm_multiclassification_model.h5'
#     model_multitouch.save(keras_model_path)
#
#     print(f"LSTM Model saved to {keras_model_path}")
#     # print(f"LSTM Model saved to {tflite_model_path} and {keras_model_path}")


# Debug flag

# Save the trained lstm model as h5 files
model_multitouch.save('Savings/lstm_multiclassification_model.h5')

test = 1