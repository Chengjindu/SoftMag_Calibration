import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import os
import pickle

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def check_cuda():
    if tf.config.list_physical_devices('GPU'):
        print("CUDA is available")
    else:
        print("CUDA is not available")
    print("TensorFlow was built with CUDA:", tf.test.is_built_with_cuda())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Function to remove NaN values and ensure size consistency between datasets
def remove_nan_data(x_force, y_force, x_lstm, y_lstm):
    nan_mask = np.isnan(y_force).any(axis=1)
    x_force_clean = x_force[~nan_mask]
    y_force_clean = y_force[~nan_mask]
    x_lstm_clean = x_lstm[~nan_mask]
    y_lstm_clean = y_lstm[~nan_mask]

    return x_force_clean, y_force_clean, x_lstm_clean, y_lstm_clean

# ------------------------------Shear Data Import and Preprocessing------------------------------------------

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

shear_flux_data = pd.concat(shear_flux_data_list, ignore_index=True)
shear_xyforce_data = pd.concat(shear_fx_fy_data_list, ignore_index=True)
shear_normal_force_data = pd.concat(shear_fz_data_list, ignore_index=True)

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

# ------------------------------Normal Data Import and Preprocessing------------------------------------------

# Load the normal data with headers
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

# Combine the datasets for normal force model
# normal_flux_training = pd.concat([normal_flux_data, shear_flux_data], ignore_index=True)
# normal_force_training = pd.concat([normal_force_data, shear_normal_force_data], ignore_index=True)
normal_flux_training = normal_flux_data
normal_force_training = normal_force_data

# Normalize the flux data and force data
scaler_normal_flux = MinMaxScaler(feature_range=(-1, 1))
x_normalflux = scaler_normal_flux.fit_transform(normal_flux_training[['Bx', 'By', 'Bz']])  # Use Bx, By, Bz for 3-axial prediction

scaler_normal_force = MinMaxScaler(feature_range=(0, 1))
y_normalforce = scaler_normal_force.fit_transform(normal_force_training)  # Normalize Fz

# Check the minimum and maximum values used by the scaler
fz_min, fz_max = scaler_normal_force.data_min_[0], scaler_normal_force.data_max_[0]
print("Fz Min:", fz_min)
print("Fz Max:", fz_max)

# ------------------------------Position Data Import and Preprocessing Dataset 1----------------------------------------

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

normal_flux_data['Position'] = normal_flux_data.apply(lambda row: assign_position_label(row['x'], row['y']), axis=1)
# Filter out the rows with Position_2x2 label as 4
filtered_data = normal_flux_data[normal_flux_data['Position'] != 4].reset_index(drop=True)

# Normalize the filtered flux data to the range (-1, 1)
normalized_filtered_flux_data = scaler_normal_flux.fit_transform(filtered_data[['Bx', 'By', 'Bz']])


def create_time_step_window_data(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Prepare sequential data for LSTM
time_steps = 10  # number of timesteps in each sample
x_lstm, y_lstm = create_time_step_window_data(normalized_filtered_flux_data, filtered_data['Position'].values, time_steps)
y_lstm_onehot = to_categorical(y_lstm, num_classes=4)       # Convert y_lstm to one-hot encoding


# ------------------------------Position Data Import and Processing Dataset 2------------------------------

# # Load datasets and assign labels
# folder_path = 'Data_onehotposition'
# positions = ['P1.txt', 'P2.txt', 'P3.txt', 'P4.txt']
# dataframes = []
#
# for i, position in enumerate(positions, start=1):
#     file_path = os.path.join(folder_path, position)
#     df = pd.read_csv(file_path, sep=r"\s*,\s*", header=0, engine='python')  # regex separator to handle whitespace
#     df['Position'] = i  # Assigning label based on file order
#     dataframes.append(df)
#
# # Combine all data into a single DataFrame
# training_data = pd.concat(dataframes, ignore_index=True)
#
# # Splitting data into features (X) and labels (y) for classification
# x_class = training_data.iloc[:, :3]  # Assume first 3 columns are Bx, By, Bz
# y_class = training_data['Position'] - 1  # Subtract 1 to have labels start from 0
#
# # Normalize features using previously saved normalization parameters
# with open('Savings/normalization_position_params.pkl', 'rb') as f:
#     normalization_params = pickle.load(f)
#
# def normalize_feature(data, min_value, max_value):
#     return 2 * ((data - min_value) / (max_value - min_value)) - 1
#
# x_class['Bx'] = normalize_feature(x_class['Bx'], normalization_params['normal_flux']['Bx_min'], normalization_params['normal_flux']['Bx_max'])  # Normalizing Bx
# x_class['By'] = normalize_feature(x_class['By'], normalization_params['normal_flux']['By_min'], normalization_params['normal_flux']['By_max'])  # Normalizing By
# x_class['Bz'] = normalize_feature(x_class['Bz'], normalization_params['normal_flux']['Bz_min'], normalization_params['normal_flux']['Bz_max'])  # Normalizing Bz
#
# # Prepare sequential data for LSTM
# time_steps = 10  # number of timesteps in each sample
# n_features = 3

# -----------------------------------Sliding Window Reshaping-------------------------------------

# # Function to create time-step window data with overlapping sequences
# def create_time_step_window_data(X, y, time_steps):
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         v = X[i:(i + time_steps)]
#         Xs.append(v)
#         ys.append(y[i + time_steps])
#     return np.array(Xs), np.array(ys)
#
# # Create LSTM input data using sliding window method
# x_lstm, y_lstm = create_time_step_window_data(x_class.values, y_class.values, time_steps)
#
# # Convert y_lstm to one-hot encoding
# y_lstm_onehot = to_categorical(y_lstm, num_classes=4)

# # -----------------------------------Direct Non-overlap Reshaping-------------------------------------
#
# data_array = x_class.values # Convert the DataFrame into a numpy array
#
# n_sequences = len(data_array) // time_steps
#
# x_lstm = data_array[:n_sequences * time_steps].reshape((n_sequences, time_steps, n_features))
# y_sequences = y_class.values[:n_sequences * time_steps: time_steps]
# y_lstm_onehot = to_categorical(y_sequences, num_classes=4)

# ------------------------------Dataset Alignment------------------------------------------

# Ensure datasets are of compatible dimensions for normal training
if len(x_shearflux) > len(x_normalflux):
    repeat_factor = len(x_shearflux) // len(x_normalflux)
    x_normalflux_repeated = np.tile(x_normalflux, (repeat_factor + 1, 1))[:len(x_shearflux)]
    y_normalforce_repeated = np.tile(y_normalforce, (repeat_factor + 1, 1))[:len(x_shearflux)]

    # Use x_shearflux and y_shearforce directly since they are already larger
    x_train_shear, x_test_shear, y_train_shear, y_test_shear = train_test_split(x_shearflux, y_shearforce, test_size=0.2, random_state=42)
    # Use repeated normal datasets
    x_train_normal, x_test_normal, y_train_normal, y_test_normal = train_test_split(x_normalflux_repeated, y_normalforce_repeated, test_size=0.2, random_state=42)

else:
    repeat_factor = len(x_normalflux) // len(x_shearflux)
    x_shearflux_repeated = np.tile(x_shearflux, (repeat_factor + 1, 1))[:len(x_normalflux)]
    y_shearforce_repeated = np.tile(y_shearforce, (repeat_factor + 1, 1))[:len(x_normalflux)]

    # Use x_normalflux and y_normalforce directly since they are already larger
    x_train_normal, x_test_normal, y_train_normal, y_test_normal = train_test_split(x_normalflux, y_normalforce, test_size=0.2, random_state=42)
    # Use repeated shear and lstm datasets
    x_train_shear, x_test_shear, y_train_shear, y_test_shear = train_test_split(x_shearflux_repeated, y_shearforce_repeated, test_size=0.2, random_state=42)

# Ensure datasets are of compatible dimensions for positioning training
repeat_factor = len(x_shearflux) // len(x_lstm)
x_lstm_repeated = np.tile(x_lstm, (repeat_factor + 1, 1, 1))[:len(x_shearflux)]
y_lstm_onehot_repeated = np.tile(y_lstm_onehot, (repeat_factor + 1, 1))[:len(x_shearflux)]

x_train_lstm, x_test_lstm, y_train_lstm, y_test_lstm = train_test_split(x_lstm_repeated, y_lstm_onehot_repeated,
                                                                        test_size=0.2, random_state=42)


# Clean NaN values
x_test_normal, y_test_normal, x_test_lstm, y_test_lstm = remove_nan_data(x_test_normal, y_test_normal, x_test_lstm, y_test_lstm)
x_train_normal, y_train_normal, x_train_lstm, y_train_lstm = remove_nan_data(x_train_normal, y_train_normal, x_train_lstm, y_train_lstm)


# ------------------------------Multitask Model Definition and Training------------------------------------------

# Define the multitask model
input_flux_shared = Input(shape=(2,), name='flux_shared_input')     # Shared input for Bx and By
input_bz = Input(shape=(1,), name='bz_input')    # Additional input for Bz (specific to the normal task)
input_normal = Concatenate()([input_flux_shared, input_bz])     # Combine Bx, By, and Bz for the normal task
input_flux_seq = Input(shape=(time_steps, 3), name='flux_seq_input')  # Sequence input for LSTM (positioning task)

# Task-specific Dense layers for shear and normal force predictions
shear_dense = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(input_flux_shared)
normal_dense = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(input_normal)

# # Optional task-specific dropout for added regularization
# shear_dense = Dropout(0.5)(shared_dense_shear)
# normal_dense = Dropout(0.5)(shared_dense_normal)

# LSTM layers for sequence data
lstm_out = LSTM(64, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(1e-4))(input_flux_seq)

# Combine features from the two parts
combined_features = Concatenate()([shear_dense, normal_dense, lstm_out])

# Dense layers for multitask learning
combined_dense_1 = Dense(96, activation='relu', kernel_regularizer=l2(1e-4))(combined_features)
combined_dense_2 = Dense(48, activation='relu', kernel_regularizer=l2(1e-4))(combined_dense_1)
combined_dense_3 = Dense(16, activation='relu', kernel_regularizer=l2(1e-4))(combined_dense_2)

# Output layers
shear_output = Dense(2, activation='linear', name='shear_output')(combined_dense_3)  # Fx, Fy
normal_output = Dense(1, activation='linear', name='normal_output')(combined_dense_3)  # Fz
position_output = Dense(4, activation='softmax', name='position_output')(combined_dense_3)  # P1-P4

print(f"Training multi-task model without weight coefficients")
# Define and compile the multitask model
multitask_model = Model(inputs=[input_flux_shared, input_bz, input_flux_seq],
                        outputs=[shear_output, normal_output, position_output])

opt = Adam(learning_rate=1e-4, clipnorm=1.0)    # Use a smaller learning rate and add gradient clipping

multitask_model.compile(optimizer=opt,
                        loss={'shear_output': 'mean_squared_error',
                              'normal_output': 'mean_squared_error',
                              'position_output': 'categorical_crossentropy'},
                        metrics={'shear_output': 'mae',
                                 'normal_output': 'mae',
                                 'position_output': 'accuracy'})

# Early stopping callback to avoid overfitting or prolonged training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Train the model
history = multitask_model.fit({'flux_shared_input': x_train_shear, 'bz_input': x_train_normal[:, 2:], 'flux_seq_input': x_train_lstm},
                              {'shear_output': y_train_shear, 'normal_output': y_train_normal, 'position_output': y_train_lstm},
                              validation_data=({'flux_shared_input': x_test_shear, 'bz_input': x_test_normal[:, 2:], 'flux_seq_input': x_test_lstm},
                                               {'shear_output': y_test_shear, 'normal_output': y_test_normal, 'position_output': y_test_lstm}),
                              epochs=200,
                              batch_size=32,
                              callbacks=[early_stopping],
                              verbose=1)

# Save the trained model
multitask_model.save('Savings/multitask_model.h5')

# Evaluate the model
evaluation_results = multitask_model.evaluate(
    {'flux_shared_input': x_test_shear, 'bz_input': x_test_normal[:, 2:], 'flux_seq_input': x_test_lstm},
    {'shear_output': y_test_shear, 'normal_output': y_test_normal, 'position_output': y_test_lstm},
    verbose=1
)

# Unpack the results correctly
total_loss = evaluation_results[0]
shear_loss = evaluation_results[1]
normal_loss = evaluation_results[2]
position_loss = evaluation_results[3]
shear_mae = evaluation_results[4]
normal_mae = evaluation_results[5]
position_accuracy = evaluation_results[6]

# Print the results
print(f"Total Loss: {total_loss}")
print(f"Shear Loss (MSE): {shear_loss}")
print(f"Normal Loss (MSE): {normal_loss}")
print(f"Position Loss (Categorical Crossentropy): {position_loss}")
print(f"Shear MAE: {shear_mae}")
print(f"Normal MAE: {normal_mae}")
print(f"Position Accuracy: {position_accuracy}")

# Predict on the test set
force_preds, normal_preds, position_preds = multitask_model.predict(
    {'flux_shared_input': x_test_shear, 'bz_input': x_test_normal[:, 2:], 'flux_seq_input': x_test_lstm}
)

print("force_preds contains NaN: ", np.isnan(force_preds).any())
print("normal_preds contains NaN: ", np.isnan(normal_preds).any())
print("position_preds contains NaN: ", np.isnan(position_preds).any())

# ------------------------------Multitask training with weighted losses------------------------------------------

print(f"Training multi-task model without weight coefficients")
# Define weights for each task's loss
loss_weights = {'shear_output': 0.5, 'normal_output': 1.0, 'position_output': 0.5}

# Compile the model again with the weighted losses
multitask_model_weighted = Model(inputs=[input_flux_shared, input_bz, input_flux_seq],
                                 outputs=[shear_output, normal_output, position_output])

multitask_model_weighted.compile(optimizer=opt,
                                 loss={'shear_output': 'mean_squared_error',
                                       'normal_output': 'mean_squared_error',
                                       'position_output': 'categorical_crossentropy'},
                                 loss_weights=loss_weights,
                                 metrics={'shear_output': 'mae',
                                          'normal_output': 'mae',
                                          'position_output': 'accuracy'})

# Train the weighted loss model
history_weighted = multitask_model_weighted.fit({'flux_shared_input': x_train_shear, 'bz_input': x_train_normal[:, 2:], 'flux_seq_input': x_train_lstm},
                                                {'shear_output': y_train_shear, 'normal_output': y_train_normal, 'position_output': y_train_lstm},
                                                validation_data=({'flux_shared_input': x_test_shear, 'bz_input': x_test_normal[:, 2:], 'flux_seq_input': x_test_lstm},
                                                                 {'shear_output': y_test_shear, 'normal_output': y_test_normal, 'position_output': y_test_lstm}),
                                                epochs=200,
                                                batch_size=32,
                                                callbacks=[early_stopping],
                                                verbose=1)

# Save the weighted model
multitask_model_weighted.save('Savings/multitask_model_weighted.h5')

# Evaluate the weighted model
evaluation_results_weighted = multitask_model_weighted.evaluate(
    {'flux_shared_input': x_test_shear, 'bz_input': x_test_normal[:, 2:], 'flux_seq_input': x_test_lstm},
    {'shear_output': y_test_shear, 'normal_output': y_test_normal, 'position_output': y_test_lstm},
    verbose=1
)

# Unpack the results correctly
total_loss_weighted = evaluation_results_weighted[0]
shear_loss_weighted = evaluation_results_weighted[1]
normal_loss_weighted = evaluation_results_weighted[2]
position_loss_weighted = evaluation_results_weighted[3]
shear_mae_weighted = evaluation_results_weighted[4]
normal_mae_weighted = evaluation_results_weighted[5]
position_accuracy_weighted = evaluation_results_weighted[6]

# Print the results for weighted training
print(f"Total Loss (Weighted): {total_loss_weighted}")
print(f"Shear Loss (Weighted MSE): {shear_loss_weighted}")
print(f"Normal Loss (Weighted MSE): {normal_loss_weighted}")
print(f"Position Loss (Weighted Categorical Crossentropy): {position_loss_weighted}")
print(f"Shear MAE (Weighted): {shear_mae_weighted}")
print(f"Normal MAE (Weighted): {normal_mae_weighted}")
print(f"Position Accuracy (Weighted): {position_accuracy_weighted}")

# Predict on the test set with the weighted model
force_preds_weighted, normal_preds_weighted, position_preds_weighted = multitask_model_weighted.predict(
    {'flux_shared_input': x_test_shear, 'bz_input': x_test_normal[:, 2:], 'flux_seq_input': x_test_lstm}
)

print("force_preds_weighted contains NaN: ", np.isnan(force_preds_weighted).any())
print("normal_preds_weighted contains NaN: ", np.isnan(normal_preds_weighted).any())
print("position_preds_weighted contains NaN: ", np.isnan(position_preds_weighted).any())

# ------------------------------Comparison between Standard and Weighted Loss Training------------------------------------------

# Compare the results from the standard training and weighted training
print("\n-------------------- Comparison of Standard vs Weighted Training --------------------")
print(f"Standard Training - Total Loss: {total_loss}, Shear Loss: {shear_loss}, Normal Loss: {normal_loss}, Position Loss: {position_loss}")
print(f"Weighted Training - Total Loss: {total_loss_weighted}, Shear Loss: {shear_loss_weighted}, Normal Loss: {normal_loss_weighted}, Position Loss: {position_loss_weighted}")

print(f"\nStandard Training - Shear MAE: {shear_mae}, Normal MAE: {normal_mae}, Position Accuracy: {position_accuracy}")
print(f"Weighted Training - Shear MAE: {shear_mae_weighted}, Normal MAE: {normal_mae_weighted}, Position Accuracy: {position_accuracy_weighted}")

# Compare the final predictions for a few samples
print("\nComparison of Predictions (First 5 samples):")
for i in range(5):
    print(f"\nSample {i+1}:")
    print(f"Standard - Shear Pred: {force_preds[i]}, Normal Pred: {normal_preds[i]}, Position Pred: {position_preds[i]}")
    print(f"Weighted - Shear Pred: {force_preds_weighted[i]}, Normal Pred: {normal_preds_weighted[i]}, Position Pred: {position_preds_weighted[i]}")


# Debug flag
debugflag = 1