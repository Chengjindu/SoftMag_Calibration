import pandas as pd                     # for data handling
import os
import matplotlib.pyplot as plt
from scipy.signal import cheby1, lfilter
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pickle

# Assuming your data files are in the 'Datasets' directory
data_dir = '/home/chengjin/Projects/SoftMag/Calibration/Data_coordinates'
data_files = sorted(os.listdir(data_dir))

# Function to read and preprocess a single file
data_frames = {}
def read_and_preprocess(file_path):
    # Read the file with whitespace delimiter, skip the first two rows (time and headers)
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=2, dtype=float)
    # Drop rows with NaN values
    df = df.dropna()
    # Assign column names
    df.columns = ['Time', 'Displacement', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz', 'Bx', 'By', 'Bz']
    return df

# Loop through files and read data
for file_name in data_files:
    if file_name.endswith("CAL.txt"):
        position = file_name[2]  # Extracting the position number
        file_path = os.path.join(data_dir, file_name)
        data_frames[position] = read_and_preprocess(file_path)

# Extract the desired columns from each dataframe
raw_data = {}
for position, df in data_frames.items():
    # Get displacement, Fz, Bx, By, Bz. .iloc: Purely integer-location based indexing for selection by position
    raw_data[position] = df[['Displacement', 'Fz', 'Bx', 'By', 'Bz']]
    # Make Fz values positive
    raw_data[position].loc[:,'Displacement'] = raw_data[position]['Displacement'].abs()
    raw_data[position].loc[:,'Fz'] = raw_data[position]['Fz'].abs()

# Observation on the raw data
# Plotting Displacement vs Force for each position
plt.figure(figsize=(14, 10))
for i, position in enumerate(raw_data.keys(), 1):
    plt.subplot(2, 2, i)
    plt.plot(raw_data[position]['Displacement'], raw_data[position]['Fz'], label='Displacement vs Force')
    plt.title(f'Position {position}: Displacement vs Force')
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.legend()
plt.tight_layout()

# Plotting Bx, By, Bz for each position
plt.figure(figsize=(14, 10))
for i, position in enumerate(raw_data.keys(), 1):
    plt.subplot(2, 2, i)
    plt.plot(raw_data[position]['Bx'], label='Bx')
    plt.plot(raw_data[position]['By'], label='By')
    plt.plot(raw_data[position]['Bz'], label='Bz')
    plt.title(f'Position {position}: Magnetic Flux (Bx, By, Bz)')
    plt.xlabel('Sample Number')
    plt.ylabel('Magnetic Flux [Gauss]')
    plt.legend()
plt.tight_layout()

# Filter parameters
order = 5
sampling_freq = 100  # Hz
nyquist_freq = sampling_freq / 2
passband_freq = 3  # Hz
stopband_freq = 10  # Hz
passband_ripple = 1  # dB
stopband_attenuation = 60  # dB

# Normalize the frequency to the Nyquist frequency (half the sampling rate)
normalized_passband_freq = passband_freq / nyquist_freq

# Design the Chebyshev Type I filter
b, a = cheby1(N=order, rp=passband_ripple, Wn=normalized_passband_freq, btype='low', analog=False)

# Create a deep copy of raw_data to store the filtered data
filtered_data = copy.deepcopy(raw_data)

# Apply the filter to each column of interest in the copied data
for position, df in filtered_data.items():
    for col in ['Fz', 'Bx', 'By', 'Bz']:  # Assuming columns 1 to 4 (Fz, Bx, By, Bz) need to be filtered
        df[col] = lfilter(b, a, df[col])

# Observation after filtering
# Plotting Displacement vs Force for each position (Raw vs Filtered)
plt.figure(figsize=(14, 10))
for i, position in enumerate(raw_data.keys(), 1):
    plt.subplot(2, 2, i)
    plt.plot(raw_data[position]['Fz'], label='Raw Data')
    plt.plot(filtered_data[position]['Fz'], label='Filtered Data', linestyle='--')
    plt.title(f'Position {position}: Displacement vs Force')
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.legend()
plt.tight_layout()

# Plotting Bx, By, Bz for each position (Raw vs Filtered)
plt.figure(figsize=(14, 10))
for i, position in enumerate(raw_data.keys(), 1):
    plt.subplot(2, 2, i)
    plt.plot(raw_data[position]['Bx'], label='Bx')
    plt.plot(raw_data[position]['By'], label='By')
    plt.plot(raw_data[position]['Bz'], label='Bz')
    plt.plot(filtered_data[position]['Bx'], label='Filtered Bx', linestyle='--')
    plt.plot(filtered_data[position]['By'], label='Filtered By', linestyle='--')
    plt.plot(filtered_data[position]['Bz'], label='Filtered Bz', linestyle='--')
    plt.title(f'Position {position}: Magnetic Flux (Bx, By, Bz)')
    plt.xlabel('Sample Number')
    plt.ylabel('Magnetic Flux [Gauss]')
    plt.legend()
plt.tight_layout()

# Eliminating deviation
for position, df in filtered_data.items():
    for col in df.columns:
        initial_value = df[col].iloc[0]
        df[col] = df[col] - initial_value

# Plotting Bx, By, Bz for each position (Raw vs Filtered)
plt.figure(figsize=(14, 10))
for i, position in enumerate(filtered_data.keys(), 1):
    plt.subplot(2, 2, i)
    plt.plot(filtered_data[position]['Bx'], label='Filtered Bx', linestyle='-')
    plt.plot(filtered_data[position]['By'], label='Filtered By', linestyle='-')
    plt.plot(filtered_data[position]['Bz'], label='Filtered Bz', linestyle='-')
    plt.title(f'Position {position}: Magnetic Flux (Bx, By, Bz)')
    plt.xlabel('Sample Number')
    plt.ylabel('Magnetic Flux [Gauss]')
    plt.legend()
plt.tight_layout()

# Normalizing the last three columns, which are assumed to be Bx, By, Bz
scaler = StandardScaler()
# Fit the scaler on the data from position 1 (Because the mean and deviation of the position 1 are all positive)
position_1_data = filtered_data['1'][['Bx', 'By', 'Bz']]
scaler.fit(position_1_data)
# Save the fitted scaler for later use
scaler_file_path = os.path.join('Savings', 'fitted_scaler.pkl')
with open(scaler_file_path, 'wb') as file:
    pickle.dump(scaler, file)

normalized_data = {}
# Perform normalization on each position's data
for position,df in filtered_data.items():
    df[['Bx', 'By', 'Bz']] = scaler.transform(df[['Bx', 'By', 'Bz']])
    normalized_data[position] = df

# Adding labels
for position,df in normalized_data.items():
    df['Position'] = int(position)

# Observation
plt.figure(figsize=(14, 10))
for i, position in enumerate(normalized_data.keys(), 1):
    plt.subplot(2, 2, i)  # Adjust the subplot grid if needed
    plt.plot(normalized_data[position]['Bx'], label='Normalized Bx', linestyle='-')
    plt.plot(normalized_data[position]['By'], label='Normalized By', linestyle='-')
    plt.plot(normalized_data[position]['Bz'], label='Normalized Bz', linestyle='-')
    plt.title(f'Position {position}: Normalized Magnetic Flux (Bx, By, Bz)')
    plt.xlabel('Sample Number')
    plt.ylabel('Normalized Magnetic Flux')
    plt.legend()
plt.tight_layout()
# plt.show()

# Concatenation and shuffle
training_data = pd.concat(normalized_data.values(), ignore_index=True)
training_data = shuffle(training_data)

# Save the processed data to a file
training_data.to_pickle('/home/chengjin/Projects/SoftMag/Savings/training_data.pkl')


# Debug flag
test = 1