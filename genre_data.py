# This code loads the genre audio files, and converts it into data which can be passed into the ML models
# It then stores this data to a json file

import os  # Module for interacting with file paths
import numpy as np  # Library for numerical computing
import librosa  # Library for audio and music analysis
import json  # Library for handling files stored in JSON format

HOP_LENGTH = 512  # The jump between the start of each frame
N_FFT = 2048  # The number of points to use in the FFTs
N_MFCC = 13  # The number of MFCCs to get
SR = 22050  # The sampling rate
max_segment_length = 10  # The segment length it will be divided into
overlap = 5  # The overlap between the start of one segment and the next
genres = {'0': 'blues', '1': 'classical', '2': 'country', '3': 'disco', '4': 'hiphop', '5': 'jazz', '6': 'metal',
          '7': 'pop', '8': 'reggae', '9': 'rock'}  # The dictionary to match stored keys to genres
json_path = r'E:\NEA_Files\Training_Data_Genre.json.txt'  # The path to store the data to


def process_dataset(dataset_folder, num_mfcc=N_MFCC):
    all_mfccs = []  # The variable to store all the MFCC data
    labels = []  # The variable to store all the labels data
    label = 0  # Current label file being iterated through
    folder_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)]
    for folder in folder_files:
        file_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        for file_path in file_paths:
            try:
                # Load audio file
                audio, sr = librosa.load(file_path, sr=SR)
                segment_length_samples = int(max_segment_length * sr)
                overlap_samples = int(overlap * sr)
                num_segments = int(np.ceil(len(audio) / (segment_length_samples - overlap_samples)))

                # Split the data into segments and calculate the MFCC values
                for i in range(num_segments):
                    start = i * (segment_length_samples - overlap_samples)
                    end = min(start + segment_length_samples, len(audio))
                    segment = audio[start:end]
                    # Calculate MFCCs for the segment
                    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=num_mfcc, hop_length=HOP_LENGTH, n_fft=N_FFT)
                    frames = int(np.ceil(((max_segment_length * SR) / HOP_LENGTH)))
                    # Pad or truncate MFCCs to match the desired length
                    if mfccs.shape[1] < frames:
                        pad_width = frames - mfccs.shape[1]
                        # Pad the MFCCs with zeros to match the desired frame length
                        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
                    else:
                        # Truncate MFCCs to match the desired frame length
                        mfccs = mfccs[:, :frames]
                    all_mfccs.append(mfccs.T)
                    labels.append(str(label))

            except:
                print(file_path)
        label += 1

    print('data processed')

    return np.array(all_mfccs), np.array(labels)


def store_to_json(mfccs_labels):
    '''
    Stores the data to a json file, putting it in a dictionary with the key values 'MFCCs and Labels'
    :param mfccs_labels: The MFCCs and Labels data
    :return:
    '''
    my_dict = {'mfccs': mfccs_labels[0], 'labels': mfccs_labels[1]}
    with open(json_path, 'w') as fp:
        json.dump(my_dict, fp, indent=4)
        print('Done!')


# Path to the dataset folder
dataset_folder = r'E:\NEA_Files\genre\Data\genres_original'
# Process the dataset to extract MFCCs and labels
mfccs_data, labels_data = process_dataset(dataset_folder, num_mfcc=13)
# Combine MFCCs and labels into a list
mfccs_labels = [mfccs_data.tolist(), labels_data.tolist()]
# Store the data to a JSON file
store_to_json(mfccs_labels)
