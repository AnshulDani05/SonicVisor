import os
import numpy as np
import librosa
import json
import sys

HOP_LENGTH = 512  # Distance between the start of one frame to the start of the next one
N_FFT = 2048  # Number of points to use in the FFT
N_MFCC = 13  # Number of MFCCs to extract
SR = 22050  # Sampling rate
max_segment_length = 10  # Length of the segments
overlap = 5  # Distance between start of one segment to start of next (in seconds)
# Dictionary mapping instruments to labels
instruments = {"accordion": 0, "banjo": 1, "bass": 2, "cello": 3, "clarinet": 4, "cymbals": 5, "drums": 6, "flute": 7,
               "guitar": 8, "mallet_percussion": 9, "mandolin": 10, "organ": 11, "piano": 12, "saxophone": 13,
               "synthesizer": 14, "trombone": 15, "trumpet": 16, "ukulele": 17, "violin": 18, "voice": 19}
json_path = r'E:\NEA_Files\Training_Data_Instruments'  # Json path to store processed data to


def process_dataset(audio_dataset_folder, labels_dataset_folder, num_mfcc=N_MFCC):
    '''
    Process the data set.
    1. Iterate through the folders in the audio_dataset_folder
    2. Split each track into segments, then calculate the mfccs
    3. Store the mfccs of each frame against the labels
    4. Return the mfccs
    :param audio_dataset_folder: the folder containing all the audio data
    :param labels_dataset_folder: the labels of the data set folders
    :param num_mfcc: the number of mfccs to extract
    :return:
    '''
    all_mfccs = {}
    labels = {}
    first_line = True

    with open(labels_dataset_folder, 'r') as file:
        for line in file:
            if first_line == True:
                first_line = False
            else:
                line = line.split(',')
                if line[0] not in labels:
                    labels[line[0]] = []
                labels[line[0]].append(instruments.get(line[1]))

    folder_files = [os.path.join(audio_dataset_folder, f) for f in os.listdir(audio_dataset_folder)]
    count = 0
    for folder in folder_files:
        file_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        for file_path in file_paths:
            try:
                audio, sr = librosa.load(file_path, sr=SR)
                segment_length_samples = int(max_segment_length * sr)
                overlap_samples = int(overlap * sr)
                num_segments = int(np.ceil(len(audio) / (segment_length_samples - overlap_samples)))

                for i in range(num_segments):
                    start = i * (segment_length_samples - overlap_samples)
                    end = min(start + segment_length_samples, len(audio))
                    segment = audio[start:end]
                    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=num_mfcc, hop_length=HOP_LENGTH, n_fft=N_FFT)
                    frames = int(np.ceil(((max_segment_length * SR) / HOP_LENGTH)))

                    if mfccs.shape[1] < frames:
                        pad_width = frames - mfccs.shape[1]
                        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
                    else:
                        mfccs = mfccs[:, :frames]
                    path = os.path.splitext(os.path.basename(file_path))[0]
                    all_mfccs[f'{path}-{i}'] = [mfccs.T.tolist(), labels.get(path)]


            except:
                print(file_path)
        print(count)
        count += 1
    return all_mfccs


def store_to_json(mfccs_data):
    '''
    Store the processed data to a json file
    :param mfccs_data: data to store
    :return:
    '''
    for i in range(len(mfccs_data)):
        with open(fr'{json_path}\{i}.json.txt', 'w') as fp:
            json.dump(list((mfccs_data.values()))[i], fp, indent=4)
            print('data stored!')


def process_data(file_path, num_mfcc=N_MFCC):
    '''
    Split the input data into segments and MFCCs and return all the MFCCs
    :param file_path: the file path to the audio
    :param num_mfcc: the number of mfccs
    :return:
    '''
    all_mfccs = []
    try:
        audio, sr = librosa.load(file_path, sr=SR)
        segment_length_samples = int(max_segment_length * sr)
        overlap_samples = int(overlap * sr)
        num_segments = int(np.ceil(len(audio) / (segment_length_samples - overlap_samples)))

        for i in range(num_segments):
            start = i * (segment_length_samples - overlap_samples)
            end = min(start + segment_length_samples, len(audio))
            segment = audio[start:end]
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=num_mfcc, hop_length=HOP_LENGTH, n_fft=N_FFT)
            frames = int(np.ceil(((max_segment_length * SR) / HOP_LENGTH)))

            if mfccs.shape[1] < frames:
                pad_width = frames - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :frames]
            all_mfccs.append(mfccs.T)

    except:
        print('Failed')

    return all_mfccs


def main():
    '''
    Main function to process data
    :return:
    '''
    audio_dataset_folder = r'E:\NEA_Files\openmic-2018-v1.0.0\openmic-2018\audio'
    labels_dataset_folder = r'E:\NEA_Files\openmic-2018-v1.0.0\openmic-2018\openmic-2018-aggregated-labels.csv'
    mfccs_data = process_dataset(audio_dataset_folder, labels_dataset_folder, num_mfcc=13)
    store_to_json(mfccs_data)


# main()
