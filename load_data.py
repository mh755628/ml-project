import os
from signal import signal

import librosa

import json

from tqdm import tqdm

SAMPLE_RATE = 22050
DURATION = 60
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
n_mfcc = 13
n_fft = 2048
hop_length = 512
num_segment = 10

def load_mfccs(data_path):
    S, sr = librosa.load(data_path)
    length_in_sec = int(len(S) / sr)

    mfccs = []

    number_of_sample_per_segment = int(SAMPLES_PER_TRACK / num_segment)

    for z in tqdm(range(1, int(length_in_sec / 60))):
        signal = S[z * 60 * sr: (z + 1) * 60 * sr]
        for s in range(num_segment):
            start_sample = number_of_sample_per_segment * s 
            finish_sample = start_sample + number_of_sample_per_segment

            mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], 
                sr = sr, n_fft = n_fft, n_mfcc = n_mfcc, hop_length = hop_length)

            mfcc = mfcc.T 
            mfccs.append(mfcc)
    return mfccs


import json

import itertools

import numpy as np


from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix



def get_model(model_path):
    model = load_model(model_path)
    return model
