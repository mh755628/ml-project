import load_data
import numpy as np
import sys

mfccs = load_data.load_mfccs(input("Enter the file path: "))

model = load_data.get_model("cnn")

label = ['bangla', 'english', 'hindi']


frequency = [0, 0, 0]

import matplotlib.pyplot as plt


for mfcc in mfccs:
    prediction = model.predict_classes(mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1))
    frequency[prediction[0]] += 1
    print('current prediction: ' + label[prediction[0]] + ', currently playing: ' + label[np.argmax(frequency)])
    plt.bar(label, frequency)
    plt.draw()
    plt.pause(0.05)


plt.show()