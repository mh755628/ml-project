
import json

import itertools

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix






def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


DATASET_PATH

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    outputs = np.array(data["label"])

    return inputs, outputs


def get_model(input_shape):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape = input_shape))
    model.add(MaxPool2D((3, 3), strides = (2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape = input_shape))
    model.add(MaxPool2D((3, 3), strides = (2, 2), padding='same'))
    model.add(BatchNormalization())


    model.add(Conv2D(32, (2, 2), activation='relu', input_shape = input_shape))
    model.add(MaxPool2D((2, 2), strides = (2, 2), padding='same'))
    model.add(BatchNormalization())


    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.3))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.2))

    model.add(Dense(3, activation = 'softmax'))

    optimizer = Adam(learning_rate = 0.001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

    return model


def prepare_datasets(test_size, validation_size):

    X, y = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, random_state=1)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

if __name__ == "__main__":

    DATASET_PATH = input("Enter dataset path:");

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(test_size = .25, validation_size = .2)


    model = get_model(X_train[0].shape)

    print(X_train[0].shape)

    model.fit(X_train, y_train, validation_data = (X_validation, y_validation), batch_size=32, epochs=20)


    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

    predictions = model.predict(X_test, batch_size=10, verbose=0)

    

    rounded_predictions = np.argmax(predictions, axis=-1)

    print(rounded_predictions)

    cm = confusion_matrix(y_true=y_test, y_pred=rounded_predictions)

    cm_plot_labels = ['bangla','english', 'hindi']

    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')