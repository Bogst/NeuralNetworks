from keras import models, layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def data_normalization(train_data: np.ndarray, test_data: np.ndarray):
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std


def plotts(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values)+1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()
    acc_values = history_dict['categorical_accuracy']
    val_acc_values = history_dict['val_categorical_accuracy']

    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def main():
    data, label = load_digits(return_X_y=True)

    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.30, random_state=0)

    # data_normalization(train_data, test_data)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = models.Sequential()
    model.add(layers.Dense(50, activation='relu', input_shape=(64,)))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(train_data, train_labels, epochs=15, batch_size=10, validation_data=(test_data, test_labels))
    plotts(history)


if __name__ == "__main__":
    main()
