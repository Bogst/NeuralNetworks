from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
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
    acc_values = history_dict['binary_accuracy']
    val_acc_values = history_dict['val_binary_accuracy']

    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def main():
    # Perceptron can't deal with this data set because it is not linearly separable
    # coord, cl = make_moons(300, noise=0.05)

    # Nice non overlapping data set
    coord, cl = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0, cluster_std=0.5)

    # Not so nice, overlapping data set
    # coord, cl = make_blobs(n_samples=300, centers=2, n_features=2, random_state=0, cluster_std=1, center_box=(80, 100))

    train_data, test_data, train_labels, test_labels = train_test_split(coord, cl, test_size=0.30, random_state=0)

    data_normalization(train_data, test_data)

    train_labels = np.asanyarray(train_labels).astype('float32')
    test_data = np.asanyarray(test_data).astype('float32')

    model = models.Sequential()
    model.add(layers.Dense(1, activation='sigmoid', input_shape=(2,)))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_data, train_labels, epochs=30, batch_size=10, validation_data=(test_data, test_labels))
    plotts(history)


if __name__ == "__main__":
    main()
