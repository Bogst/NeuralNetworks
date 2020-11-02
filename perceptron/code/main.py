import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
# test

def data_normalization(train_data: np.ndarray, test_data: np.ndarray):
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std


def plot(train_data, train_label):
    plt.scatter(train_data[:, 0], train_data[:, 1], s=25, c=train_label, cmap=plt.cm.Set1)
    plt.show()


def activation_function(x):
    return 1 if x >= 0 else 0


def main():

    # Perceptron can't deal with this data set because it is not linearly separable
    coord, cl = make_moons(300, noise=0.05)

    # Nice non overlapping data set
    # coord, cl = make_blobs(n_samples=300, centers=2, n_features=2, random_state=0, cluster_std=0.5)

    # Not so nice, overlapping data set
    # coord, cl = make_blobs(n_samples=300, centers=2, n_features=2, random_state=0, cluster_std=1, center_box=(80, 100))

    # Split the data set into train data and test data
    X, Xt, y, yt = train_test_split(coord, cl, test_size=0.30, random_state=0)

    # Plot training data
    plot(X, y)

    # Normalize data and display it
    data_normalization(X, Xt)
    plot(X, y)

    # Initialize the perceptrons weights (in case of the perceptron they can all be zeros, if more than one perceptron
    # do not initialize with 0)
    weights = np.zeros(2)

    # Learning rate
    alpha = .5

    # Number of training rounds
    epoch = 30

    # Perceptron bias
    bias = 0

    # Initializing the error vector
    errors = []

    # learning
    for _ in range(epoch):
        predictions = np.vectorize(activation_function)(np.dot(X, weights)+bias)  # predict classes for each point in test set
        error = sum(np.power(y-predictions, 2))  # calculate the error
        errors.append(error)
        weights += alpha*(np.dot(y-predictions, X))  # update the weights according to the error
        bias += np.sum(alpha * (y-predictions))

    # display training data and learned border line
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], s=25, c=y, cmap=plt.cm.Set1)
    a = [(-bias/weights[0]), 0]
    c = [0, (-bias/weights[1])]

    x_axis = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])
    if c[0]-a[0]==0:
        m = 0
    else:
        m = (c[1]-a[1])/(c[0]-a[0])
    y_axis = m*(x_axis - a[0])+a[1]
    ax.plot(x_axis, y_axis)

    plt.show()

    # display the error graph
    plt.plot(errors)
    plt.xlabel('Nr of Epochs')
    plt.ylabel('Number of incorrect classifications')
    plt.show()

    # calculate the accuracy on the test set
    test_predictions = np.vectorize(activation_function)(np.dot(Xt, weights)+bias)
    test_error = sum(np.power(yt - test_predictions, 2))

    print(f"Training error score: {errors[-1]/X.shape[0]}, \nTest error score: {test_error/Xt.shape[0]}")


if __name__ == '__main__':
    main()
