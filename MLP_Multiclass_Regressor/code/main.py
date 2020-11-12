import math
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


def data_normalization(train_data: np.ndarray, test_data: np.ndarray):
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def sigmoid_prime(z):
    return z * (1 - z)


def Softmax(x):
    return np.exp(x - np.max(x))/(np.sum(np.exp(x - np.max(x))))


def init(inp, out):
    return np.random.randn(inp, out) / np.sqrt(inp)


def create_architecture(layers, random_seed=0):
    np.random.seed(random_seed)
    arch = list(zip(layers[:-1], layers[1:]))
    weights = [init(out, inp) for inp, out in arch]
    biases = [init(out, 1) for inp, out in arch]
    return weights, biases


def feedforward(input_image, weights, bias, activation_function):
    a = [input_image.reshape(len(input_image), 1)]
    for i in range(len(weights)-1):
        a.append(activation_function((weights[i] @ a[i]).reshape(len(weights[i]), 1) + bias[i]))
    y_hat = (weights[-1] @ a[-1]) + bias[-1]
    a.append(y_hat)
    return a


def backprop(outputs, y, batch_weights, batch_bias, weights, activation_function_prime):
    delta_error = list(np.empty_like(outputs))
    index_count = len(weights)
    delta_error[index_count] = [[(math.pow(outputs[index_count] - y, 1))]]
    batch_bias[index_count - 1] = batch_bias[index_count - 1] + delta_error[index_count] # Output Layer
    batch_weights[index_count - 1] = batch_weights[index_count - 1] - (delta_error[index_count] @ outputs[index_count - 1].T) # Output Layer
    for i in range(index_count - 1, 0, -1):
        h_derivative = np.array(activation_function_prime(outputs[i])).reshape(1, len(outputs[i])) * np.eye(len(outputs[i]))
        delta_error[i] = h_derivative.T @ weights[i].T @ delta_error[i+1]
        batch_bias[i - 1] = batch_bias[i - 1] + delta_error[i]
        batch_weights[i - 1] = batch_weights[i - 1] + (delta_error[i] @ outputs[i - 1].T)

    sum_error = 0
    for error in delta_error:
        if error is not None:
            sum_error += np.absolute(error).sum()
    return sum_error


def main():
    # Load mnist data set
    boston = load_boston()

    # Split data set in train set and test set
    X, Xt, y, yt = train_test_split(boston.data, boston.target, test_size=0.30, random_state=0)

    # Normalize data
    data_normalization(X, Xt)

    hyper_parameters = {
        'architecture': [X.shape[1], 50, 1],
        'hidden_layer_activation': sigmoid,
        'hidden_layer_activation_derivative': sigmoid_prime,
        'learning_rate': 0.035,
        'batch_size': 50,
        'epochs': 20
    }
    batch_numbers = int(len(X) / hyper_parameters['batch_size'])

    history = {
        "batch_average_acc": [],
        "epoch_average_acc": [],
        'batch_sum_loss': [],
        'epoch_sum_loss': []
    }

    # Create Model
    weights, biases = create_architecture(hyper_parameters['architecture'])
    batch_weights = weights.copy()
    batch_biases = biases.copy()

    for i in range(hyper_parameters['epochs']):
        print(f"Running Epoch {i+1}")
        if 1 < i < 4:
            hyper_parameters['learning_rate'] = hyper_parameters['learning_rate']/2
        for j in range(batch_numbers):

            # Calculate batch loss
            batch_loss = 0
            batch_correct_predictions = 0
            for k in range(hyper_parameters['batch_size']):
                batch_index = j * hyper_parameters['batch_size'] + k
                outputs = feedforward(X[batch_index], weights, biases, hyper_parameters['hidden_layer_activation'])
                batch_correct_predictions += math.sqrt(math.pow(outputs[-1] -y[batch_index], 2))
                loss = backprop(outputs, y[batch_index], batch_weights, batch_biases, weights, hyper_parameters['hidden_layer_activation_derivative'])
                batch_loss += abs(loss)

            # Update weights according to batch loss
            for layer_index in range(len(weights)):
                weight = weights[layer_index]
                batch_average_weight = np.multiply(batch_weights[layer_index], (hyper_parameters['learning_rate'] / hyper_parameters['batch_size']))
                weights[layer_index] = np.subtract(weight, batch_average_weight)
                bias = biases[layer_index]
                batch_average_bias = np.multiply(batch_biases[layer_index], (hyper_parameters['learning_rate'] / hyper_parameters['batch_size']))
                biases[layer_index] = np.subtract(bias, batch_average_bias)

            batch_weights = weights.copy()
            batch_biases = biases.copy()

            # Record batch sum loss
            print(f"Loss for batch number {j+1} in epoch {i+1}: {batch_loss}")
            history['batch_sum_loss'].append(batch_loss)

            # Record batch average accuracy
            history['batch_average_acc'].append((batch_correct_predictions / hyper_parameters['batch_size']))
            print(f"Batch average {j} acc {history['batch_average_acc'][-1]}")

        # Record epoch sum loss
        history['epoch_sum_loss'].append(np.sum(np.abs(history['batch_sum_loss'][i*batch_numbers: (i+1)*batch_numbers])))
        print(f"Loss for epoch {i}: {history['epoch_sum_loss'][i]}")

        # Record end of epoch accuracy
        epoch_correct_prediction = 0
        for curr_x, curr_y in zip(X, y):
            prediction = feedforward(curr_x, weights, biases, hyper_parameters['hidden_layer_activation'])[-1]
            epoch_correct_prediction += math.sqrt(math.pow(prediction - curr_y, 2))
        history['epoch_average_acc'].append((epoch_correct_prediction / len(X)))
        print(f'Accuracy for epoch {i}: {history["epoch_average_acc"][i]}')

    # Validation set accuracy
    test_set_correct_prediction = 0
    for curr_x, curr_y in zip(Xt, yt):
        prediction = feedforward(curr_x, weights, biases, hyper_parameters['hidden_layer_activation'])[-1]
        test_set_correct_prediction += math.sqrt(math.pow(prediction - curr_y, 2)) / len(Xt)  # Mean square error
    print(f'Test set accuracy {test_set_correct_prediction}')

    plt.plot(history['batch_sum_loss'])
    plt.xlabel('Nr of Batches')
    plt.ylabel('Sum Loss Value')
    plt.title('Batch Sum Loss')
    plt.show()

    plt.plot(history['batch_average_acc'])
    plt.xlabel('Nr of Batches')
    plt.ylabel('Mean Error')
    plt.title("Batch Mean Error")
    plt.show()

    plt.plot(history['epoch_sum_loss'])
    plt.xlabel('Nr of Epochs')
    plt.ylabel('Sum Loss Value')
    plt.title('Epoch Sum Loss')
    plt.show()

    plt.plot(history['epoch_average_acc'])
    plt.xlabel('Nr of Epochs')
    plt.ylabel('Mean Error')
    plt.title('Epoch average error')
    plt.show()


if __name__ == "__main__":
    main()
