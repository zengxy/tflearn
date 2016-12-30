import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


def make_data():
    np.random.seed(0)

    X, y = make_moons(200, noise=0.2)
    # plt.scatter(X[:, 0], X[:, 1], s=60, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    return X, y


def basic_LR():
    X, y = make_data()
    clf = LogisticRegressionCV()
    clf.fit(X, y)
    plot_decision_boundary(lambda x: clf.predict(x), X, y)
    plt.title("Logistic Regression")
    plt.show()


def basic_nn():
    X, y = make_data()
    num_example = len(X)
    nn_input_dim = 2
    nn_hidden_dim = 4
    nn_output_dim = 2
    leanring_rate = 0.01
    regularization_lambda = 0.01

    hidden_layer_dimensions = [1, 2, 3, 4, 5, 10, 20, 50]
    for i, nn_hidden_dim in enumerate(hidden_layer_dimensions):
        plt.subplot(4, 2, i + 1)
        plt.title('Hidden Layer size %d' % nn_hidden_dim)
        model = build_model(X, y,
                            nn_input_dim, nn_hidden_dim, nn_output_dim,
                            regularization_lambda, leanring_rate)
        plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.show()

    plot_decision_boundary(lambda x: predict(model, x), X, y)


def caculate_loss(model, regularization, X, y):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    probs = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)
    log_loss = np.sum(-np.log(probs[range(len(X)), y]))
    regularization_loss = 0.5 * regularization * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return log_loss + regularization_loss


def evaluate(model, X, y):
    prediction = predict(model, X)
    right_rate = np.sum(np.equal(prediction, y)) / len(X)
    return right_rate


def predict(model, X):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    prediction = np.argmax(z2, axis=1)
    return prediction


def build_model(X, y,
                nn_input_dim, nn_hidden_dim, nn_output_dim,
                regularization, learning_rate, num_passes=20000):
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hidden_dim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hidden_dim))
    W2 = np.random.randn(nn_hidden_dim, nn_output_dim) / np.sqrt(nn_hidden_dim)
    b2 = np.zeros((1, nn_output_dim))

    for i in range(num_passes):
        # forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        probs = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

        # back propagation
        delta3 = probs
        delta3[range(len(X)), y] -= 1
        dW2 = a1.T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = X.T.dot(delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        # regularization
        dW2 += regularization * W2
        dW1 += regularization * W1

        # update
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if i % 1000 == 0:
            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            print("Step:{:d}, Loss:{:.3f}, Right rate:{:.3f}".
                  format(i, caculate_loss(model, regularization, X, y), evaluate(model, X, y)))

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


if __name__ == '__main__':
    basic_nn()
