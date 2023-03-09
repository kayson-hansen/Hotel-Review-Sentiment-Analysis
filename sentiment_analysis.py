from load_data import get_outputs
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.regularizers import L2
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def create_inputs_and_outputs(input_file, output_files, multiclass, shuffle=True):
    """ Takes a .npy input file of sentence embeddings and a list of files
    containing the reviews to get outputs from and generates numpy arrays X and Y.
    Called by evaluate_model.

    Args:
        input_file (String): a .npy file containing the sentence embeddings for the input reviews
        output_files (List[String]): a list containing the filenames that have all the input reviews and ratings
        multiclass (Bool): true if we want to return the rating out of 5, false if we want to
        return a positive or negative sentiment (1 or 0, respectively)

    Returns:
        X (np.ndarray), Y (np.ndarray): the inputs and outputs for the model
    """
    embeddings = np.load(input_file)
    ratings = get_outputs(output_files, multiclass)

    m = embeddings.shape[0]
    n = embeddings.shape[1]
    reviews = []
    for i in range(m):
        reviews.append((embeddings[i], ratings[i]))

    # randomize data points so that each dataset is more evenly distributed across reviews from each hotel
    if shuffle:
        random.seed(10)
        random.shuffle(reviews)

    X = np.zeros((m, n))
    Y = np.zeros((m, 1))
    for i in range(m):
        X[i] = reviews[i][0]
        Y[i] = reviews[i][1]

    return X, Y


def train_neural_network_model(x_train, y_train, learning_rate=0.05, epochs=3, batch_size=64):
    """ Given a set of inputs and outputs, train a neural network model using tensorflow.
    Called by evaluate_model.

    Args:
        x_train (np.ndarray): train set inputs
        y_train (np.ndarray): train set outputs
        learning_rate (Float): learning rate of the model
        epochs (Int): number of epochs to train the model
        batch_size (Int): size of each training batch

    Returns:
        (tf.keras.model): trained neural network model
    """
    # n = number of input features
    n = x_train.shape[1]

    model = Sequential(
        [
            InputLayer(input_shape=(n, )),
            Dense(units=48, activation='relu', name='layer2'),
            Dense(units=20, activation='relu', name='layer3'),
            Dense(units=10, activation='relu', name='layer4'),
            Dense(units=1, activation='sigmoid', name='output')
        ]
    )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )

    model.fit(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size
    )

    return model


def train_softmax_neural_network_model(x_train, y_train, learning_rate=0.05, epochs=3, batch_size=64):
    """ Given a set of inputs and outputs, train a neural network model using tensorflow.
    Called by evaluate_model.

    Args:
        x_train (np.ndarray): train set inputs
        y_train (np.ndarray): train set outputs
        learning_rate (Float): learning rate of the model
        epochs (Int): number of epochs to train the model
        batch_size (Int): size of each training batch

    Returns:
        (tf.keras.model): trained neural network model
    """
    # n = number of input features
    n = x_train.shape[1]

    model = Sequential(
        [
            InputLayer(input_shape=(n, )),
            Dense(units=48, activation='relu', name='layer2'),
            Dense(units=20, activation='relu', name='layer3'),
            Dense(units=10, activation='relu', name='layer4'),
            Dense(units=5, activation='linear', name='output')
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )

    model.fit(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size
    )

    return model


def train_logistic_regression_model(x_train, y_train, learning_rate=0.05, epochs=3, batch_size=64):
    """ Given a set of inputs and outputs, train a logistic regression model using tensorflow.
    Called by evaluate_model.

    Args:
        x_train (np.ndarray): train set inputs
        y_train (np.ndarray): train set outputs
        learning_rate (Float): learning rate of the model
        epochs (Int): number of epochs to train the model
        batch_size (Int): size of each training batch

    Returns:
        (tf.keras.model): trained logistic regression model
    """
    # n = number of input features
    n = x_train.shape[1]

    model = Sequential(
        [
            InputLayer(input_shape=(n, )),
            Dense(units=1, activation='sigmoid',
                  name='output')
        ]
    )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )

    model.fit(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size
    )

    return model


def train_softmax_logistic_regression_model(x_train, y_train, learning_rate=0.05, epochs=3, batch_size=64):
    """ Given a set of inputs and outputs, train a logistic regression model using tensorflow.
    Called by evaluate_model.

    Args:
        x_train (np.ndarray): train set inputs
        y_train (np.ndarray): train set outputs
        learning_rate (Float): learning rate of the model
        epochs (Int): number of epochs to train the model
        batch_size (Int): size of each training batch

    Returns:
        (tf.keras.model): trained logistic regression model
    """
    # n = number of input features
    n = x_train.shape[1]

    model = Sequential(
        [
            InputLayer(input_shape=(n, )),
            Dense(units=5, activation='linear',
                  name='output')
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )

    model.fit(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size
    )

    return model


def evaluate_model(input_filename, output_filenames, algorithm, softmax, confusion_matrix=False):
    """ Given an .npy file containing sentence embeddings as inputs, a list of files containing
    the reviews and ratings, and an algorithm to use, train and evaluate the accuracy, precision,
    and recall of the given model. Also generates a confusion matrix.

    Args:
        input_filename (String): a .npy file containing all the sentence embeddings for each review
        output_filenames (List_String): a list of files containing the reviews and ratings
        algorithm (String): which type of model to train, either "neural network" or "logistic regression"
        softmax (Bool): True if the model is softmax, false if it's binary
        confusion_matrix (Bool): True if we wnat to generate a confusion matrix graph, False otherwise
    """
    X, Y = create_inputs_and_outputs(
        input_filename, output_filenames, softmax)
    m = X.shape[0]

    # splits are 80/10/10 train/cross-validation/test
    m1 = int(m * 8/10)
    m2 = int(m * 9/10)
    x_train = X[:m1, :]
    x_cv = X[m1:m2, :]
    x_test = X[m2:m, :]
    y_train = Y[:m1, :]

    if algorithm == 'neural network' and softmax:
        model = train_softmax_neural_network_model(x_train, y_train)
    elif algorithm == 'neural network' and not softmax:
        model = train_neural_network_model(x_train, y_train)
    elif algorithm == 'logistic regression' and softmax:
        model = train_softmax_logistic_regression_model(x_train, y_train)
    else:
        model = train_logistic_regression_model(x_train, y_train)

    # if the output is multiclass, add 1 to go from 0-4 star ratings to 1-5 star ratings
    if softmax:
        Y += 1
    y_train = Y[:m1, :]
    y_cv = Y[m1:m2, :]
    y_test = Y[m2:m, :]

    # evaluate model on train set
    train_set_predictions = model.predict(x_train)
    train_set_yhat = np.zeros_like(y_train)
    for i in range(len(x_train)):
        if softmax:
            # add 1 to go from 0-4 star ratings to 1-5 star ratings
            train_set_yhat[i] = np.argmax(train_set_predictions[i]) + 1
        else:
            if train_set_predictions[i] >= 0.5:
                train_set_yhat[i] = 1
            else:
                train_set_yhat[i] = 0
    print("Train set accuracy: ", accuracy_score(y_train, train_set_yhat))
    if not softmax:
        print("Train set precision: ", precision_score(y_train, train_set_yhat))
        print("Train set recall: ", recall_score(y_train, train_set_yhat))

    # evaluate model on cross-validation set
    cv_set_predictions = model.predict(x_cv)
    cv_set_yhat = np.zeros_like(y_cv)
    for i in range(len(x_cv)):
        if softmax:
            # add 1 to go from 0-4 star ratings to 1-5 star ratings
            cv_set_yhat[i] = np.argmax(cv_set_predictions[i]) + 1
        else:
            if cv_set_predictions[i] >= 0.5:
                cv_set_yhat[i] = 1
            else:
                cv_set_yhat[i] = 0
    print("Cross-validation set accuracy: ", accuracy_score(y_cv, cv_set_yhat))
    if not softmax:
        print("Cross-validation set precision: ",
              precision_score(y_cv, cv_set_yhat))
        print("Cross-validation set recall: ", recall_score(y_cv, cv_set_yhat))

    # evaluate model on test set
    test_set_predictions = model.predict(x_test)
    test_set_yhat = np.zeros_like(y_test)
    for i in range(len(x_test)):
        if softmax:
            # add 1 to go from 0-4 star ratings to 1-5 star ratings
            test_set_yhat[i] = np.argmax(test_set_predictions[i]) + 1
        else:
            if test_set_predictions[i] >= 0.5:
                test_set_yhat[i] = 1
            else:
                test_set_yhat[i] = 0
    print("Test set accuracy: ", accuracy_score(y_test, test_set_yhat))
    if not softmax:
        print("Test set precision: ", precision_score(y_test, test_set_yhat))
        print("Test set recall: ", recall_score(y_test, test_set_yhat))

    if confusion_matrix:
        generate_confusion_matrix(y_cv, cv_set_yhat)


def generate_confusion_matrix(y_cv, cv_set_yhat):
    cm = confusion_matrix(y_cv, cv_set_yhat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Cross-validation set confusion matrix')
    plt.xlabel('Predicted sentiment')
    plt.ylabel('True sentiment')
    plt.show()


# iterate over batch size, epochs, learning rate, and regularization parameters
# to find the best ones; also graph accuracy vs each of these parameters
def training_loop(input_filename, output_filenames, algorithm, softmax, metric):
    """ Given an .npy file containing sentence embeddings as inputs, a list of files containing
    the reviews and ratings, and an algorithm to use, train and evaluate the accuracy of the
    given model for a set of learning parameters, numbers of epochs, and batch sizes.

    Args:
        input_filename (String): a .npy file containing all the sentence embeddings for each review
        output_filenames (List_String): a list of files containing the reviews and ratings
        algorithm (String): which type of model to train, either "neural network" or "logistic regression"
        softmax (Bool): True if the model is softmax, false if it's binary
        metric (String): metric we want to iterate over to find its optimal value
    """
    X, Y = create_inputs_and_outputs(
        input_filename, output_filenames, softmax)
    m = X.shape[0]

    # splits are 80/10/10 train/cross-validation/test
    m1 = int(m * 8/10)
    m2 = int(m * 9/10)
    x_train = X[:m1, :]
    x_cv = X[m1:m2, :]
    y_train = Y[:m1, :]

    scores = {}
    learning_rate = 0.01
    num_epochs = 3
    batch_size = 32
    for i in range(10):
        if algorithm == 'neural network' and softmax:
            model = train_softmax_neural_network_model(
                x_train, y_train, learning_rate, num_epochs, batch_size)
        elif algorithm == 'neural network' and not softmax:
            model = train_neural_network_model(
                x_train, y_train, learning_rate, num_epochs, batch_size)
        elif algorithm == 'logistic regression' and softmax:
            model = train_softmax_logistic_regression_model(
                x_train, y_train, learning_rate, num_epochs, batch_size)
        else:
            model = train_logistic_regression_model(
                x_train, y_train, learning_rate, num_epochs, batch_size)

        # add 1 to go from 0-4 star ratings to 1-5 star ratings
        Y += 1
        y_cv = Y[m1:m2, :]

        # evaluate model on train set
        cv_set_predictions = model.predict(x_cv)
        cv_set_yhat = np.zeros_like(y_cv)
        for i in range(len(x_cv)):
            if softmax:
                # add 1 to go from 0-4 star ratings to 1-5 star ratings
                cv_set_yhat[i] = np.argmax(cv_set_predictions[i]) + 1
            else:
                if cv_set_predictions[i] >= 0.5:
                    cv_set_yhat[i] = 1
                else:
                    cv_set_yhat[i] = 0

        accuracy = accuracy_score(y_cv, cv_set_yhat)
        scores[i] = accuracy

    print(scores)
