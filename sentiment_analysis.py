from load_data import get_inputs_and_outputs
import tensorflow as tf
import spacy
import numpy as np
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


file_paths = ["/users/kaysonhansen/cs129/HotelReviewData/TestFile.csv"]

X, Y = get_inputs_and_outputs(file_paths)
m = X.shape[0]
n = X.shape[1]

# splits are 60/20/20 train/cross-validation/test
m1 = int(m * 3/5)
m2 = int(m * 4/5)
x_train = X[:m1, :]
x_cv = X[m1:m2, :]
x_test = X[m2:m, :]
y_train = Y[:m1, :]
y_cv = Y[m1:m2, :]
y_test = Y[m2:m, :]

print(m)
print(n)

# Tensorflow Neural Network
model = Sequential(
    [
        Dense(units=n, activation='sigmoid', name='layer1'),
        Dense(units=(n+1) / 2, activation='sigmoid', name='layer2'),
        Dense(units=1, activation='linear', name='layer3')
    ]
)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=[tf.keras.metrics.Accuracy()]
)

model.fit(
    x_train, y_train,
    epochs=10, batch_size=13
)

print(model.evaluate(x_test, y_test, batch_size=13))
