import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten, Embedding, GRU
from keras.optimizers import SGD
import numpy as np
import copy
import random
import h5py
import csv

datas = []
labels = []

with open('train.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        datas.append(row[1])
        labels.append(row[2])
del datas[0]
del labels[0]

# Convert letters to integers
label = np.zeros((2000, 1))
label = np.reshape(labels, (2000, 1))
input = np.zeros((2000, 14, 4))


def switch(letter=''):
    if letter == 'A':
        return np.array([1, 0, 0, 0])
    elif letter == 'C':
        return np.array([0, 1, 0, 0])
    elif letter == 'G':
        return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])


for i in range(2000):
    for j in range(14):
        vec = copy.copy(switch(datas[i][j]))
        input[i][j] = vec

# Initialize Network
model = Sequential()
model.add(Conv1D(64, kernel_size=3, strides=1, activation='relu', input_shape=(14, 4)))
model.add(Conv1D(128, kernel_size=3, strides=1, activation='relu', input_shape=(14, 64)))
model.add(Conv1D(256, kernel_size=3, strides=1, activation='relu', input_shape=(14, 128)))
model.add(MaxPooling1D(pool_size=3, strides=1))
model.add(GRU(512,dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(GRU(512,dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

adamx = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adamx, metrics=['accuracy'])

model.fit(input, label, epochs=3000, batch_size=100)
score, acc = model.evaluate(input, label, batch_size=100)
print('Test score:', score)
print('Test accuracy:', acc)
model.save("./model.h5")
