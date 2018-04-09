import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
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
read=np.column_stack((datas,labels))
np.random.shuffle(read)
data_p=read[:,0]
label_p=read[:,1]
datas,y_train,labels,y_test=train_test_split(data_p,label_p,test_size=0.2)
print(y_train.shape)
# Convert letters to integers
label = np.zeros((1600, 1))
label = np.reshape(labels, (1600, 1))
input = np.zeros((1600, 14, 4))
input2 = np.zeros((400, 14, 4))

def switch(letter=''):
    if letter == 'A':
        return np.array([1, 0, 0, 0])
    elif letter == 'C':
        return np.array([0, 1, 0, 0])
    elif letter == 'G':
        return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])


for i in range(1600):
    for j in range(14):
        vec = copy.copy(switch(datas[i][j]))
        input[i][j] = vec

for i in range(400):
    for j in range(14):
        vec2 = copy.copy(switch(y_train[i][j]))
        input2[i][j] = vec
max_features=14

# Initialize Network
model = Sequential()
'''
model.add(Conv1D(32, kernel_size=3, strides=1, activation='relu', input_shape=(14, 4)))
model.add(Conv1D(64, kernel_size=3, strides=1, activation='relu', input_shape=(14, 32)))
model.add(Conv1D(128, kernel_size=3, strides=1, activation='relu', input_shape=(14, 64)))
model.add(Conv1D(256, kernel_size=3, strides=1, activation='relu', input_shape=(14, 128)))
model.add(MaxPooling1D(pool_size=3, strides=1))
model.add(GRU(512,dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(GRU(512,dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
'''
#model.add(Embedding(14,4))

model.add(LSTM(128,return_sequences=True,input_shape=(14,4)))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128))
model.add(Dense(64, activation='relu', input_dim=14))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
adamx = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(input, label, epochs=1000, batch_size=100)
score, acc = model.evaluate(input2,y_test, batch_size=100)
print('Test score:', score)
print('Test accuracy:', acc)
model.save("./model4.h5")
