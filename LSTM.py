from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from random import randint
import numpy as np

n = 10
m = 5


def array_creator(n):
    array = np.zeros(n)
    for i in range(n):
        array[i] = randint(-49999, 50000)
    return array


model = Sequential()
# model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(n * m))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(X_train, Y_train, batch_size=1000, nb_epoch = 1, show_accuracy = True)

# result = model.predict_proba(X)
