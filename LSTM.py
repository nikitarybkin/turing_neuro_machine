from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import random
import numpy as np

n = 10
m = 5
sample_size = 20000


def random_array_creator(n):
    array = np.zeros((n, 1))
    for i in range(n):
        array[i] = random.randrange(-500000, 49999)
    return array


def arrays_creator(n, m):
    arrays = random_array_creator(n)
    for i in range(m - 1):
        arrays = np.concatenate((arrays, random_array_creator(n)))
    return arrays


X_train = arrays_creator(n, sample_size)
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
