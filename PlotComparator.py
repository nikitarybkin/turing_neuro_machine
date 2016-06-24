import matplotlib.pyplot as plt
import numpy as np


def compare (X, Y):
    c = 0
    if(X.ndim == 1):
        for i in range(X.shape[0]):
                if (X[i] == Y[i]):
                    c += 1.0
        return c / X.shape[0]
    for i in range(X.shape[0]):
        for j in range (X.shape[1]):
             if (X[i,j] == Y[i,j]):
                 c += 1.0
    return c / (X.shape[0] * X.shape[1])

def vectors_compare (X, Y):
    C = np.zeros((1, X.shape[0]))
    for i in range(X.shape[0]):
        C[0, i] = compare(X[i], Y[i])
    return C[0]

def plot_compare (X_test, Y_test):
    Y = vectors_compare(X_test, Y_test)
    X = np.zeros ((1, X_test.shape[0]))
    X = X[0]
    for i in range (X.shape[0]):
        X[i] = i+1
    line = plt.plot(X, Y)
    plt.legend((line), (u'Similarity \u00b0C'), loc='best')
    plt.grid()
    plt.savefig('similarity.png', format='png')
    return Y


print plot_compare(np.array([[1,0], [2,1], [1,1], [3,3], [5,5], [1,1]]), np.array([[1,1], [0,4], [1,1], [3,3], [5,0], [2,3]]))

n = 10
sample_size = 20
