import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import time

np.seterr(all='raise')
df = pd.read_csv('./data/winequality-red.csv')

x_df = df.drop('quality', axis=1)
y_df = df[['quality']]

X = x_df.to_numpy()
y = y_df.to_numpy().reshape(len(X), 1)
# Scale all inputs from 0 to 1
X = X / X.max(axis=0)

m = np.size(y)
X = np.insert(X, 0, np.ones(m), axis=1)

nb_samples = X.shape[0]
border1 = int(0.8*nb_samples)
border2 = int(0.9*nb_samples)

indices = np.random.permutation(nb_samples)
train_indices = indices[:border1]
validate_indices = indices[border1:border2]
test_indices = indices[border2:]

X_train = X[train_indices]
y_train = y[train_indices]

X_validate = X[validate_indices]
y_validate = y[validate_indices]

X_test = X[test_indices]
y_test = y[test_indices]


def computecost(xx, yy, theta):

    a = 1/(2*m)
    b = np.nansum((np.dot(xx, theta).flatten()-yy.flatten())**2)
    j = (a)*(b)

    return j


def accuracy(xx, yy, theta):
    return ((np.abs(y_test.flatten() - hypothesis(theta, X_test).flatten())) < 1).sum() / y_test.shape[0]
    # return 1 - (np.abs(np.ones(y_test.shape[0]).flatten() - np.divide(y_test.flatten(), hypothesis(theta, X_test).flatten()))).mean()


def gradient_function(y_pred, y_truth):
    out_matrix = (y_pred.flatten() - y_truth.flatten())

    return out_matrix


def hypothesis(theta, xx):
    return np.dot(xx, theta)


def cost(theta, xx, yy):
    return np.sum((np.dot(xx, theta) - yy)**2)


def gradient(xx, yy, theta, iteration):

    alpha = 0.01
    J_history = np.zeros([iteration, 1])

    for iter in range(0, iteration):

        # error = np.dot(xx, theta).flatten() - yy.flatten()
        gradient = gradient_function(hypothesis(theta, xx), yy)
        theta = theta.flatten() - ((alpha/m) * (np.dot(gradient, xx)))
        # compute J value for each iteration
        J_history[iter] = (1 / (2*m)) * cost(theta, xx, yy)
    return theta, J_history


theta = np.random.rand(len(X.T), 1)
print(f'Zero computation: {computecost(X_train, y_train, theta)}')
for i in range(200):
    theta, J = gradient(X_train, y_train, theta, 100)
    print(
        f'Iteration {i} Train: {computecost(X_train, y_train, theta)}, Validate: {computecost(X_validate, y_validate, theta)}')

print(f'Test: {computecost(X_test, y_test, theta)}')
print(f'Accuracy: {accuracy(X_test, y_test, theta)*100:.1f}%')

np.savetxt(f"models/theta-{int(time.time())}.csv", theta)
