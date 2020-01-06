import numpy as np



def computeCost(X, y, theta):
    m = len(y)
    return np.sum(np.square(np.dot(X, theta) - y)) / (2*m)


def gradientDescent(X, y, theta, alpha, iterations):
    J_history = []
    theta_history = []
    m = len(y)
    for iter in range(iterations):
        theta = theta - alpha/m * np.sum(((np.dot(X, theta) - y) * X.T).T, 0)
        J = computeCost(X, y, theta)
        J_history.append(J)
        theta_history.append(list(theta)) # I will use this to plot cost minimization path

    return theta, J_history, theta_history


def featureNormalize(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    X_norm = (X - mean) / std
    return X_norm, mean, std
    
