#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import *


# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m
# print('Running warmUpExercise ...');
# print('5x5 Identity Matrix:');
# print(np.eye(5))
# input("Program paused. Press enter to continue.")


# ======================= Part 2: Plotting =======================
# print('Plotting Data ...')
data = np.genfromtxt('./ex1/ex1data1.txt', delimiter=',')
x = data[:,0]
y = data[:,1]
xmin = min(x)
xmax = max(x)
xrange = xmax - xmin
m = len(x)
plt.scatter(x, y, s=4)
plt.xlabel('population')
plt.ylabel('revenue')
plt.title('starting data set')
plt.show()

# =================== Part 3: Cost and Gradient descent ===================
X = np.vstack([np.ones(m), x]).T
theta = np.zeros(2)

# Some gradient descent settings
num_iters = 1500
alpha = 0.01

# cost function:
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = %f' % J)
print('Expected cost value (approx) 32.07')
# further testing of the cost function
J = computeCost(X, y, [-1, 2])
print('With theta = [-1 2]\nCost computed = %f' % J)

# run gradient descent
print('\nRunning Gradient Descent ...\n')
theta, J_history, theta_history = gradientDescent(X, y, theta, alpha, num_iters)
theta_history = np.array(theta_history) # this will be used for plotting
# Plot how the cost function changes
plt.plot(J_history)
plt.title('Cost function convergence')
plt.xlabel('iterations')
plt.ylabel('cost function')
plt.show()

# Plot the lineat fit of the data
plt.scatter(x, y, s=4)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = theta[0] + theta[1] * x_vals
plt.plot(x_vals, y_vals, '--', color="red")
plt.title('Original data and linear fit')
plt.xlabel('population')
plt.ylabel('revenue')
#plt.legend(['original data', 'fit'], loc = "lower right")
plt.legend(['fit', 'original data'], loc = "lower right")
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of %f' % (predict1*10000))
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of %f' % (predict2*10000))

# ============= Part 4: Visualizing J(theta_0, theta_1) =============

NX = 30
NY = 30
xx = np.linspace(-4,  1,  NX)
yy = np.linspace(0.5, 1.3, NY)
XX, YY = np.meshgrid(xx,yy)
myx = np.reshape(XX, NX*NY, order = 'C')
myy = np.reshape(YY, NX*NY, order = 'C')
myJJ = []

for i in range(NX*NY):
    myJJ.append(computeCost(X, y, [myx[i], myy[i]]))

myJJ = np.reshape(myJJ, (NY, NX))

fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')

# Plot the Theta0 - Theta1 surface
ax.plot_surface(XX, YY, myJJ, rstride=1, cstride=1, cmap='ocean')
# plot scatter plot of J_history vs corresponding values of theta
# instead of plotting all of minimization trajectory, let me take only every p'th point:
p = 100
plt.plot(theta_history[::p,0], theta_history[::p,1], J_history[::p], '.', ms = 5, color = 'red')

plt.xlabel(r'$\theta_0$', fontsize=20)
plt.ylabel(r'$\theta_1$', fontsize=20)
plt.title(r'Cost minimization path on the $\theta_0$ - $\theta_1$ surface')
plt.show()

# ============= Part 5: Mutlivariate linear regression ===========
data = np.genfromtxt('./ex1/ex1data2.txt', delimiter=',')
X = data[:, [0,1]] # these are 'features', first two columns
#X = np.array([np.ones(m), X])
y = data[:, 2] # prices
m = len(y)

X_norm, xmean, xstd = featureNormalize(X)
# now add "zero'th" feature to the normalized feature matrix
F = np.hstack([np.ones([X_norm.shape[0], 1]), X_norm])

alpha = 0.01;
num_iters = 400;

# initial value of thetas 
theta = np.zeros(3)

# run gradientDescent with 3 different learning rates and plot convergence
# all the theta vectors should be very close to each other
alpha1 = 0.3
alpha2 = 0.1
alpha3 = 0.03
alpha4 = 0.01
theta1, J_history1, theta_history = gradientDescent(F, y, np.zeros(3), alpha1, num_iters)
theta2, J_history2, theta_history = gradientDescent(F, y, np.zeros(3), alpha2, num_iters)
theta3, J_history3, theta_history = gradientDescent(F, y, np.zeros(3), alpha3, num_iters)
theta4, J_history4, theta_history = gradientDescent(F, y, np.zeros(3), alpha4, num_iters)
plt.plot(J_history1)
plt.plot(J_history2)
plt.plot(J_history3)
plt.plot(J_history4)
plt.title(r'Convergence for different learning rates $\alpha$')
plt.legend([r'$\alpha = 0.3$', r'$\alpha = 0.1$', r'$\alpha = 0.03$', r'$\alpha = 0.01$'])
plt.ylabel("Cost Function")
plt.xlabel("# of iteration")
plt.show()


 # Estimate the price of a 1650 sq-ft, 3 br house
price = theta1[0] + theta1[1]*(1500-xmean[0])/xstd[0] + theta1[2]*(3-xmean[1])/xstd[1]
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n' % price);


# ============= Part 6: Normal Equation ===========
# reminder:
# F is m x 3 matrix, therefore
# F.T is 3 x m
# y is m-long vector
print('Using normal equation approach')
A = np.dot(F.T, F) # gives 3 x 3 matrix
A1 = np.linalg.inv(A) # still  a 3 x 3
B = np.dot(A1, F.T)  # 3 x m matrix 
theta = np.dot(B, y) # finally theta here is vector with 3 components
# this value of theta should be the same as what was found using the gradient descent method
print('Difference between normal equation solution and theta found from gradient descent (should be negligible):')
print(theta - theta1)
