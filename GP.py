import numpy as np
import matplotlib.pylab as plt

# input x and y training points and calculate correlation function
x = np.array([0, 1/6, 1/3, 1/2, 2/3, 5/6, 1])
y = np.array([1, -0.20, -0.55, 0.35, 0.20, -0.30, 0])
θ = np.array([1.5, 40])
# σ = 1.0 and ω = 20

# Define covariance function
def exponential_cov(x, y, params):
    return (params[0]**2) * np.exp( -params[1] * np.subtract.outer(x, y)**2)

'''
def conditional(x_new, X, Y, params):
 
    B = exponential_cov(x_new, X, params)
    C = exponential_cov(X, X, params)
    A = exponential_cov(x_new, x_new, params)
 
    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))
 
    return(mu.squeeze(), sigma.squeeze())
'''

def predict(Q, data, kernel, S, sigma, t):
    k = [kernel(Q, R, S) for R in data]
    Sinv = np.linalg.inv(sigma)
    y_pred1 = np.dot(k, Sinv).dot(t)
    sigma_new = kernel(Q, Q, S) - np.dot(k, Sinv).dot(k)
    return y_pred1.squeeze(), sigma_new.squeeze()

σ_1 = exponential_cov(x, x, θ)
x_pred = np.linspace(0, 1, 1000)

predictions = [predict(i, x, exponential_cov, θ, σ_1, y) for i in x_pred]
    

y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")


"""
Spyder Editor

This is a temporary script file.
"""

