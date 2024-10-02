import pandas as pd
import numpy as np

df = pd.read_csv("airline.txt")

####### Compute beta hat using OLS definition ##############

# Make dummy vars
df['day1'] = (df['DAY_OF_WEEK'] == 1).astype(int)
df['day2'] = (df['DAY_OF_WEEK'] == 2).astype(int)
df['day3'] = (df['DAY_OF_WEEK'] == 3).astype(int)
df['day4'] = (df['DAY_OF_WEEK'] == 4).astype(int)
df['day5'] = (df['DAY_OF_WEEK'] == 5).astype(int)
df['day6'] = (df['DAY_OF_WEEK'] == 6).astype(int)
df['day7'] = (df['DAY_OF_WEEK'] == 7).astype(int)

df['ones'] = 1

# Define X matrix and Y 
X = df[['ones', 'DISTANCE', 'DEP_DELAY', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6']].to_numpy()
Y = df[['ARR_DELAY']].to_numpy()

# Calculate (X'X)^-1
Xt = X.T
X_prime_X = np.dot(Xt, X)
X_prime_X_inv = np.linalg.inv(X_prime_X)

# Calculate (X'X)^-1 X'Y
beta_hat = np.dot(np.dot(X_prime_X_inv, Xt), Y)

# Print the result
print(beta_hat)


######## Part B ######## compute beta using minimized sum of squares
from scipy.optimize import minimize

# Define the SSE (sum of squared errors) function
def sse(beta, Y, X):
    # Initialize error_sum at 0
    error_sum = 0
    
    # Loop over all observations and calculate SSE
    for i in range(len(Y)):
        error = (Y[i] - np.dot(X[i, :9], beta)) ** 2
        error_sum += error

    return error_sum

# Use scipy's minimize to find optimal beta
initial_guess = np.zeros(9)  # Initial guess for beta-hat
result = minimize(sse, initial_guess, args=(Y, X))

# Return beta-hat that minimizes SSE
optimal_beta = result.x
print(optimal_beta)

######## Part C ######## Use MLE to compute logit 

# Generate a binary variable for a flight arriving more than 15 minutes late
df['late15'] = (df['ARR_DELAY'] > 15).astype(int)
df['constant'] = 1

# Define the X matrix and y vector
X_c = df[['constant', 'DISTANCE', 'DEP_DELAY']].to_numpy()
y_c = df[['late15']].to_numpy()

# Define the log-likelihood function
def loglike(beta, X, y):
    # Initialize likelihood at 0
    likelihood = 0

    # Loop over all observations and compute the likelihood
    for i in range(len(y)):
        # Calculate p
        p = np.exp(np.dot(beta, X[i, :3])) / (1 + np.exp(np.dot(beta, X[i, :3])))

        # Compute individual observation of log likelihood
        value = y[i] * np.log(p) + (1 - y[i]) * np.log(1 - p)

        # Add the value to the likelihood
        likelihood += value

    # Return the negative log likelihood for minimization
    return -likelihood

# Use scipy's minimize function to find the maximum log likelihood
initial_guess = np.zeros(3)  # Initial guess for beta
result = minimize(loglike, initial_guess, args=(X_c, y_c), method='BFGS')

# Return the parameters that maximize log likelihood
optimal_beta = result.x
print(optimal_beta)

