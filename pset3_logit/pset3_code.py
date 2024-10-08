import pandas as pd
import numpy as np
import statsmodels.api as sm

############################################################### Q1

# create the BLP semi-elasticities matrix
data = {
    "Car Model": ["Mazda 323", "Nissan Sentra", "Ford Escort", "Chevy Cavalier", "Honda Accord", "Ford Taurus", "Buick Century", 
                  "Nissan Maxima", "Acura Legend", "Lincoln Town Car", "Cadillac Seville", "Lexus LS400", "BMW 735i"],
    "Mazda 323": [-125.933, 0.705, 0.713, 0.754, 0.12, 0.063, 0.099, 0.013, 0.004, 0.002, 0.001, 0.001, 0.0],
    "Nissan Sentra": [1.518, -115.319, 1.375, 1.414, 0.293, 0.144, 0.228, 0.046, 0.014, 0.006, 0.005, 0.003, 0.002],
    "Ford Escort": [8.954, 8.024, -106.497, 7.406, 1.59, 0.653, 1.146, 0.236, 0.083, 0.029, 0.026, 0.018, 0.009],
    "Chevy Cavalier": [9.68, 8.435, 7.57, -110.972, 1.621, 1.02, 1.7, 0.256, 0.084, 0.046, 0.035, 0.019, 0.012],
    "Honda Accord": [2.185, 2.473, 2.298, 2.291, -51.637, 2.041, 1.722, 1.293, 0.736, 0.475, 0.425, 0.302, 0.203],
    "Ford Taurus": [0.852, 0.909, 0.708, 1.083, 1.532, -43.634, 0.937, 0.768, 0.532, 0.614, 0.42, 0.185, 0.176],
    "Buick Century": [0.485, 0.516, 0.445, 0.646, 0.463, 0.335, -66.635, 0.866, 0.318, 0.21, 0.131, 0.079, 0.05],
    "Nissan Maxima": [0.056, 0.093, 0.082, 0.087, 0.31, 0.245, 0.773, -35.378, 0.506, 0.389, 0.351, 0.28, 0.19],
    "Acura Legend": [0.009, 0.015, 0.015, 0.015, 0.095, 0.091, 0.152, 0.271, -21.82, 0.28, 0.296, 0.274, 0.223],
    "Lincoln Town Car": [0.012, 0.019, 0.015, 0.023, 0.169, 0.291, 0.278, 0.579, 0.775, -20.175, 0.226, 0.168, 0.048],
    "Cadillac Seville": [0.002, 0.003, 0.003, 0.004, 0.034, 0.045, 0.039, 0.116, 0.183, 0.226, -16.313, 0.263, 0.215],
    "Lexus LS400": [0.002, 0.003, 0.003, 0.004, 0.03, 0.024, 0.029, 0.115, 0.21, 0.168, 0.263, -11.199, 0.336],
    "BMW 735i": [0.0, 0.0, 0.0, 0.0, 0.005, 0.006, 0.005, 0.02, 0.043, 0.048, 0.068, 0.086, -9.376]
}

semi_el = pd.DataFrame(data)
print(semi_el)

price_data = {
    "Car Model": ["Mazda 323", "Nissan Sentra", "Ford Escort", "Chevy Cavalier", "Honda Accord", "Ford Taurus", "Buick Century", 
                  "Nissan Maxima", "Acura Legend", "Lincoln Town Car", "Cadillac Seville", "Lexus LS400", "BMW 735i"],
    "Estimated Price (1990, USD)": [8000, 9500, 9000, 9500, 12000, 14000, 14500, 16000, 23000, 28000, 32000, 36000, 45000]
}

price_df = pd.DataFrame(price_data)

for column in semi_el.columns[1:]:
    price = price_df.loc[price_df["Car Model"] == column, "Estimated Price (1990, USD)"].values[0]
    semi_el[column] = semi_el[column] * price



############################################################### Q2

####### Compute beta hat using OLS definition ##############

df = pd.read_csv("airline.txt")

# Make dummy vars
df['day1'] = (df['DAY_OF_WEEK'] == 1).astype(int)
df['day2'] = (df['DAY_OF_WEEK'] == 2).astype(int)
df['day3'] = (df['DAY_OF_WEEK'] == 3).astype(int)
df['day4'] = (df['DAY_OF_WEEK'] == 4).astype(int)
df['day5'] = (df['DAY_OF_WEEK'] == 5).astype(int)
df['day6'] = (df['DAY_OF_WEEK'] == 6).astype(int)
df['day7'] = (df['DAY_OF_WEEK'] == 7).astype(int)

df['ones'] = 1

X = df[['ones', 'DISTANCE', 'DEP_DELAY', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6']].to_numpy()
Y = df[['ARR_DELAY']].to_numpy()

# Calculate (X'X)^-1
Xt = X.T
X_prime_X = np.dot(Xt, X)
X_prime_X_inv = np.linalg.inv(X_prime_X)

# Calculate (X'X)^-1 X'Y
beta_hat = np.dot(np.dot(X_prime_X_inv, Xt), Y)

print(beta_hat)


######## Part B ######## compute beta using minimized sum of squares
from scipy.optimize import minimize

# Define SSE function
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

optimal_beta = result.x
print(optimal_beta)


######## Part C ######## Use MLE to compute logit 

# Gen boolean for flight arriving more than 15 min late
df['late15'] = (df['ARR_DELAY'] > 15).astype(int)
df['constant'] = 1

X_c = df[['constant', 'DISTANCE', 'DEP_DELAY']].to_numpy()
y_c = df[['late15']].to_numpy()

def loglike(beta, X, y):
    likelihood = 0
    
    # Define a small value to avoid log(0)
    epsilon = 1e-10
    
    # Compute likelihood
    for i in range(len(y)):
        # Calculate p, ensuring it's within a valid range
        p = np.exp(np.dot(beta, X[i, :3])) / (1 + np.exp(np.dot(beta, X[i, :3])))
        
        # Apply epsilon to avoid taking the log of 0
        p = np.clip(p, epsilon, 1 - epsilon)

        # Compute individual observation of log likelihood
        value = y[i] * np.log(p) + (1 - y[i]) * np.log(1 - p)

        likelihood += value

    # Return the negative log likelihood for minimization
    return -likelihood

# Use scipy's minimize function to find the maximum log likelihood
initial_guess = np.zeros(3)  # Initial guess for beta
result = minimize(loglike, initial_guess, args=(X_c, y_c), method='BFGS')

# Return the parameters that maximize log likelihood
optimal_beta = result.x
print(optimal_beta)



############################################################### Q3

books = pd.read_csv("usedbooksales.csv")

# 'bnum', = book number
#  'totalprice', = list price 
#  'q', = book is gone in 2 months (sold, maybe)
#  'rank', = 
#  'ocond', = listing-specific quality, 7 is best
#  'bookseller', = name of bookstore 
#  'star', = rating of bookstore 
#  'storetitles', = number of titles bookstore lists 
#  'localint', = local interest
#  'popular', = book has large number of listed copies 
#  'bprice' = price in other offline bookstore 

## Question 3a #################### 
books['markup'] = (books['totalprice'] - books['bprice']) / books['bprice']
books['competitor_count'] = books.groupby('bnum')['bnum'].transform('count')

# Step 2: Set up regression variables
X = books[['competitor_count', 'ocond', 'popular']]  # Independent variable: competitor count
y = books['markup']  # Dependent variable: markup
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())


## Question 3d ####################
books['min_price'] = books.groupby('bnum')['totalprice'].transform('min')
books['is_min_price'] = (books['totalprice'] == books['min_price']).astype(int)

# Step 2: Rank each book's listings by price within each 'bnum'
books['price_rank'] = books.groupby('bnum')['totalprice'].rank(method='min')

X = books[['is_min_price', 'price_rank']]
X = sm.add_constant(X)
Y = books['q']

model = sm.OLS(Y, X).fit()

# Use robust SEs (HC0, HC1, HC2, HC3 for different types)
robust_model = model.get_robustcov_results(cov_type='HC3')

print(robust_model.summary())