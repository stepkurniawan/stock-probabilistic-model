# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 12:56:02 2022

@author: steph

Our simulation consists of 2 stocks 
we simulate their drift, volatility, and transition matrix
and plug them in max likelihood
before testing using MCMCs
"""

import numpy as np
import pandas as pd

# drift = normal distribution(mean, sigma)
# dim(drift) = [stock] [state] 
# list of mean of stocks : [stock1, stock2]
mean_stocks = [0,0]
# list of sigma of stocks : [stock1, stock2]
sigma_stock = [0,0]

num_stocks = 2
num_state = 3

# create zero matrix
b = np.zeros((num_stocks, num_state))

for i in range(num_stocks):
    for j in range(num_state):
        b[i][j] = np.random.normal(mean_stocks[i], sigma_stock[j])


print(b)

# volatility

# transition matrix