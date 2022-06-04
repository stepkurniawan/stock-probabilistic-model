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

# randomly generated 2 stocks
num_stocks = 2

# we have 3 states: 
# state 0: crisis, state 1: stable, state 2: bubble
num_state = 3

#fix the random seed
np.random.seed(0)

# b : drift for each stocks, for each state
# ex: b[0,1] is drift for stock 0 and state 1
b = np.random.randn(num_stocks, num_state)

# b0 < b1 < b2
for i in range(num_stocks):
    for j in range(num_state):
        if i == 0 and j == 0:
            b[i,j] = np.random.randn(1)
        else: 
            # make sure b of the next state is larger than the one before
            # b0 (crisis period) < b1 (stable) < b2 (bubble periode)
            b[i,j] = np.random.normal(b[i,j-1] +1, 0.5,1) 

print(b)

# volatility

# transition matrixs 
# P_1 : transition matrix for stock 1
# P_2 : transition matrix for stock 2
P_1 = [[0.3, 0.4, 0.3],
       [0.1, 0.9, 0.0],
       [0.4, 0.3, 0.3]]

P_2 = [[0.2, 0.6, 0.2],
       [0.01, 0.9, 0.09],
       [0.4, 0.3, 0.3]]
