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

##############################################################################
### VARIABLES ###
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

# how many time (index of time)
N = 50

#fix the random seed
# np.random.seed(0)
#############################################################################
# b : drift for each stocks, for each state
# ex: b[0,1] is drift for stock 0 and state 1
b = np.random.randn(num_stocks, num_state)

# b0 < b1 < b2
for j in range(num_stocks):
    for i in range(num_state):
        if i == 0:
            b[j,i] = np.random.randn(1)
        else: 
            # make sure b of the next state is larger than the one before
            # b0 (crisis period) < b1 (stable) < b2 (bubble periode)
            b[j,i] = np.random.normal(b[j,i-1] +1, 0.5,1) 


print("drift matrix",b)

# volatility
sigma = [
    np.full((num_stocks, num_stocks), 0.2 if x == 1 else 0.5, dtype=np.float32)
    for x in range(num_state)
]
for x in sigma:
    np.fill_diagonal(x, 1)
sigma
# create 3 matrix of dxd matrix
# for state 1 & 3 -> similar covariance matrix

# transition matrixs 
# P [stock][from state][to state]
P = [[0.3, 0.4, 0.3], [0.1, 0.9, 0.0], [0.4, 0.3, 0.3]], \
    [[0.2, 0.6, 0.2], [0.01, 0.9, 0.09], [0.4, 0.3, 0.3]]

# simulate y : state matrix
# t = 0 : first state : random between state 1,2,3
# y[i,t] : stock i, and t time
# ex: y[0,1] : state at stock 0 when time is in index 1.  

y = np.zeros((num_stocks, N))

for i in range(num_stocks):
    for t in range(N):
        if t == 0:
            y[i,t] = np.random.randint(0,3)
        else: 
            prev_state = int(y[i, t-1])
            y[i,t] = np.random.choice( np.arange(0,3), p=[P[i][prev_state][0], \
                                                          P[i][prev_state][1], \
                                                          P[i][prev_state][2]] )
y = y.astype('int') # state is always integer
print("state matrix", y)

# DELTA of BROWNIAN MOTION
# 1 stock have 1 brownian motion
# the length of 1 brownian motion is N
# brownian motion have dim(num_stock, N)

brownian_motion_delta = np.zeros((num_stocks, N))
for j in range(num_stocks): 
    brownian_motion_delta[j] = np.random.standard_normal(N)
    

# r : rate of Return of stock price from timme 0 to time t
r = np.zeros((num_stocks, N))
for i in range(num_stocks):
    for n in range(1,N+1):
        sum_b = 0
        sum_sigma = 0
        for t in range(n):
            state_now = y[i,t]
            sum_b = sum_b + b[i , state_now]
            for d in range(num_stocks):
                sum_sigma = sum_sigma + (sigma[state_now][i][d] * brownian_motion_delta[i, t])
            r[i,t] = sum_b + sum_sigma
print("return matrix", r)

# Price matrix S
# dimension: number_of_stock x N times
# ex: S[0,1] : stock 0 , time 1
S = np.zeros((num_stocks, N))
for i in range(num_stocks):
    for n in range(N):
        if n == 0:
            S[i,n] = np.random.randint(50,100)
        else:
            S[i,n] = S[i, n-1]*r[i,n]
print("price matrix", S)

