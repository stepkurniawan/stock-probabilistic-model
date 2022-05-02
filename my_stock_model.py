# -*- coding: utf-8 -*-
"""
Created on Mon May  2 21:24:58 2022

@author: steph
"""

import numpy as np
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials

DAX = '^GDAXI'
CAC = '^FCHI'
FTSE100 = '^FTSE'
NIKKEI = '^N225'
SNP500 = '^GSPC'
NASDAQ = '^IXIC'
DOW = '^DJI'

stock_indices = [DAX, CAC, FTSE100, NIKKEI, SNP500, NASDAQ, DOW]

# %%

aapl_df = yf.download('AAPL', 
                      start='1991-01-01', 
                      end='2011-12-01', 
                      progress=False,
)
aapl_df.head()

# %%

# %%

dax_df = yf.download(DAX,
                     start='1991-01-01',
                     end='2011-12-25',
                     progress=False,
                     interval='1wk')
dax_df.head()

ticker = yf.Ticker(DAX)

dax_df = ticker.history(start='1991-01-01',
                        end='2011-12-25',
                        progress=False,
                        interval='1wk')

dax_df['Close'].plot(title='DAX Stock Price')#


def stock_return(arr):
    r = np.array(1)
    
    for i in range(1, len(arr['Close'])):
        r_t = 100*np.log(arr['Close'][i] / arr['Close'][i-1])
        # print(r_t)
        
    np.append(r, r_t) # TODO fix this
    return r

print(stock_return(dax_df))
    