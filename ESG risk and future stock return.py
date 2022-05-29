#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Relation between ESG risk and future stock return. Adopting
# RepRisk Index (RRI) as the proxy for ESG risk, test the relation between ESG risk and
# future 1-month stock return with control variables (Student should choose their own
# control variables)

# Step 1: CRSP Block

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats
from finance_byu.fama_macbeth import fama_macbeth, fama_macbeth_parallel, fm_summary, fama_macbeth_numba

# import the crsp data
crsp_m = pd.read_csv(r'C:\Users\PC\coding\finance\2. data_files\crsp.csv')

# Change variable format to integer
crsp_m[['PERMNO']] = crsp_m[['PERMNO']].astype(int)

# covert date to pandas date format
crsp_m['date'] = crsp_m['date'].astype(str)
crsp_m['date'] = pd.to_datetime(crsp_m['date'])

# sort the data
crsp_m = crsp_m.sort_values(['PERMNO', 'date'])

# get month and quarter-end dates
crsp_m['mdate'] = crsp_m['date'] + MonthEnd(0)

# Only keep common shares --> share code=10 and 11, because institutional
# investors are only required to keep the common shares not any other type.
crsp_m = crsp_m[crsp_m['SHRCD'].isin([10, 11])]

# Remove string value from numerical return column (e.g., B and C) --> They represent errors
crsp_m['RET'] = pd.to_numeric(crsp_m['RET'], errors='coerce')  # Covert string value to missing value

# Generate past 12 months return
tmp_crsp = crsp_m[['PERMNO', 'date', 'RET']].sort_values(['PERMNO', 'date']).set_index('date')  # Set Index, in here

# Drop missing return
tmp_crsp = tmp_crsp[tmp_crsp['RET'].notnull()]

# Calculate log return
tmp_crsp['logret'] = np.log(1 + tmp_crsp['RET'])  # Rt = ln(Pt/Pt-1)

# For last 12 month and require 12 non-missing return(This is a rolling window basis)
ret = tmp_crsp.groupby(['PERMNO'])['logret'].rolling(12, min_periods=12).sum().reset_index()

# Cumulative return for past 12 month
ret['cumret'] = np.exp(ret['logret']) - 1  # cumulative return --> e^(ln(Pt+j/Pt)-1 = Pt+j/Pt-1

# Merge the data to crisp_m
crsp_m = pd.merge(crsp_m, ret[['PERMNO', 'date', 'cumret']], how='left', on=['PERMNO',
                                                                             'date'])  # left merge merging on the basis of the file in the beginning ie CRSPm here the next is the file from where you merge the data and the how tells what do you use as a common point or the basis to merge

# calculate adjusted price, total shares and market cap --> adjusted by stock split factors
crsp_m['p'] = crsp_m['PRC'].abs() / crsp_m['CFACPR']  # price adjusted
crsp_m['tso'] = crsp_m['SHROUT'] * crsp_m['CFACSHR'] * 1000  # total shares out adjusted
crsp_m['me'] = crsp_m['p'] * crsp_m['tso'] / 1e6  # market cap in $million

# identify the last month (i.e., 3, 6, 9, 12) of each quarter to make sure that we have end-quarter data. Institutional investors are required to report their holdings at the end of each quarter, so we can keep the data, at the end of each quarter.
crsp_m['year'] = crsp_m['mdate'].dt.year
crsp_m['month'] = crsp_m['mdate'].dt.month

# Keep only relevant variables
crsp_mend = crsp_m[['PERMNO', 'year', 'month', 'date', 'NCUSIP', 'CFACSHR', 'p', 'tso', 'me', 'cumret']]

# save memory
crsp_m = None
ret = None
tmp_crsp = None
filt_lastmonth = None


# In[4]:


#  Step 2 of Task 2: Calculate control variables and merge

# Import the compustat_crsp merged data and calculate the control variables
comp = pd.read_csv(r'C:\Users\PC\coding\finance\2. data_files\Crsp_Compmerg.csv')

# Note that this time we merge based on fiscal year rather than the calendar year of datadate (as factor models)
# Drop if missing fiscal year
comp = comp[comp['fyear'].notnull()]
comp['fyear'] = comp['fyear'].astype(int)

# rename for matching
comp = comp.rename(columns={'LPERMNO': 'PERMNO', 'fyear': 'year'})

# Construct control variables
comp['log_TA'] = np.log(comp['at'] * 1e6)  # Log of total asset
comp['tang'] = comp['ppent'] / comp['at']  # Tangibility
comp['q'] = (comp['seq'] + comp['dltt'] + comp['dlc']) / comp['at']  # Tobin's q
comp['gp'] = comp['revt'] - comp['cogs']  # Gross Profitability
comp['REVGR'] = comp['ni'].pct_change().fillna(0)  # Revenue Growth


# In[7]:


# Step 3: Calculate quarterly average RRI index

#  import all the data from reprisk index (RRI)
RRI = pd.read_csv(r'C:\Users\PC\coding\finance\2. data_files\RRI.csv')

# sort the data
RRI = RRI.sort_values(by=['RepRisk_ID', 'date'])

# covert date to pandas date format
RRI['date'] = RRI['date'].astype(str)
RRI['date'] = pd.to_datetime(RRI['date'])
RRI_check = RRI.head(10000)

# save the memory --> downcast data (2.2GB --> 1.4GB)
for col in ['RepRisk_ID', 'current_RRI', 'peak_RRI']:
    RRI[col] = pd.to_numeric(RRI[col], downcast='integer')




# In[ ]:




