#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Perform the Fama-French five-factor regression
# Step 1 : Compustat Data
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats
from finance_byu.fama_macbeth import fama_macbeth, fama_macbeth_parallel, fm_summary, fama_macbeth_numba

# Import Compustat data (Please declare your path when you import data --> the place you store the data)
comp = pd.read_csv(r'C:\Users\PC\coding\finance\1. data_files\risyp8drkftwbthi.csv')

# Convertion of the date to pandas date format
comp['datadate'] = comp['datadate'].astype(str)
comp['datadate'] = pd.to_datetime(comp['datadate'])

#  Generation of the calendar year of datadate
comp['year'] = comp['datadate'].dt.year

# Creation of preferrerd stock (in the the order: pstkrv> pstkl> pstk)
comp['ps'] = np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
comp['ps'] = np.where(comp['ps'].isnull(), comp['pstk'], comp['ps'])
comp['ps'] = np.where(comp['ps'].isnull(), 0, comp['ps'])

# Replacing missing txditc with zero
comp['txditc'] = comp['txditc'].fillna(0)

# Creation of Book Equity (be) variable (cannot be negative)
comp['be'] = comp['seq'] + comp['txditc'] - comp['ps']
comp['be'] = np.where(comp['be'] > 0, comp['be'], np.nan)
comp = comp.sort_values(by=['LPERMNO', 'datadate'])
comp['count'] = comp.groupby(['LPERMNO']).cumcount()

# Creation of Operating Profitability (op)
comp['roe'] = comp['revt'] - comp['cogs'] - comp['xsga'] - comp['xint']
comp['op'] = comp['roe'].fillna(0) / comp['be']
comp['inv'] = comp.groupby(['LPERMNO'])['at'].pct_change().fillna(0)
comp['cogs'] = comp['cogs'].fillna(0)
comp['revt'] = comp['revt'].fillna(0)
comp['xsga'] = comp['xsga'].fillna(0)
comp['xint'] = comp['xint'].fillna(0)
comp['at'] = comp['at'].fillna(0)

# Removal of not needed variables
comp = comp.drop_duplicates(subset=['LPERMNO', 'year'], keep='last')
# keep necessary variables and rename for future matching
comp = comp[['LPERMNO', 'GVKEY', 'datadate', 'year', 'be', 'count', 'inv', 'op']].rename(columns={'LPERMNO': 'PERMNO'})


# In[3]:


#  Step 2: CRSP Data
#  Downloading variables from CRSP Monthly stock
crsp_m = pd.read_csv(r'C:\Users\PC\coding\finance\1. data_files\crsp.csv')

#  Filterring out necessary data
# The stock exchanges are NYSE(exchange code=1), AMEX(2) and NASDAQ(3)
crsp_m = crsp_m[crsp_m['EXCHCD'].isin([1, 2, 3])]

# Only keep common shares --> share code=10 and 11
crsp_m = crsp_m[crsp_m['SHRCD'].isin([10, 11])]

# Dropping missing returns
# Remove string value from numerical return column
crsp_m['RET'] = pd.to_numeric(crsp_m['RET'], errors='coerce')
crsp_m = crsp_m[crsp_m['RET'].notnull()]

# Changing variable format to int
crsp_m[['PERMNO', 'PERMCO', 'SHRCD', 'EXCHCD']] = crsp_m[['PERMNO', 'PERMCO', 'SHRCD', 'EXCHCD']].astype(int)

# Covertion of the data to pandas date format and line up the date to the end of month
crsp_m['date'] = crsp_m['date'].astype(str)
crsp_m['date'] = pd.to_datetime(crsp_m['date'])
crsp_m['jdate'] = crsp_m['date'] + MonthEnd(0)

# Generation of adjusted return by considering the delisting return
crsp_m['DLRET'] = pd.to_numeric(crsp_m['DLRET'], errors='coerce')  # Covert string value to missing value
crsp_m['DLRET'] = crsp_m['DLRET'].fillna(0)
crsp_m['RET_ADJ'] = (1 + crsp_m['RET']) * (1 + crsp_m['DLRET']) - 1

# Generation of market value (in millions)
crsp_m['me'] = crsp_m['PRC'].abs() * crsp_m['SHROUT'] / 1000  # Price can be negative if is the average of bid and ask

# Sorting values and keeping necessary variables
crsp_m = crsp_m.sort_values(by=['PERMCO', 'jdate'])
crsp = crsp_m.drop(['DLRET', 'PRC', 'SHROUT', 'RET', 'SHRCD'], axis=1)  # axis=1 refers to colum

# Sum of me across different permnos belonging to the same permco in a given date
crsp_summe = crsp.groupby(['jdate', 'PERMCO'])['me'].sum().reset_index()

# Largest market cap within a permco in a given date
crsp_maxme = crsp.groupby(['jdate', 'PERMCO'])['me'].max().reset_index()

# Join by jdate/maxme to find the permno --> find the permno which has the largest market cap under one permco
crsp1 = pd.merge(crsp, crsp_maxme, how='inner', on=['jdate', 'PERMCO', 'me'])

# Drop me column and replace with the sum me
crsp1 = crsp1.drop(['me'], axis=1)

# Join with sum of me to get the correct market cap info
crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=['jdate', 'PERMCO'])

# Sort by permno and date and also drop duplicates
crsp2 = crsp2.sort_values(by=['PERMNO', 'jdate']).drop_duplicates()

# keep December market cap -> When we calculate value factor (B/M) (we use the market cap on December in prior year)
crsp2['year'] = crsp2['jdate'].dt.year
crsp2['month'] = crsp2['jdate'].dt.month
decme = crsp2[crsp2['month'] == 12]
decme = decme[['PERMNO', 'jdate', 'me']].rename(columns={'me': 'dec_me', 'jdate': 'ffdate'})

# Generation of dates
crsp2['ffdate'] = crsp2['jdate'] + MonthEnd(-6)
crsp2['ffyear'] = crsp2['ffdate'].dt.year
crsp2['ffmonth'] = crsp2['ffdate'].dt.month

#  Generate the market cap of prior month as the portfolio weight (value-weighted portfolio)
crsp2['lme'] = crsp2.groupby(['PERMNO'])['me'].shift(1)  # lagged variable

#  Create a dataset in each June (Portfolio forming month) merged with market cap from previous December
#  Because there is at least 6 month gap for accounting information to be incorporated into stock price

# Keep only the data on June --> Portfolios are sorted on June of each year
crsp3 = crsp2[crsp2['month'] == 6]

# Merge with market cap in last December --> 20190630 <--> 20181231
crspjune = pd.merge(crsp3, decme, how='left', on=['PERMNO', 'ffdate'])


# In[6]:


# Part 3 of Task 1: Data Merge
# Merge CRSP with Compustat data on June of each year

# Prepare compustat data for matching
comp['jdate'] = comp['datadate'] + YearEnd(0)
comp['jdate'] = comp['jdate'] + MonthEnd(6)

# Keep necessary variables in Compustat
comp2 = comp[['PERMNO', 'jdate', 'be', 'count', 'inv', 'op']]

# Keep necessary variables in crspjune
crspjune2 = crspjune[['PERMNO', 'PERMCO', 'jdate', 'RET_ADJ', 'me', 'lme', 'dec_me', 'EXCHCD']]

# Merge the crspjune2 and compustat2
ccm_june = pd.merge(crspjune, comp2, how='inner', on=['PERMNO', 'jdate'])

# Generate book to market ratio (B/M)
ccm_june['beme'] = ccm_june['be'] / ccm_june['dec_me']

#  Part 4 of Task 1: Formation of Portfolios
#  Forming Portolios by ME and BEME as of each June
#  Calculate NYSE Breakpoints (size factor) for Market Equity (ME) and  Book-to-Market (BEME)
#  Note that we only use the stocks in NYSE to define the "big" and "small" stock / "value" and "growth" stock (median)
# Select NYSE stocks for bucket breakdown
# exchcd = 1 and positive beme and positive me and at least 2 years in comp
nyse = ccm_june[(ccm_june['EXCHCD'] == 1) & (ccm_june['beme'] > 0) & (ccm_june['me'] > 0) & (ccm_june['count'] > 1)]

# Size breakdown
nyse_sz = nyse.groupby(['jdate'])['me'].median().to_frame().reset_index().rename(columns={'me': 'sizemedn'})

# BEME breakdown
nyse_bm = nyse.groupby(['jdate'])['beme'].describe(percentiles=[0.3, 0.7]).reset_index()
nyse_bm = nyse_bm[['jdate', '30%', '70%']].rename(columns={'30%': 'bm30', '70%': 'bm70'})

# Merge size and BEME breakdown
nyse_breaks_bm = pd.merge(nyse_sz, nyse_bm, how='inner', on=['jdate'])

# Join back size and beme breakdown
ccm_june2 = pd.merge(ccm_june, nyse_breaks_bm, how='left', on=['jdate'])

# Creation of profitability factor
nyse_operprofit = nyse.groupby(['jdate'])['op'].describe(percentiles=[0.3, 0.7]).reset_index()
nyse_operprofit = nyse_operprofit[['jdate', '30%', '70%']].rename(
    columns={'30%': 'operprofit30', '70%': 'operprofit70'})

# Creation of investment factor
nyse_invest = nyse.groupby(['jdate'])['inv'].describe(percentiles=[0.3, 0.7]).reset_index()
nyse_invest = nyse_invest[['jdate', '30%', '70%']].rename(columns={'30%': 'invest30', '70%': 'invest70'})

# Merge size and operating profit breakdown
nyse_breaks_op = pd.merge(nyse_sz, nyse_operprofit, how='inner', on=['jdate'])

# Merge size and investment breakdown
nyse_breaks_invest = pd.merge(nyse_sz, nyse_invest, how='inner', on=['jdate'])

# Join back size and beme breakdown
ccm_june2 = pd.merge(ccm_june, nyse_breaks_bm, how='left', on=['jdate'])

# Join back size and operating profitabilty breakdown
ccm_june3 = pd.merge(ccm_june2, nyse_breaks_op, how='left', on=['jdate'])

# Join back size and investment breakdown
ccm_june4 = pd.merge(ccm_june3, nyse_breaks_invest, how='left', on=['jdate'])
ccm_june4 = ccm_june4.drop_duplicates()
ccm_june4 = ccm_june4.fillna(0)


# ccm_june5 = pd.merge(ccm_june2, ccm_june3, how = 'left', on = ['jdate'])
# ccm_june5 = pd.merge(ccm_june5, ccm_june4, how = 'left', on = ['jdate'])
# Two small functions to define types
# functions to assign sz and bm bucket
# Buckets for different factors

# Definition of the necessary functions

def sz_bucket(row):
    if row['me'] == np.nan:
        value = ''
    elif row['me'] <= row['sizemedn']:
        value = 'S'
    else:
        value = 'B'
    return value


def bm_bucket_1(row):
    if 0 <= row['beme'] <= row['bm30']:
        value = 'L'
    elif row['beme'] <= row['bm70']:
        value = 'M'
    elif row['beme'] > row['bm70']:
        value = 'H'
    else:
        value = ''
    return value


def bm_bucket_2(row):
    if row['op'] <= row['operprofit30']:
        value = 'W'
    elif row['op'] <= row['operprofit70']:
        value = 'N'
    elif row['op'] > row['operprofit70']:
        value = 'R'
    else:
        value = ''
    return value


def bm_bucket_3(row):
    if row['inv'] <= row['invest30']:
        value = 'C'
    elif row['inv'] <= row['invest70']:
        value = 'Z'
    elif row['inv'] > row['invest70']:
        value = 'A'
    else:
        value = ''
    return value


# BE portfolio
# Assign size portfolio for book to market
ccm_june4['sizeport'] = np.where((ccm_june4['beme'] > 0) & (ccm_june4['me'] > 0) & (ccm_june4['count'] >= 1),
                                 ccm_june4.apply(sz_bucket, axis=1), '')

# Assign book-to-market portfolio
ccm_june4['booktomarketport'] = np.where((ccm_june4['beme'] > 0) & (ccm_june4['me'] > 0) & (ccm_june4['count'] >= 1),
                                         ccm_june4.apply(bm_bucket_1, axis=1), '')

# Assign book-to-market portfolio
ccm_june4['vkport'] = np.where((ccm_june4['beme'] > 0) & (ccm_june4['me'] > 0) & (ccm_june4['count'] >= 1),
                               ccm_june4.apply(bm_bucket_2, axis=1), '')

# Assign book-to-market portfolio
ccm_june4['abport'] = np.where((ccm_june4['beme'] > 0) & (ccm_june4['me'] > 0) & (ccm_june4['count'] >= 1),
                               ccm_june4.apply(bm_bucket_3, axis=1), '')
# Create positivebmeme and nonmissport variable
ccm_june4['posbm'] = np.where((ccm_june4['beme'] > 0) & (ccm_june4['me'] > 0) & (ccm_june4['count'] >= 1), 1, 0)
ccm_june4['nonmissport'] = np.where((ccm_june4['booktomarketport'] != ''), 1, 0)
# Store portfolio assignment as of June
june = ccm_june4[['PERMNO', 'jdate', 'booktomarketport', 'sizeport', 'posbm', 'vkport', 'abport', 'nonmissport']]
june['ffyear'] = june['jdate'].dt.year

# Merge back with monthly records
crsp4 = crsp2[['date', 'PERMNO', 'RET_ADJ', 'me', 'lme', 'ffyear', 'jdate']]
ccm = pd.merge(crsp4,
               june[['PERMNO', 'ffyear', 'sizeport', 'booktomarketport', 'vkport', 'abport', 'posbm', 'nonmissport']],
               how='left',
               on=['PERMNO', 'ffyear'])


#  Step 5: Formation of size and value factors and evaluation of replicated results
#  Function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


vwret = ccm.groupby(['jdate', 'sizeport', 'booktomarketport']).apply(wavg, 'RET_ADJ', 'lme').to_frame().reset_index() \
    .rename(columns={0: 'vwret'})
vwret['sizetobookport'] = vwret['sizeport'] + vwret['booktomarketport']

# Firm count --> How many firms in one portfolio in a given month (jdate)
vwret_n = ccm.groupby(['jdate', 'sizeport', 'booktomarketport'])['RET_ADJ'].count().reset_index().rename(
    columns={'RET_ADJ': 'n_firms'})
vwret_n['sizetobookport'] = vwret_n['sizeport'] + vwret_n[
    'booktomarketport']  # 2 X 3 symbols or indicators SH/SL/SM/BH/BL/BM

# operating profitability
vwret_1 = ccm.groupby(['jdate', 'sizeport', 'vkport']).apply(wavg, 'RET_ADJ', 'lme').to_frame().reset_index() \
    .rename(columns={0: 'operating_vwret'})
vwret_1['operport'] = vwret_1['sizeport'] + vwret_1['vkport']
vwret_n1 = ccm.groupby(['jdate', 'sizeport', 'vkport'])['RET_ADJ'].count().reset_index().rename(
    columns={'RET_ADJ': 'n_firms'})
vwret_n1['operport'] = vwret_n1['sizeport'] + vwret_n1['vkport']  # 2 X 3 symbols or indicators SH/SL/SM/BH/BL/BM

# investment portfolio
vwret_2 = vwret_3 = ccm.groupby(['jdate', 'sizeport', 'abport']).apply(wavg, 'RET_ADJ', 'lme').to_frame().reset_index() \
    .rename(columns={0: 'investment_vwret'})
vwret_2['investmentport'] = vwret_2['sizeport'] + vwret_2['abport']

vwret_n2 = ccm.groupby(['jdate', 'sizeport', 'abport'])['RET_ADJ'].count().reset_index().rename(
    columns={'RET_ADJ': 'n_firms'})
vwret_n2['investmentport'] = vwret_n2['sizeport'] + vwret_n2['abport']  # 2 X 3 symbols or indicators SH/SL/SM/BH/BL/BM

# Transpose the data for size
ff_factors = vwret.pivot(index='jdate', columns='sizetobookport', values='vwret').reset_index()
ff_nfirms = vwret_n.pivot(index='jdate', columns='sizetobookport', values='n_firms').reset_index()
# Transpose the data for opearting
ff_factors_operating = vwret_1.pivot(index='jdate', columns='operport', values='operating_vwret').reset_index()
ff_nfirms1 = vwret_n1.pivot(index='jdate', columns='operport', values='n_firms').reset_index()
# Transpose the data for investing
ff_factors_investing = vwret_2.pivot(index='jdate', columns='investmentport', values='investment_vwret').reset_index()
ff_nfirms2 = vwret_n2.pivot(index='jdate', columns='investmentport', values='n_firms').reset_index()

# Create SMB and HML factors  two more factors --> RMW, CWA

ff_factors['Big'] = (ff_factors['BL'] + ff_factors['BM'] + ff_factors['BH']) / 3  # (Big low-Big Medium- Big High)/3
ff_factors['Small'] = (ff_factors['SL'] + ff_factors['SM'] + ff_factors[
    'SH']) / 3  # (Small low-small medium-Big high)/3
ff_factors['SMB OF B\M'] = ff_factors['Small'] - ff_factors['Big']  # Size factor
ff_factors = ff_factors.rename(columns={'jdate': 'date'})

ff_factors['High'] = (ff_factors['BH'] + ff_factors['SH']) / 2  # (Big value+Small value)/2
ff_factors['Low'] = (ff_factors['BL'] + ff_factors['SL']) / 2  # (Big growth+Small growth)/2
ff_factors['HighMLow'] = ff_factors['High'] - ff_factors['Low']  # Value factor
ff_factors_investing = ff_factors_investing.rename(columns={'jdate': 'date'})

# OPERATING Profitability

ff_factors_operating['BigOp'] = (ff_factors_operating['BR'] + ff_factors_operating['BN'] + ff_factors_operating[
    'BW']) / 3  # (SmalL Robust +Small Neutral +small weak)/3
ff_factors_operating['SmallOp'] = (ff_factors_operating['SR'] + ff_factors_operating['SN'] + ff_factors_operating[
    'SW']) / 3  # (Big Robust +Big Neutral +Big weak) /3
ff_factors_operating['SMB OF Operating'] = ff_factors_operating['SmallOp'] - ff_factors_operating[
    'BigOp']  # Size factor
ff_factors_operating = ff_factors_operating.fillna(0)

ff_factors_operating['High'] = (ff_factors_operating['SR'] + ff_factors_operating['BR']) / 2
ff_factors_operating['Low'] = (ff_factors_operating['SW'] + ff_factors_operating['BW']) / 2
ff_factors_operating['RMW_m'] = ff_factors_operating['High'] - ff_factors_operating['Low']  # Value Factor
ff_factors_operating = ff_factors_operating.rename(columns={'jdate': 'date'})

# Investment Factor

ff_factors_investing['Aggr'] = (ff_factors_investing['BA'] + ff_factors_investing['BZ'] + ff_factors_investing[
    'BC']) / 3  # (SmalL Conservative +Small Neutral +small Aggressive)/3
ff_factors_investing['Consv'] = (ff_factors_investing['SA'] + ff_factors_investing['SZ'] + ff_factors_investing[
    'SC']) / 3  # (Big Conservative +Big Neutral +Big Aggressive)
ff_factors_investing['SMB of investing'] = ff_factors_investing['Consv'] - ff_factors_investing['Aggr']  # Size factor

ff_factors_investing['High'] = (ff_factors_investing['BC'] + ff_factors_investing['SC']) / 2
ff_factors_investing['Low'] = (ff_factors_investing['BA'] + ff_factors_investing['SA']) / 2
ff_factors_investing['CMA_m'] = ff_factors_investing['High'] - ff_factors_investing['Low']  # Value factor

ff_factors = pd.merge(ff_factors_operating[['date', 'SMB OF Operating', 'RMW_m']],
                      ff_factors[['date', 'SMB OF B\M', 'HighMLow']], how='left', on=['date'])
ff_factors = pd.merge(ff_factors, ff_factors_investing[['date', 'SMB of investing', 'CMA_m']], how='left', on=['date'])
ff_factors['SMB_m'] = ff_factors['SMB OF B\M'] + ff_factors_operating['SMB OF Operating'] + ff_factors_investing[
    'SMB of investing']
# ff_factors['HML'] = ff_factors['HighMinusLow'] + ff_factors_operating['SMB OF Operating'] + ff_factors_investing['SMB of investing']
ff5_factors = ff_factors[['date', 'SMB_m', 'HighMLow', 'RMW_m', 'CMA_m']]
ff5_factors = ff5_factors.fillna(0)

#  Mutual Funds Data

Mflarge = pd.read_csv(r'C:\Users\PC\coding\finance\1. data_files\VANGUARD Large value.csv')  # LARGE VALUE MUTUAL FUNDS
Mfsmall = pd.read_csv(r'C:\Users\PC\coding\finance\1. data_files\OPOCX Small growth.csv')  # SMALL GROWTH MUTUAL FUNDS

Mflarge = Mflarge.rename(columns={'caldt': 'date', 'crsp_fundno': 'MEIAX_Lvalue', 'mret': 'ret_Lvalue'})
Mflarge['date'] = Mflarge['date'].astype(str)
Mflarge['year'] = Mflarge['date'].str[:4]  # first 4 digits --> from 0 to 3 excluding 4
Mflarge['month'] = Mflarge['date'].str[4:6]  # excluding 4 --> last two digits
Mflarge[['year', 'month']] = Mflarge[['year', 'month']].astype(int)
# Mflarge = Mflarge.drop('date')

# Filter sample from 2010 jan to Dec 2020
Mflarge = Mflarge[(Mflarge['year'] <= 2020) & (Mflarge['year'] >= 2010)]
Mflarge = Mflarge.drop(Mflarge[(Mflarge['year'] == 2010) & (Mflarge['month'] <= 6)].index).reset_index()

Mfsmall = Mfsmall.rename(columns={'caldt': 'date', 'crsp_fundno': 'OPOCX_Sgrowth', 'mret': 'ret_SGrowth'})
Mfsmall['date'] = Mfsmall['date'].astype(str)
Mfsmall['year'] = Mfsmall['date'].str[:4]  # first 4 digits --> from 0 to 3 excluding 4
Mfsmall['month'] = Mfsmall['date'].str[4:6]  # excluding 4 --> last two digits
Mfsmall[['year', 'month']] = Mfsmall[['year', 'month']].astype(int)

# Mfsmall = Mfsmall.drop('date')

# Filter sample from 2010 to 2020
Mfsmall = Mfsmall[(Mfsmall['year'] <= 2020) & (Mfsmall['year'] >= 2010)]
Mfsmall = Mfsmall.drop(Mfsmall[(Mfsmall['year'] == 2010) & (Mfsmall['month'] <= 6)].index).reset_index()

MFund = pd.merge(Mflarge, Mfsmall, how='left', on=['year', 'month'])

# Comparison with the Fama-Frech data

# Import and download the Farma-French
ff5 = pd.read_csv(r'C:\Users\PC\coding\finance\1. data_files\ff-data_lib.csv')
ff5['Date'] = ff5['Date'].astype(str)
ff5['year'] = ff5['Date'].str[:4]  # first 4 digits --> from 0 to 3 excluding 4
ff5['month'] = ff5['Date'].str[4:]  # excluding 4 --> last two digits
ff5[['year', 'month']] = ff5[['year', 'month']].astype(int)

# Filter sample from 2010-202012
ff5 = ff5[(ff5['year'] <= 2020) & (ff5['year'] >= 2010)]
ff5 = ff5.drop(ff5[(ff5['year'] == 2010) & (ff5['month'] <= 6)].index).reset_index()
regress_ff5 = pd.merge(ff5, MFund, how='left', on=['year', 'month'])
regress_ff5['Y_LV'] = regress_ff5['ret_Lvalue'] - regress_ff5['RF']
regress_ff5['Y_SG'] = regress_ff5['ret_SGrowth'] - regress_ff5['RF']
regress_ff5['Mkt-RF'] = regress_ff5['Mkt-RF'] / 100
regress_ff5['SMB'] = regress_ff5['SMB'] / 100
regress_ff5['HML'] = regress_ff5['HML'] / 100
regress_ff5['RMW'] = regress_ff5['RMW'] / 100
regress_ff5['CMA'] = regress_ff5['CMA'] / 100


def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1
    result = sm.OLS(Y, X).fit()
    return result.params


# This step is to estimate the beta for different risk factors in the first stage
beta_LV = regress_ff5.groupby('Date').apply(regress, 'Y_LV', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
beta_SG = regress_ff5.groupby('Date').apply(regress, 'Y_SG', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])

beta_LV = beta_LV.rename(
    columns={'Mkt-RF': 'beta_LV_mkt', 'SMB': 'beta_LV_smb', 'HML': 'beta_LV_hml', 'RMW': 'beta_LV_rmw',
             'CMA': 'beta_LV_cma'})
beta_SG = beta_SG.rename(
    columns={'Mkt-RF': 'beta_SG_mkt', 'SMB': 'beta_SG_smb', 'HML': 'beta_SG_hml', 'RMW': 'beta_SG_rmw',
             'CMA': 'beta_SG_cma'})

# Merge back with original dataset
# arge_value = regress_ff5[['Date', 'Y_LV', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
MF_LV = pd.merge(regress_ff5[['Y_LV', 'Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']], beta_LV, how='left', on=['Date'])
# small_growth = regress_ff5[['Date', 'Y_SG', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
MF_SG = pd.merge(regress_ff5[['Y_SG', 'Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']], beta_SG, how='left', on=['Date'])

# The second-stage cross-sectional regressions
result_largevalue = fama_macbeth(MF_LV, 'Date', 'Y_LV', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'], intercept=True)
result_smallgrowth = fama_macbeth(MF_SG, 'Date', 'Y_SG', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'], intercept=True)

# Clean our replicated results for matching
ff5_factors['year'] = ff5_factors['date'].dt.year
ff5_factors['month'] = ff5_factors['date'].dt.month

ff5_factors = ff5_factors[(ff5_factors['year'] <= 2020) & (ff5_factors['year'] >= 2010)]
ff5_factors = ff5_factors.drop(
    ff5_factors[(ff5_factors['year'] == 2010) & (ff5_factors['month'] <= 6)].index).reset_index()

# Merge constructed factors and replicated factors
factor = pd.merge(ff5_factors, regress_ff5[['year', 'month', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']], how='left',
                  on=['year', 'month'])

# factor = factor.dropna()

# outputs
# regression
print('\n Mutual fund Large Value = RSQAX \n', fm_summary(result_largevalue))
print('\n Mutaula fund Small Growth = CSMCX \n', fm_summary(result_smallgrowth))

# Test the correlation between constructed factors and our replicated factors
SMB_corr = factor['SMB_m'].corr(factor['SMB'])
HML_corr = factor['HighMLow'].corr(factor['HML'])
RMW_corr = factor['RMW_m'].corr(factor['RMW'])
CMA_corr = factor['CMA_m'].corr(factor['CMA'])
print(SMB_corr, HML_corr, RMW_corr, CMA_corr)

factor.plot(x='date', y=['SMB_m', 'SMB'])
factor.plot(x='date', y=['HighMLow', 'HML'])
factor.plot(x='date', y=['RMW_m', 'RMW'])
factor.plot(x='date', y=['CMA_m', 'CMA'])
plt.show()  # show the graphs


# In[ ]:




