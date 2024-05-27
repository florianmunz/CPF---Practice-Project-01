# -*- coding: utf-8 -*-
"""
Created on Sat May  4 19:37:07 2024

@author: flori
"""



from pylab import plt
import pandas as pd
import numpy as np
plt.style.use('seaborn-v0_8')
%matplotlib inline

import requests
import cufflinks
import pandas as pd
from io import StringIO
from eod import EodHistoricalData
cufflinks.set_config_file(offline=True)

import warnings
warnings.simplefilter('ignore')

from BSM_imp_vol import call_option


import math
import numpy as np
from scipy.integrate import quad

def M76_characteristic_function(u, T, r, sigma, lamb, mu, delta):
    omega = r - 0.5 * sigma ** 2 - lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
    value = np.exp((1j * u * omega - 0.5 * u ** 2 * sigma ** 2 +
            lamb * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1))  * T)
    return value


def M76_integration_function(u, S0, K, T, r, sigma, lamb, mu, delta):
    JDCF = M76_characteristic_function(u - 0.5 * 1j, T, r,
                                       sigma, lamb, mu, delta)
    value = 1 / (u ** 2 + 0.25) * (np.exp(1j * u * math.log(S0 / K))
                                    * JDCF).real
    return value


def M76_value_call_INT(S0, K, T, r, sigma, lamb, mu, delta):
    int_value = quad(lambda u: M76_integration_function(u, S0, K, T, r,
                    sigma, lamb, mu, delta), 0, 50, limit=250)[0]
    call_value = S0 - np.exp(-r * T) * math.sqrt(S0 * K) / math.pi * int_value
    return call_value


# Base Data for the EuroStoxx 50 - Jan 2024 Data Set

S0 = 4663.5                       # EuroStoxx50 level
r = 0.0379                        # Assumption (Base Rate Europe)

df = pd.read_csv('EUROStoxx50_Options_2024-01_SHORT.csv').dropna()
df['ts'] = pd.to_datetime(df['ts'], dayfirst=True)
df['Timestamp_txt'] = pd.to_datetime(df['Timestamp_txt'], dayfirst=True)
df['Maturity'] = pd.to_datetime(df['Maturity'], dayfirst=True)
df.info()

df.rename(columns={'StrikePrice': 'Strike'}, inplace=True)
df.rename(columns={'sigma': 'Imp_Vol'}, inplace=True)
df.rename(columns={'ts': 'Date'}, inplace=True)
df.rename(columns={'ask': 'Close'}, inplace=True)
df.info()
df.head()
df.tail()

markers = ['.', 'o', '^', 'v', 'x', 'D', 'd', '>', '<']

def plot_imp_vols(data):
    ''' Plot the implied volatilites. '''
    maturities = sorted(set(data['Maturity']))
    plt.figure(figsize=(10, 6))
    for i, mat in enumerate(maturities):
        dat = data[(data['Maturity'] == mat) & (data['Imp_Vol'] > 0)]
        plt.plot(dat['Strike'].values, dat['Imp_Vol'].values,
                 'b%s' % markers[i], label=str(mat)[:10])
    plt.grid(True)
    plt.legend()
    plt.xlabel('strike')
    plt.ylabel('implied volatility')
    plt.show()

# Implied Volas from Base Data for EuroStoxx 50 Options - Jan 2024
plot_imp_vols(df)


# Separation of Puts & Call in Imp Vola Plot ---------------------------- #

def plot_imp_vols_sep(data):
    ''' Plot the implied volatilites. '''
    maturities = sorted(set(data['Maturity']))
    plt.figure(figsize=(10, 6))
    for i, mat in enumerate(maturities):
        dat = data[(data['Maturity'] == mat) & (data['Imp_Vol'] > 0) & (data['PutOrCall'] == 'Call')]
        plt.plot(dat['Strike'].values, dat['Imp_Vol'].values,
                 'r%s' % markers[i], label=str('CALLS')[:10])
    for i, mat in enumerate(maturities):
        dat = data[(data['Maturity'] == mat) & (data['Imp_Vol'] > 0) & (data['PutOrCall'] == 'Put')]
        plt.plot(dat['Strike'].values, dat['Imp_Vol'].values,
                 'b%s' % markers[i], label=str('PUTS')[:10])
    plt.grid(True)
    plt.legend()
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.tight_layout()
    plt.savefig('Imp_Volas_EUROStoxx50_Jan 24_Base Data.png')


plot_imp_vols_sep(df)

Calls = df['PutOrCall'] == 'Call'
Puts = df['PutOrCall'] == 'Put'

df_call = df[~Puts]
df_put = df[~Calls]

plot_imp_vols(df_call)

plot_imp_vols(df_put)


# Option Selection (from Calls & Puts)
tol = 0.02
options = df[(np.abs(df['Strike'] - S0) / S0) < tol]
mats = sorted(set(options['Maturity']))
options = options[options['Maturity'] == mats[0]]
options

options.set_index('Strike')['Close'].plot(style='o', figsize=(10, 6));

# Option Selection (from Calls only)
tol = 0.02
options_call = df_call[(np.abs(df_call['Strike'] - S0) / S0) < tol]
mats = sorted(set(options_call['Maturity']))
options_call = options_call[options_call['Maturity'] == mats[0]]
options_call

options_call.set_index('Strike')['Close'].plot(style='o', figsize=(10, 6))
options_call.set_index('Strike')['Imp_Vol'].plot(style='o', figsize=(10, 6))


# Model Calculation of Implied Volas from Base Data - EuroStoxx50 Jan 2024 Base Data

def calculate_imp_vols_II(data):
    ''' Calculate all implied volatilities for the European call options
    given the tolerance level for moneyness of the option.'''
    data['Imp_Vol_II'] = 0.0
    tol = 0.30  																	# tolerance for moneyness
    for row in data.index:
        t = data['Date'][row]
        T = data['Maturity'][row]
        ttm = (T - t).days / 365.
        forward = np.exp(r * ttm) * S0
        if (abs(data['Strike'][row] - forward) / forward) < tol:
            call = call_option(S0, data['Strike'][row], t, T, r, 0.2)
            data['Imp_Vol_II'][row] = call.imp_vol(data['Close'][row])
    return data

calculate_imp_vols_II(options_call)


def plot_imp_vols_II(data):
    ''' Plot the implied volatilites. '''
    maturities = sorted(set(data['Maturity']))
    plt.figure(figsize=(10, 6))
    for i, mat in enumerate(maturities):
        dat = data[(data['Maturity'] == mat) & (data['Imp_Vol_II'] > 0)]
        plt.plot(dat['Strike'].values, dat['Imp_Vol_II'].values,
                 'b%s' % markers[i], label=str(mat)[:10])
    plt.grid(True)
    plt.legend()
    plt.xlabel('strike')
    plt.ylabel('implied volatility')
    plt.show()


plot_imp_vols_II(options_call)
