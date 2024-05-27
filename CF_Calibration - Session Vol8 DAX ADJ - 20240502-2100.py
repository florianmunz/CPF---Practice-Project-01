# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:34:41 2024

@author: flori
"""


"""
Computational Finance - Sesssion Vol 8 - Full Calibration of Models H93 & B96
"""



import math
import numpy as np
np.set_printoptions(suppress=True,
        formatter={'all': lambda x: '%7.6f' % x})
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
%matplotlib inline
import scipy.interpolate as sci
from scipy.optimize import fmin
from CIR_zcb_valuation_gen import B


#
# Market Data: Eonia rate (01.10.2014) + Euribor rates -------------------- #
# Source: http://www.emmi-benchmarks.eu
# on 30. September 2014
#

t_list = np.array((7, 30, 90, 180, 360)) / 360 
r_list =  np.array((3.802, 3.873, 3.825, 3.795, 3.696)) / 100

factors = (1 + t_list * r_list)
zero_rates = 1 / t_list * np.log(factors)

r0 = r_list[0] # 0.0  # set to zero 

#
# Interpolation of Market Data -------------------------------------------- #
#

tck = sci.splrep(t_list, zero_rates, k=3)                  # cubic splines
tn_list = np.linspace(0.0, 1.0, 24)
ts_list = sci.splev(tn_list, tck, der=0)
de_list = sci.splev(tn_list, tck, der=1)

f = ts_list + de_list * tn_list
  # forward rate transformation
  

plt.figure(figsize=(8, 5))
plt.plot(t_list, r_list, 'ro', label='rates')
plt.plot(tn_list, ts_list, 'b', label='interpolation', lw=1.5)
  # cubic splines
plt.plot(tn_list, de_list, 'g--', label='1st derivative', lw=1.5) 
  # first derivative
plt.legend(loc=0)
plt.xlabel('time horizon in years')
plt.ylabel('rate')

# Calculation of Forward Rates -------------------------------------------- #

def CIR_forward_rate(opt):
    kappa_r, theta_r, sigma_r = opt
    t = tn_list
    g = np.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)
    sum1 = ((kappa_r * theta_r * (np.exp(g * t) - 1)) /
          (2 * g + (kappa_r + g) * (np.exp(g * t) - 1)))
    sum2 = r0 * ((4 * g ** 2 * np.exp(g * t)) /
            (2 * g + (kappa_r + g) * (np.exp(g * t) - 1)) ** 2)
    forward_rate = sum1 + sum2
    return forward_rate


def CIR_error_function(opt):
    ''' Error function for CIR85 model calibration. '''
    kappa_r, theta_r, sigma_r = opt
    if 2 * kappa_r * theta_r < sigma_r ** 2:
        return 100
    if kappa_r < 0 or theta_r < 0 or sigma_r < 0.001:
        return 100
    forward_rates = CIR_forward_rate(opt)
    MSE = np.sum((f - forward_rates) ** 2) / len(f)
    # print opt, MSE
    return MSE


def CIR_calibration():
    opt = fmin(CIR_error_function, [1.0, 0.02, 0.1],
            xtol=0.00001, ftol=0.00001,
            maxiter=300, maxfun=500)
    return opt


opt = CIR_calibration()
opt
kappa_r, theta_r, sigma_r = CIR_calibration()
kappa_r, theta_r, sigma_r

# Visualization of Calibrated Forward Rates ------------------------------- #

def plot_calibrated_frc(opt):
    ''' Plots market and calibrated forward rate curves. '''
    forward_rates = CIR_forward_rate(opt)
    plt.figure(figsize=(8, 7))
    plt.subplot(211)
    plt.ylabel('forward rate $f(0,T)$')
    plt.plot(tn_list, f, 'b', label='market')
    plt.plot(tn_list, forward_rates, 'ro', label='model')
    plt.legend(loc=0)
    plt.axis([min(tn_list) - 0.05, max(tn_list) + 0.05,
          min(f) - 0.005, max(f) * 1.1])
    plt.subplot(212)
    wi = 0.02
    plt.bar(tn_list, forward_rates - f, width=wi)
    plt.xlabel('time horizon in years')
    plt.ylabel('difference')
    plt.axis([min(tn_list) - 0.05, max(tn_list) + 0.05,
          min(forward_rates - f) * 1.1, max(forward_rates - f) * 1.1])
    plt.tight_layout()


plot_calibrated_frc(opt)

#  ZCB - Zero Coupn Bond Values on the basis of Calibrated Forward Curves - #

def plot_zcb_values(p0, T):
    ''' Plots unit zero-coupon bond values (discount factors). '''
    t_list = np.linspace(0.0, T, 20)
    r_list = B([r0, p0[0], p0[1], p0[2], t_list, T])
    plt.figure(figsize=(8, 5))
    plt.plot(t_list, r_list, 'b')
    plt.plot(t_list, r_list, 'ro')
    plt.xlabel('time horizon in years')
    plt.ylabel('unit zero-coupon bond value')

plot_zcb_values(opt, 2)


# Calibration of Equity Component for H93 Model --------------------------- #

import math
import numpy as np
import pandas as pd
from scipy.optimize import brute, fmin, minimize
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
from BCC_option_valuation import H93_call_value, BCC_call_value
from CIR_calibration import CIR_calibration, r_list
from CIR_zcb_valuation import B

#
# Calibrate Short Rate Model
#

"""
CIR_calibration with "orginal" parameters from Session Vol 8


kappa_r, theta_r, sigma_r = CIR_calibration()

#
# Market Data from www.eurexchange.com
# as of 30. September 2014
#

"""

"""
# Base Data for Option Selection ------------------------------------------ #
# Original Data sourcing from Euro Stoxx 50 ------------------------------- #

h5 = pd.HDFStore('option_data.h5', 'r')
data = h5['data']  # European call & put option data (3 maturities)
data['Date'] = data['Date'].apply(lambda x: pd.Timestamp(x))
data['Maturity'] = data['Maturity'].apply(lambda x: pd.Timestamp(x))
h5.close()
S0 = 3225.93  # EURO STOXX 50 level 30.09.2014
r0 = r_list[0]  # initial short rate (Eonia 30.09.2014)

"""

# ----- Data From DAX Option Data (Practice Project) ---------------------- #

Start = pd.Timestamp('2020-05-15')
End =  pd.Timestamp('2020-06-19')

S0 = 10337.02                       # EURO STOXX 50 level
# r = 0.0362                        # Assumption (Base Rate GER)

df = pd.read_csv('ref_eikon_option_data_adj_II.csv', index_col=0).dropna()
df.info()

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Maturity'] = pd.to_datetime(df['Maturity'], dayfirst=True)
df.info()
df.head()
df.tail()


# Implied Volas from DAX Index Base Data ---------------------------------- # 


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
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.show()


plot_imp_vols(df)


# Separation of Call and Puts --------------------------------------------- #

Calls = df['PUTCALLIND'] == 'CALL'
Puts = df['PUTCALLIND'] == 'PUT '

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

options_call.set_index('Strike')['Close'].plot(style='o', figsize=(10, 6));


# For Running Purposes of the Code ---------------------------------------- #

options = options_call


"""
# Section of ORG Code on the Basis of EuroStoxx 50 Options ---------------- #
"""

"""
#
# Option Selection
#
tol = 0.02  # percent ITM/OTM options
options = data[(np.abs(data['Strike'] - S0) / S0) < tol]
options
"""


# Adding Time-to-Maturity and Short Rates 
#

for row, option in options.iterrows():
    T = (option['Maturity'] - option['Date']).days / 365.
    options.loc[row, 'T'] = T
    B0T = B([kappa_r, theta_r, sigma_r, r0, T])
    options.loc[row, 'r'] = -math.log(B0T) / T

options


i = 0
min_MSE = 500
def H93_error_function(p0):
    np.set_printoptions(suppress=True,
            formatter={'all': lambda x: '%5.3f' % x})
    global i, min_MSE
    kappa_v, theta_v, sigma_v, rho, v0 = p0 
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or \
                rho < -1.0 or rho > 1.0:
        return 500.0
    if 2 * kappa_v * theta_v < sigma_v ** 2:
        return 500.0
    se = []
    for row, option in options.iterrows():
        model_value = H93_call_value(S0, option['Strike'], option['T'],
                            option['r'], kappa_v, theta_v, sigma_v, rho, v0)
        se.append((model_value - option['Close']) ** 2)
    MSE = sum(se) / len(se)
    min_MSE = min(min_MSE, MSE)
    if i % 50 == 0:
        print('%4d |' % i, np.array(p0), '| %8.3f | %7.3f' % (MSE, min_MSE))
    i += 1
    return MSE

def H93_calibration_full():
    # first run with brute force
    # (scan sensible regions)
    p0 = brute(H93_error_function,
                ((25.0, 50.0, 5.0),                     # kappa_v
                (0.01, 0.041, 0.01),                    # theta_v
                (0.05, 0.251, 0.1),                     # sigma_v
                (-0.75, 0.01, 0.75),                    # rho
                (0.01, 0.031, 0.01)),                   # v0
                finish=None)

    # second run with local, convex minimization
    # (dig deeper where promising)
    opt = fmin(H93_error_function, p0,  
                 xtol=0.000001, ftol=0.000001,
                 maxiter=750, maxfun=900)
    np.save('opt_sv_test', np.array(opt))
    return opt

%time opt = H93_calibration_full()


# Calibration of the Jump Component --------------------------------------- #

#
# Option Selection
#

mats = sorted(set(options['Maturity']))
optionss = options[options['Maturity'] == mats[0]]
  # only shortest maturity

#
# Initial Parameter Guesses
#
kappa_v, theta_v, sigma_v, rho, v0 = np.load('opt_sv_test.npy')
    # from H93 model calibration
    
    
i = 0
min_MSE = 5000.0
local_opt = False
def BCC_error_function(p0):
    global i, min_MSE, local_opt, opt1
    lamb, mu, delta = p0
    if lamb < 0.0 or mu < -0.6 or mu > 0.0 or delta < 0.0:
        return 5000.0
    se = []
    for row, option in optionss.iterrows():
        model_value = BCC_call_value(S0, option['Strike'], option['T'],
                            option['r'], kappa_v, theta_v, sigma_v, rho, v0,
                            lamb, mu, delta)
        se.append((model_value - option['Close']) ** 2)
    MSE = sum(se) / len(se)
    min_MSE = min(min_MSE, MSE)
    if i % 25 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (MSE, min_MSE))
    i += 1
    if local_opt:
        penalty = np.sqrt(np.sum((p0 - opt1) ** 2)) * 1
        return MSE + penalty
    return MSE



def BCC_calibration_short():
    # first run with brute force
    # (scan sensible regions)
    opt1 = 0.0
    opt1 = brute(BCC_error_function,
                ((0.0, 0.51, 0.1),  # lambda
                (-0.5, -0.11, 0.1), # mu
                (0.0, 0.51, 0.25)), # delta
                finish=None)

    # second run with local, convex minimization
    # (dig deeper where promising)
    local_opt = True
    opt2 = fmin(BCC_error_function, opt1,
                xtol=0.0000001, ftol=0.0000001,
                maxiter=550, maxfun=750)
    np.save('opt_jump', np.array(opt2))
    return opt2

%time opt2 = BCC_calibration_short()


# Final and Full Calibration of the Model - BCC -------------------------- #

#
# Parameters from H93 & jump calibrations
#


kappa_v, theta_v, sigma_v, rho, v0 = np.load('opt_sv_test.npy')
lamb, mu, delta = np.load('opt_jump.npy')
p0 = [kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta]


def BCC_error_function(p0):
    np.set_printoptions(suppress=True,
            formatter={'all': lambda x: '%5.3f' % x})
    global i, min_MSE
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = p0 
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or \
        rho < -1.0 or rho > 1.0 or v0 < 0.0 or lamb < 0.0 or \
        mu < -.6 or mu > 0.0 or delta < 0.0:
        return 5000.0
    if 2 * kappa_v * theta_v < sigma_v ** 2:
        return 5000.0
    se = []
    for row, option in options.iterrows():
        model_value = BCC_call_value(S0, option['Strike'], option['T'],
                            option['r'], kappa_v, theta_v, sigma_v, rho, v0,
                            lamb, mu, delta)
        se.append((model_value - option['Close']) ** 2)
    MSE = sum(se) / len(se)
    min_MSE = min(min_MSE, MSE)
    if i % 25 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (MSE, min_MSE))
    i += 1
    return MSE

def BCC_calibration_full():
    # local, convex minimization for all parameters
    opt = fmin(BCC_error_function, p0,  
                 xtol=0.000001, ftol=0.000001,
                 maxiter=450, maxfun=650)
    np.save('opt_full', np.array(opt))
    return opt

%time opt3 = BCC_calibration_full()
opt3


def BCC_calculate_model_values(p0):
    ''' Calculates all model values given parameter vector p0. '''
    kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta = p0  
    values = []
    for row, option in options.iterrows():
        model_value = BCC_call_value(S0, option['Strike'], option['T'],
                            option['r'], kappa_v, theta_v, sigma_v, rho, v0,
                            lamb, mu, delta)
        values.append(model_value)
    return np.array(values)
options['Model'] = BCC_calculate_model_values(opt3)


fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for mat in set(options['Maturity']):
    options[options.Maturity == mat].plot(x='Strike', y='Close',
                                          style='b', lw=1.5,
                                          legend=True, ax=ax)
    options[options.Maturity == mat].plot(x='Strike', y='Model',
                                          style='ro', legend=True,
                                          ax=ax)
plt.xlabel('strike')
plt.ylabel('option values')


filename = 'cal_results_full_v03.h5'
h5 = pd.HDFStore(filename, 'w')
h5['options'] = options
h5.close()


# Calculation of Implied Volatilities ------------------------------------ #


from BSM_imp_vol import call_option
def calculate_implied_volatilities(filename):
    ''' Calculates market and model implied volatilities. '''
    h5 = pd.HDFStore(filename, 'r')
    options = h5['options']
    h5.close()
    for row, option in options.iterrows():
        T = (option['Maturity'] - option['Date']).days / 365.
        B0T = B([kappa_r, theta_r, sigma_r, r0, T])
        r = -math.log(B0T) / T
        call = call_option(S0, option['Strike'], option['Date'],
                            option['Maturity'], option['r'], 0.1)
        options.loc[row, 'market_iv'] = call.imp_vol(option['Close'], 0.15)
        options.loc[row, 'model_iv'] = call.imp_vol(option['Model'], 0.15)
    return options


options = calculate_implied_volatilities('cal_results_full_v03.h5')
options


# Plotting of Implied Volatilities --------------------------------------- #


def plot_implied_volatilities(options, model):
    ''' Plots market implied volatilities against model implied ones. '''
    mats = sorted(set(options.Maturity))
    for mat in mats:
        opts = options[options.Maturity == mat]
        plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.ylabel('implied volatility')
        plt.plot(opts.Strike, opts.market_iv, 'b', label='market', lw=1.5)
        plt.plot(opts.Strike, opts.model_iv, 'ro', label='model')
        plt.legend(loc=0)
        plt.axis([min(opts.Strike) - 10, max(opts.Strike) + 10,
              min(opts.market_iv) - 0.02, max(opts.market_iv) + 0.02])
        plt.title('Maturity %s' % str(mat)[:10])
        plt.subplot(212)
        wi = 5.0
        diffs = opts.model_iv.values - opts.market_iv.values
        plt.bar(opts.Strike, diffs, width=wi)
        plt.ylabel('difference')
        ymi = min(diffs) - (max(diffs) - min(diffs)) * 0.1
        yma = max(diffs) + (max(diffs) - min(diffs)) * 0.1
        plt.axis([min(opts.Strike) - 10, max(opts.Strike) + 10, ymi, yma])
        plt.tight_layout()

plot_implied_volatilities(options, 'BCC97')


# Option Prices Comparison - Market vs Model ----------------------------- #


# Base Chart - Market vs Model
options.set_index('Strike')[['Close', 'Model']].plot(figsize=(10,6), style=['b^', 'ro'])



# Subplots incl Delta View on Option Prices Comparison - Market vs Model


def plot_market_model_valuation(options, model):
    ''' Plots market implied volatilities against model implied ones. '''
    mats = sorted(set(options.Maturity))
    for mat in mats:
        opts = options[options.Maturity == mat]
        plt.figure(figsize=(8, 6))
        plt.subplot(211)
        plt.ylabel('Option prices')
        plt.plot(opts.Strike, opts.Close, 'b', label='market', lw=1.5)
        plt.plot(opts.Strike, opts.Model, 'ro', label='model')
        plt.legend(loc=0)
        plt.axis([min(opts.Strike) - 10, max(opts.Strike) + 10,
              min(opts.Close) - 20, max(opts.Close) + 20])
        plt.title('Maturity %s' % str(mat)[:10])
        plt.subplot(212)
        wi = 5.0
        diffs = opts.Close.values - opts.Model.values
        plt.bar(opts.Strike, diffs, width=wi)
        plt.ylabel('difference')
        ymi = min(diffs) - (max(diffs) - min(diffs)) * 0.1
        yma = max(diffs) + (max(diffs) - min(diffs)) * 0.1
        plt.axis([min(opts.Strike) - 10, max(opts.Strike) + 10, ymi, yma])
        plt.tight_layout()


plot_market_model_valuation(options, 'BCC97')
plot_market_model_valuation(options, 'H93')



















