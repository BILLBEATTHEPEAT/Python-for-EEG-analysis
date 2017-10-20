############################################################################
# This file is the demo code used to extract features for EEG analysis
# The mainly method is to use the existing package.
# However, the packages are sometimes obscure to use.
# So I wrote down this file as a reference for further work on EEG analysis.
######################################



###########################

###       imported package
####

import os
import pyedflib
import pandas as pd
import numpy as np
import pywt
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode()
from numpy.fft import fft,ifft,fftfreq
from scipy.signal import welch,iirfilter,filtfilt
from scipy.stats import rv_continuous
from scipy.signal import savgol_filter
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy.signal import find_peaks_cwt
from scipy.interpolate import UnivariateSpline
from scipy.stats import entropy
from scipy.stats import kurtosis
from scipy.stats import skew
import pyeeg
# from spectrum import *
# from pyentrp import entropy as ent



def DE(phase, col, delta, theta, alpha, beta, gamma, dfFeature):
    #Differential entropy
    en = np.zeros(alpha.shape[0])
    sum_of = alpha.sum() * 1.0
    alpha = alpha / sum_of
    for i in range(alpha.shape[0]):
        en[i] = entropy(alpha[i])
    dfFeature[col+'alpha_Entropy'+phase] = en
    
#     sample_entropy = np.zeros(alpha.shape[0])
#     std_alpha = np.std(alpha, axis=1)
#     for i in range(std_alpha.shape[0]):
#         sample_entropy[i] = ent.sample_entropy(alpha, 4, 0.2*std_alpha)
#     dfFeature[col+'alpha_Sample_Entropy'+phase] = sample_entropy
    
    
    en = np.zeros(delta.shape[0])
    sum_of = delta.sum() * 1.0
    delta = delta / sum_of
    for i in range(delta.shape[0]):
        en[i] = entropy(delta[i])
    dfFeature[col+'delta_Entropy'+phase] = en
    
    en = np.zeros(theta.shape[0])
    sum_of = theta.sum() * 1.0
    theta = theta / sum_of
    for i in range(theta.shape[0]):
        en[i] = entropy(theta[i])
    dfFeature[col+'theta_Entropy'+phase] = en
    
    en = np.zeros(beta.shape[0])
    sum_of = beta.sum() * 1.0
    beta = beta / sum_of
    for i in range(beta.shape[0]):
        en[i] = entropy(beta[i])
    dfFeature[col+'beta_Entropy'+phase] = en
    
    en = np.zeros(gamma.shape[0])
    sum_of = gamma.sum() * 1.0
    gamma = gamma / sum_of
    for i in range(gamma.shape[0]):
        en[i] = entropy(gamma[i])
    dfFeature[col+'gamma_Entropy'+phase] = en
    
    
#     dfFeature[col+'delta_Entropy'+phase] = np.log(dfFeature[col+'_delta_Energy'+phase])
#     dfFeature[col+'theta_Entropy'+phase] = np.log(dfFeature[col+'_theta_Energy'+phase])
#     dfFeature[col+'alpha_Entropy'+phase] = np.log(dfFeature[col+'_alpha_Energy'+phase])
#     dfFeature[col+'beta_Entropy'+phase] = np.log(dfFeature[col+'_beta_Energy'+phase])
#     dfFeature[col+'gamma_Entropy'+phase] = np.log(dfFeature[col+'_gamma_Energy'+phase])
    return dfFeature








############################################

### Following is the other useful functions
### Take down in the raw function from the introduction of its package
####


### From the package pyeeg: https://github.com/forrestbao/pyeeg/blob/master/pyeeg/__init__.py


def spectral_entropy(X, Band, Fs, Power_Ratio=None):
    """Compute spectral entropy of a time series from either two cases below:
    1. X, the time series (default)
    2. Power_Ratio, a list of normalized signal power in a set of frequency
    bins defined in Band (if Power_Ratio is provided, recommended to speed up)
    In case 1, Power_Ratio is computed by bin_power() function.
    """


def svd_entropy(X, Tau, DE, W=None):
    """Compute SVD Entropy from either two cases below:
    1. a time series X, with lag tau and embedding dimension dE (default)
    2. a list, W, of normalized singular values of a matrix (if W is provided,
    recommend to speed up.)
    """

def ap_entropy(X, M, R):
    """Computer approximate entropy (ApEN) of series X, specified by M and R.
    Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
    embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of
    Em is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension
    are 1 and M-1 respectively. Such a matrix can be built by calling pyeeg
    function as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only
    difference with Em is that the length of each embedding sequence is M + 1
    Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elements
    are Em[i][k] and Em[j][k] respectively. The distance between Em[i] and
    Em[j] is defined as 1) the maximum difference of their corresponding scalar
    components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two
    1-D vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance
    between them is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the
    value of R is defined as 20% - 30% of standard deviation of X.
    Pick Em[i] as a template, for all j such that 0 < j < N - M + 1, we can
    check whether Em[j] matches with Em[i]. Denote the number of Em[j],
    which is in the range of Em[i], as k[i], which is the i-th element of the
    vector k. The probability that a random row in Em matches Em[i] is
    \simga_1^{N-M+1} k[i] / (N - M + 1), thus sum(k)/ (N - M + 1),
    denoted as Cm[i].
    We repeat the same process on Emp and obtained Cmp[i], but here 0<i<N-M
    since the length of each sequence in Emp is M + 1.
    The probability that any two embedding sequences in Em match is then
    sum(Cm)/ (N - M +1 ). We define Phi_m = sum(log(Cm)) / (N - M + 1) and
    Phi_mp = sum(log(Cmp)) / (N - M ).
    And the ApEn is defined as Phi_m - Phi_mp.
    """


def samp_entropy(X, M, R):
    """Computer sample entropy (SampEn) of series X, specified by M and R.
    SampEn is very close to ApEn.
    Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
    embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of
    Em is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension
    are 1 and M-1 respectively. Such a matrix can be built by calling pyeeg
    function as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only
    difference with Em is that the length of each embedding sequence is M + 1
    Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elements
    are Em[i][k] and Em[j][k] respectively. The distance between Em[i] and
    Em[j] is defined as 1) the maximum difference of their corresponding scalar
    components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two
    1-D vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance
    between them is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the
    value of R is defined as 20% - 30% of standard deviation of X.
    Pick Em[i] as a template, for all j such that 0 < j < N - M , we can
    check whether Em[j] matches with Em[i]. Denote the number of Em[j],
    which is in the range of Em[i], as k[i], which is the i-th element of the
    vector k.
    We repeat the same process on Emp and obtained Cmp[i], 0 < i < N - M.
    The SampEn is defined as log(sum(Cm)/sum(Cmp))
    """



def permutation_entropy(x, n, tau):
    """Compute Permutation Entropy of a given time series x, specified by
    permutation order n and embedding lag tau.
    """
    
    
############################################

### From the package pyEntropy: https://github.com/nikdon/pyEntropy
### Install: pip install pyentrp
### More easy to use than the pyeeg package

def multiscale_entropy(time_series, sample_length, tolerance):
    n = len(time_series)
    mse = np.zeros((1, sample_length))

    for i in range(sample_length):
        b = int(np.fix(n / (i + 1)))
        temp_ts = [0] * int(b)
        for j in range(b):
            num = np.sum(time_series[j * (i + 1): (j + 1) * (i + 1)])
            den = i + 1
            temp_ts[j] = float(num) / float(den)
        se = pyentropy.sample_entropy(temp_ts, 1, tolerance)
        mse[0, i] = se
    
    return mse[0]

def multiscale_permutation_entropy(time_series, m, delay, scale):
    mspe = []
    for i in range(scale):
        coarse_time_series = pyentropy.util_granulate_time_series(time_series, i + 1)
        pe = pyentropy.permutation_entropy(coarse_time_series, m, delay)
        mspe.append(pe)
    return mspe

def composite_multiscale_entropy(time_series, sample_length, scale, tolerance=None):
    cmse = np.zeros((1, scale))

    for i in range(scale):
        for j in range(i):
            tmp = pyentropy.util_granulate_time_series(time_series[j:], i + 1)
            cmse[0,i] = pyentropy.sample_entropy(tmp, sample_length, tolerance).sum() / (i + 1)
    return cmse

def small_window_en(phase, col, delta, theta, alpha, beta, gamma, dfFeature):
    
    '''
    Calculating Multi-scale entropy for the EEG signals.
    Multiscale Entropy (sample entropy) and Multiscale-permutation Entropy.
    Paper: Nakamura T, Adjei T, Alqurashi Y, et al. Complexity science for sleep stage classification from EE.
    '''
    
    band = [delta, theta, alpha, beta, gamma]
    band_name = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    for i in range(5):
        signal = pd.DataFrame(band[i])
        signal_name = band_name[i]
        print "calculating en with moving small window on ", signal_name
        
        multiscale_arr = np.zeros((signal.shape[0],30))
        multipermu_arr = np.zeros((signal.shape[0],30))
#         compomulti_arr = np.zeros((signal.shape[0],30))

        for index, row in signal.iterrows():
            print index
            multiscale_arr[index] = multiscale_entropy(row, 30, tolerance=None)
            multipermu_arr[index] = multiscale_permutation_entropy(row, 5, 1, 30)
#             compomulti_arr[j] = composite_multiscale_entropy(signal[j], 30, 30)
        
        multiscale_arr = pd.DataFrame(multiscale_arr)
        dfFeature[col+signal_name+'_multiscale_min'+phase] = multiscale_arr.min(axis=1)
        dfFeature[col+signal_name+'_multiscale_max'+phase] = multiscale_arr.max(axis=1)
        dfFeature[col+signal_name+'_multiscale_mean'+phase] = multiscale_arr.mean(axis=1)
        dfFeature[col+signal_name+'_multiscale_median'+phase] = multiscale_arr.median(axis=1)
        dfFeature[col+signal_name+'_multiscale_std'+phase] = multiscale_arr.std(axis=1)
        dfFeature[col+signal_name+'_multiscale_var'+phase] = multiscale_arr.var(axis=1)
        dfFeature[col+signal_name+'_multiscale_diff_max'+phase] = multiscale_arr.diff(axis=1).max(axis=1)
        dfFeature[col+signal_name+'_multiscale_diff_min'+phase] = multiscale_arr.diff(axis=1).min(axis=1)
        dfFeature[col+signal_name+'_multiscale_diff_mean'+phase] = multiscale_arr.diff(axis=1).mean(axis=1)
        dfFeature[col+signal_name+'_multiscale_diff_std'+phase] = multiscale_arr.diff(axis=1).std(axis=1)
        
        multipermu_arr = pd.DataFrame(multipermu_arr)
        dfFeature[col+signal_name+'_multipermu_min'+phase] = multipermu_arr.min(axis=1)
        dfFeature[col+signal_name+'_multipermu_max'+phase] = multipermu_arr.max(axis=1)
        dfFeature[col+signal_name+'_multipermu_mean'+phase] = multipermu_arr.mean(axis=1)
        dfFeature[col+signal_name+'_multipermu_median'+phase] = multipermu_arr.median(axis=1)
        dfFeature[col+signal_name+'_multipermu_std'+phase] = multipermu_arr.std(axis=1)
        dfFeature[col+signal_name+'_multipermu_var'+phase] = multipermu_arr.var(axis=1)
        dfFeature[col+signal_name+'_multipermu_diff_max'+phase] = multipermu_arr.diff(axis=1).max(axis=1)
        dfFeature[col+signal_name+'_multipermu_diff_min'+phase] = multipermu_arr.diff(axis=1).min(axis=1)
        dfFeature[col+signal_name+'_multipermu_diff_mean'+phase] = multipermu_arr.diff(axis=1).mean(axis=1)
        dfFeature[col+signal_name+'_multipermu_diff_std'+phase] = multipermu_arr.diff(axis=1).std(axis=1)
        

        
    return dfFeature
