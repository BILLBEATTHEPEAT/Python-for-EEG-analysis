############################################################################
# This file is the demo code used to extract energy features for EEG analysis
# The mainly method is to use the existing package.
# However, the packages are sometimes obscure to use.
# So I wrote down this file as a reference for further work on EEG analysis.
#############################################################################




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



############################

###				Read the data from .edf files and the labels from .txt files.
###				Then concatenate the data and the label togther and 
###				reshape them into the form of (n_samples, length_of_signal)
####

def read_data(filename):

'''
Read data from the .edf and .txt files and output in pd.DataFrame.
'''

    f = pyedflib.EdfReader('eeg/edfs/'+filename+'.edf')
    headers = pd.DataFrame(f.getSignalHeaders())
    headers_512 = headers[ headers.sample_rate == 512][['sample_rate','label']]
    #load data
    data1 = []
    for row in headers_512.itertuples():
        idx = row.Index
        name = row.label
        data1.append(pd.DataFrame(f.readSignal(idx),columns=[name]))
    data = pd.concat(data1,axis=1)
    #load label
    label = pd.read_csv('eeg/edfs/'+filename+'.txt',header=None)
    label.columns = ['label']
    return data,label



def data_reshape(data,col):

'''
The data is the output of read_data in pd.DataFrame form.
The col is the name of the inpput signal channel.
Return in pd.DataFrame form
'''

    cols = [col+'_'+x for x in (map(str,range(15360)))]
    frame = pd.DataFrame(data[0][:len(data[1])*30*512][col].values.reshape((len(data[1]),15360)),columns=cols)
    return frame  



def map_label(label):

'''
Map the label from str to int. The W, N1, N2, N3, R represent each stage of sleep stages.
Return in dictionary form.
'''


    label_dic={'W':1,'N1':2,'N2':3,'N3':4,'R':5}
    return label_dic[label]



###########################

###				Filter the input signal with banpass butter filter.
###				According to the researches, we saperate the signal into 
###				5 to 7 frequency bands: alpha, delta, theta, beta, gamma, low_alpha and high_alpha
####

def Filter(frame1):

'''
Filter delta,theta,alpha,beta,gamma.
Input is the original signal data.
Outputs are the filtered time-domain signals for each band.
'''
    a,b = iirfilter(1,[1.0/1024.0,4.0/1024.0],btype='bandpass',ftype='butter')
    delta = filtfilt(a,b,frame1,axis=1)
    a,b = iirfilter(1,[4.0/1024.0,8.0/1024.0],btype='bandpass',ftype='butter')
    theta = filtfilt(a,b,frame1,axis=1)
    a,b = iirfilter(1,[8.0/1024.0,14.0/1024.0],btype='bandpass',ftype='butter')
    alpha = filtfilt(a,b,frame1,axis=1)
    a,b = iirfilter(1,[8.5/1024.0,11.5/1024.0],btype='bandpass',ftype='butter')
    low_alpha = filtfilt(a,b,frame1,axis=1)
    a,b = iirfilter(1,[11.5/1024.0,15.5/1024.0],btype='bandpass',ftype='butter')
    high_alpha = filtfilt(a,b,frame1,axis=1)
    a,b = iirfilter(1,[14.0/1024.0,31.0/1024.0],btype='bandpass',ftype='butter')
    beta = filtfilt(a,b,frame1,axis=1)
    a,b = iirfilter(1,[31.0/1024.0,50.0/1024.0],btype='bandpass',ftype='butter')
    gamma = filtfilt(a,b,frame1,axis=1)

    return delta, theta, alpha, beta, gamma, low_alpha, high_alpha




#######################################

### Calculate power spectual density for each frequency band (as a whole) and their relative ratio value.
### This might be the most useful and important feature for the Sleep stage classification
###
####

def energy_psd(phase, col, delta, theta, alpha, beta, gamma, low_alpha, high_alpha, dfFeature):

'''
Input:
	phase: we can divide the original signal into several phase with respect with time. 
		   This variable is the label of each phase
	col: The name of each channel of bological signal.
	dfFeature: The returned df.DataFrame.
'''
    print "------------------------phase ", phase, "begin:"
    delta_f,delta_psd = welch(delta,fs=512,scaling='density',axis=1)
#     if 'Origin' == phase:
    if '_phase_0' == phase:
        dfFeature = pd.DataFrame((delta_psd*delta_psd).sum(axis=1))
        dfFeature.columns = [col+'_delta_Energy'+phase]
    else:
        dfFeature[col+'_delta_Energy'+phase] = (delta_psd*delta_psd).sum(axis=1)
    theta_f,theta_psd = welch(theta,fs=512,scaling='density',axis=1)
    dfFeature[col+'_theta_Energy'+phase] = (theta_psd*theta_psd).sum(axis=1)
    alpha_f,alpha_psd = welch(alpha,fs=512,scaling='density',axis=1)
    dfFeature[col+'_alpha_Energy'+phase] = (alpha_psd*alpha_psd).sum(axis=1)
    beta_f,beta_psd = welch(beta,fs=512,scaling='density',axis=1)
    dfFeature[col+'_beta_Energy'+phase] = (beta_psd*beta_psd).sum(axis=1)
    gamma_f,gamma_psd = welch(gamma,fs=512,scaling='density',axis=1)
    dfFeature[col+'_gamma_Energy'+phase] = (gamma_psd*gamma_psd).sum(axis=1)
    
    low_alpha_f,low_alpha_psd = welch(low_alpha,fs=512,scaling='density',axis=1)
    dfFeature[col+'_low_alpha_Energy'+phase] = (low_alpha_psd*low_alpha_psd).sum(axis=1)
    high_alpha_f,high_alpha_psd = welch(high_alpha,fs=512,scaling='density',axis=1)
    dfFeature[col+'_high_alpha_Energy'+phase] = (high_alpha_psd*high_alpha_psd).sum(axis=1)
    
    dfFeature[col+'_Energy'+phase] = dfFeature[col+'_delta_Energy'+phase]+dfFeature[col+'_theta_Energy'+phase]+dfFeature[col+'_alpha_Energy'+phase]+\
                                dfFeature[col+'_beta_Energy'+phase]+dfFeature[col+'_gamma_Energy'+phase]
        
    dfFeature[col+'Energyratio1'+phase] = (dfFeature[col+'_alpha_Energy'+phase])/(dfFeature[col+'_delta_Energy'+phase]+dfFeature[col+'_theta_Energy'+phase])
    dfFeature[col+'Energyratio2'+phase] = (dfFeature[col+'_delta_Energy'+phase])/(dfFeature[col+'_theta_Energy'+phase]+dfFeature[col+'_alpha_Energy'+phase])
    dfFeature[col+'Energyratio3'+phase] = (dfFeature[col+'_theta_Energy'+phase])/(dfFeature[col+'_delta_Energy'+phase]+dfFeature[col+'_alpha_Energy'+phase])

    dfFeature[col+'Energyrelative1'+phase] = (dfFeature[col+'_alpha_Energy'+phase])/dfFeature[col+'_Energy'+phase]
    dfFeature[col+'Energyrelative2'+phase] = (dfFeature[col+'_delta_Energy'+phase])/dfFeature[col+'_Energy'+phase]   
    dfFeature[col+'Energyrelative3'+phase] = (dfFeature[col+'_theta_Energy'+phase])/dfFeature[col+'_Energy'+phase]
    dfFeature[col+'Energyrelative4'+phase] = (dfFeature[col+'_beta_Energy'+phase])/dfFeature[col+'_Energy'+phase]
    dfFeature[col+'Energyrelative5'+phase] = (dfFeature[col+'_gamma_Energy'+phase])/dfFeature[col+'_Energy'+phase]
        
    return dfFeature



##########################################

### Calculate the psd with small moving windows (6 seconds with 50% overlaping)
### Extract statistic value from these psd series.
####

def small_window_psd(phase, col, delta, theta, alpha, beta, gamma, dfFeature):
    
    band = [delta, theta, alpha, beta, gamma]
    band_name = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    for i in range(5):
        signal = np.array(band[i])
        signal_name = band_name[i]
        print "moving small windows on ", signal_name
        
        psd_arr = np.zeros((signal.shape[0],1 + (30-6)/3))
        
        for j in range(1 + (30-6)/3):
            
            segment = signal[::, j*3*512:(j+2)*3*512]
            f,psd = welch(segment,fs=512, scaling='density', nperseg=256, noverlap=128, axis=1)
            psd = pd.DataFrame(psd)
            psd_arr[:,j] = psd.sum(axis=1)
            
        psd_arr = pd.DataFrame(psd_arr)
        dfFeature[col+signal_name+'_psd_min'+phase] = psd_arr.min(axis=1)
        dfFeature[col+signal_name+'_psd_max'+phase] = psd_arr.max(axis=1)
        dfFeature[col+signal_name+'_psd_mean'+phase] = psd_arr.mean(axis=1)
        dfFeature[col+signal_name+'_psd_median'+phase] = psd_arr.median(axis=1)
        dfFeature[col+signal_name+'_psd_std'+phase] = psd_arr.std(axis=1)
        dfFeature[col+signal_name+'_psd_var'+phase] = psd_arr.var(axis=1)
        dfFeature[col+signal_name+'_psd_diff_max'+phase] = psd_arr.diff(axis=1).max(axis=1)
        dfFeature[col+signal_name+'_psd_diff_min'+phase] = psd_arr.diff(axis=1).min(axis=1)
        dfFeature[col+signal_name+'_psd_diff_mean'+phase] = psd_arr.diff(axis=1).mean(axis=1)
        dfFeature[col+signal_name+'_psd_diff_std'+phase] = psd_arr.diff(axis=1).std(axis=1)
        
    return dfFeature
