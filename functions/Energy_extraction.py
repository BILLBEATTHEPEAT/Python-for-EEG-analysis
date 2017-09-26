############################################################################
# This file is the demo code used to extract features for EEG analysis
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
