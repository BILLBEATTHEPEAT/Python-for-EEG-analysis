def calcNormalizedFFT(epoch,lvl,nt,fs=512):
    lseg = np.round(nt/fs*lvl).astype('int')
    D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis = 1))
    D[:,0]=0
    D /= D.sum()
    return D

def calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs=512, percent=0.5):
    # find the spectral edge frequency
    sfreq = fs
    tfreq = 40
    ppow = percent
    
    topfreq = int(round(nt/sfreq*tfreq)) + 1
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    A = np.cumsum(D[:topfreq, :], axis = 1)
    B = A - (A.max()*ppow)
    spedge = np.min(np.abs(B), axis = 1)
    spedge = (spedge - 1)/(topfreq - 1)*tfreq
    
    return spedge
  
def SEF(phase, col, eeg, dfFeature):

    lvl = np.array([0.4,4,8,12,30,70,180])
    [nc,nt] = eeg.shape
    
    sef1 = np.zeros((eeg.shape[0],281))
    sef2 = np.zeros((eeg.shape[0],281))
    
    for index, row in eeg.iterrows():
        sef1[index] = calcSpectralEdgeFreq(row.reshape(1,-1), lvl, nt, nc, fs=512, percent=0.5)
        sef2[index] = calcSpectralEdgeFreq(row.reshape(1,-1), lvl, nt, nc, fs=512, percent=0.95)
    sefd = sef2 - sef1

    dfFeature[col+'_SpectralEdgeFrequency95_mean_'+phase] = np.mean(sef2,axis=1)
    dfFeature[col+'_SpectralEdgeFrequency95_max_'+phase] = np.max(sef2,axis=1)
    dfFeature[col+'_SpectralEdgeFrequency95_min_'+phase] = np.min(sef2,axis=1)
    dfFeature[col+'_SpectralEdgeFrequency95_var_'+phase] = np.var(sef2,axis=1)
    dfFeature[col+'_SpectralEdgeFrequency50_mean_'+phase] = np.mean(sef1,axis=1)
    dfFeature[col+'_SpectralEdgeFrequency50_max_'+phase] = np.max(sef1,axis=1)
    dfFeature[col+'_SpectralEdgeFrequency50_min_'+phase] = np.min(sef1,axis=1)
    dfFeature[col+'_SpectralEdgeFrequency50_var_'+phase] = np.var(sef1,axis=1)
    dfFeature[col+'_SpectralEdgeFrequencyDiff_mean_'+phase] = np.mean(sefd,axis=1)
    dfFeature[col+'_SpectralEdgeFrequencyDiff_mean_'+phase] = np.max(sefd,axis=1)
    dfFeature[col+'_SpectralEdgeFrequencyDiff_mean_'+phase] = np.min(sefd,axis=1)
    dfFeature[col+'_SpectralEdgeFrequencyDiff_mean_'+phase] = np.var(sefd,axis=1)
    
    return dfFeature
