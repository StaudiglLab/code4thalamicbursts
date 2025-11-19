import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

from fooof import FOOOF

from core import *
from core.helpers import *
from burst.coreFunctions import getSignificantBands,getBurstRate1D	
from burst.plotHelpers import *

import matplotlib
sns.set_palette("deep")
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 

#function to get 1/f subtracted spectrum
def fooof(freqs,spec,aperiodic_mode='knee',peak_width_limits=[2, 25]):	
    fg = FOOOF(peak_width_limits=peak_width_limits,aperiodic_mode=aperiodic_mode, max_n_peaks=1)#peak_width_limits=[1, 12], min_peak_height=0.05, max_n_peaks=3)
    fg.fit(freqs,spec)
    aperiodicModel=fg.get_model(space='linear',component='aperiodic')
    return spec-aperiodicModel

def burstRateOsc(doEM=False,n_fft=512,tBinInSec=0.5,minLengthofStage=300):
    df=getSignificantBands(which='gamma')
    iBand=0

    pID=df['pID'].values
    pID[pID=='p14_followup']='p14'
    df['pID']=pID
    uniqPID=np.unique(pID)
    ch_name=df['ch_name'].values
    freqRange=df[['freqLow','freqHigh']].to_numpy()

    psdSubj=np.zeros((len(uniqPID),n_fft//2))
    dfPSD=pd.DataFrame()
    
    for iPID in range(0,len(uniqPID)):
        if(doEM):
            channelsThis=['dummy']
        else:
            channelsThis=ch_name[pID==pID[iPID]]
            freqRangeThis=freqRange[pID==pID[iPID]]
        
        for iBand in range(0,len(channelsThis)):
            if(doEM):
                taxis,rate,sleepScore=getEMRates(uniqPID[iPID],tBinInSec=tBinInSec)
            else:
                taxis,rate,sleepScore=getBurstRate1D(uniqPID[iPID],ch_name=channelsThis[iBand],freqRange=freqRangeThis[iBand],tBinInSec=tBinInSec)
            
            selmaskSS=sleepScore==5
            
            startStage,widthStage=getClustersFromMask(selmaskSS)
            selmaskStage=widthStage>minLengthofStage/tBinInSec


            #select segments that are more than minimum length
            startStage=startStage[selmaskStage]
            widthStage=widthStage[selmaskStage]
            
            rate=rate.astype("float")
            psd=[]
           
            for i in range(0,len(widthStage)):
                rateThis=rate[startStage[i]:startStage[i]+widthStage[i]]                    
                if(np.mean(rateThis)==0):
                    continue
               
                psd_,freqs=mne.time_frequency.psd_array_welch(rateThis,sfreq=1/tBinInSec,fmin=1e-9,n_fft=n_fft,verbose=False,output='power')
                
                psd.append(psd_)
            psd=np.array(psd)
        dfPSD['freqs']=freqs
        dfPSD['psd_%s'%uniqPID[iPID]]=np.mean(psd,axis=0)
    if(doEM):
        dfPSD.to_csv('outfiles/infraslowPSD_EMs.csv')
    else:
        dfPSD.to_csv('outfiles/infraslowPSD_osc.csv')

        
burstRateOsc()
burstRateOsc(doEM=True)