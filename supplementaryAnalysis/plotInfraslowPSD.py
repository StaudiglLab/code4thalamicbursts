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
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from fooof import FOOOF

from core import *
from core.helpers import *
from burst.coreFunctions import getBurstRate2D,getSignificantBands,getOverlaps,	getBurstRate1D,getClustersFromMask	
from burst.plotHelpers import *
from psd.periodicPowerInBand import getAlignedPeriodicPower
from scipy.stats import wilcoxon
from scipy.ndimage import gaussian_filter
import matplotlib.gridspec as gridspec

import matplotlib
sns.set_palette("deep")
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 


def fooof(freqs,spec,aperiodic_mode='knee',peak_width_limits=[2, 25]):	
    fg = FOOOF(peak_width_limits=peak_width_limits,aperiodic_mode=aperiodic_mode, max_n_peaks=1)#peak_width_limits=[1, 12], min_peak_height=0.05, max_n_peaks=3)
    fg.fit(freqs,spec)
    aperiodicModel=fg.get_model(space='linear',component='aperiodic')
    #plt.plot(freqs,spec)
    #plt.plot(freqs,aperiodicModel)
    #plt.show()
    return spec-aperiodicModel

def getEMRates(pID,eyeMovIndexToUse='peakVelocityIndex',sfreq=200.0,tBinInSec=1.0):
    taxisSS,sleepScore=readSleepScoreFinal(pID)
    eyeMovParams=pd.read_csv(rootdir+"/eyeMovParams/%s_eyeMovEvents_alldetections.csv"%pID)
    eyeMovParams=eyeMovParams.sort_values(eyeMovIndexToUse)
    eyeMovIndx=eyeMovParams[eyeMovIndexToUse]
    rate,taxis=np.histogram(eyeMovIndx/sfreq,bins=np.arange(taxisSS[0],taxisSS[-1],tBinInSec))
    taxis=(taxis[1:]+taxis[:-1])/2.
    sleepScore=interp1d(taxisSS,sleepScore,kind='nearest')(taxis)
    return taxis,rate,sleepScore		



def burstRateOsc(doEM=False):
    
    if(doEM):
        dfPSD=pd.read_csv('outfiles/infraslowPSD_EMs.csv')
    else:
        dfPSD=pd.read_csv('outfiles/infraslowPSD_osc.csv')
    freqs=dfPSD['freqs'].values
    psdSubj=np.swapaxes(dfPSD.to_numpy(),0,1)
    #removing index and freqs columns
    psdSubj=psdSubj[2:]

    psdSubj=psdSubj[:,np.logical_and(freqs>freqs[0],freqs<0.5)]  
    freqs=freqs[np.logical_and(freqs>freqs[0],freqs<0.5)] 

    for i in range(0,len(psdSubj)):
        psdSubj[i]=fooof(freqs,psdSubj[i])
   
    F_obs,cluster,cluster_pv,H0=mne.stats.permutation_cluster_1samp_test(psdSubj,n_permutations=10000,tail=1,out_type='mask')
    
    selmask=cluster_pv<0.05
    cluster=np.array(cluster)

    mean=np.mean(psdSubj,axis=0)
    sem=np.std(psdSubj,axis=0)/np.sqrt(len(psdSubj))

    return freqs,mean,sem,cluster[selmask],cluster_pv[selmask]

freqs,meanEM,semEM,clusterEM,pvalueEM=burstRateOsc(doEM=True)

freqs,meanBR,semBR,clusterBR,pvalueBR=burstRateOsc(doEM=False)

plt.plot(freqs,meanBR,c='C0',label='thalmic bursts')
print("bursts")
for i in range(0,len(clusterBR)):
    clust=freqs[clusterBR[i,0]]
    print(clust,pvalueBR[i])
    plt.plot(clust,clust*0-0.05,c='C0',lw=3)
plt.fill_between(freqs,meanBR-semBR,meanBR+semBR,fc='C0',alpha=0.5)
plt.plot(freqs,meanEM,c='C1',label='Rapid EMs')
print("EMs")

for i in range(0,len(clusterEM)):
    clust=freqs[clusterEM[i,0]]
    print(clust,clusterEM[i])
    plt.plot(clust,clust*0-0.04,c='C1',lw=3)

plt.fill_between(freqs,meanEM-semEM,meanEM+semEM,fc='C1',alpha=0.5)
plt.axhline(0,ls='--',c='black')

plt.legend()
plt.xscale("log")
plt.xticks([1e-2,1e-1],["0.01","0.1"])
plt.xlabel("Freq (Hz)")
plt.ylabel("Power (A.U.)")
#plt.yscale("log")
plt.savefig("figures/slowOsc.pdf",dpi=300.0,bbox_inches='tight')
#plt.show()
