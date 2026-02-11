import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
import matplotlib
import seaborn as  sns
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 

def plotSaccades(ax,pID,label=''):
    #the .npy files contains Tobii velocities locked to events detected on F7-F8

    trialsSess1=np.load("outfiles/%s_session%d_eegLockedTrials_data.npy"%(pID,1))
    trialsSess2=np.load("outfiles/%s_session%d_eegLockedTrials_data.npy"%(pID,2))
    taxis=np.load("outfiles/%s_session%d_eegLockedTrials_taxis.npy"%(pID,1))

    trials=np.append(trialsSess1,trialsSess2,axis=0)

    im=ax.imshow(np.abs(trials),aspect='auto',vmax=500,extent=[1e3*taxis[0],1e3*taxis[-1],1,len(trials)])
    ax.axvline(0,ls='--',c='black')
    ax.set_xlabel("Time relative detection on F7-F8 (ms)")
    ax.set_ylabel("Event ID")
    ax.set_title(label)
    plt.colorbar(im,ax=ax,label='Gaze velocity from eye tracker (deg/sec)',location='bottom',pad=0.07)

    return im
    

fig,axs=plt.subplots(1,2,figsize=(10,10))
plotSaccades(ax=axs[0],pID='pthal103',label='patient P15')
plotSaccades(ax=axs[1],pID='pthal106b',label='patient P17')

plt.savefig("figures/Chowdhury_EDF6.jpg",bbox_inches='tight',dpi=300)