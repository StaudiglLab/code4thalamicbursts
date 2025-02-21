import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib import ticker
#from scipy.stats import PermutationMethod

from core import *
from core.helpers import *
from averageBursts import *
from burst.coreFunctions import *
import matplotlib.gridspec as gridspec

import matplotlib

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 


def spindleSingle(pID,ch_name):
    taxis,evoked_NREM,df_NREM=getEvokedResponse(getSignificantBands('spindleInGammaChannels'),'NREM',filterEvoked=False)        
    evoked_NREM=evoked_NREM[np.logical_and(df_NREM['pID'].values==pID,df_NREM['ch_name'].values==ch_name)][0] 
    fig,ax = plt.subplots(1,1,figsize=(5,4),sharex=True)   
    ax.plot(taxis,evoked_NREM,lw=1.5,c='C2')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Deflection (microV)")
    ax.set_xlim((-2,2))
    plt.savefig("figures/%s_%s_spindle.png"%(pID,ch_name),bbox_inches='tight')
    print(evoked_NREM.shape)
    '''
    plt.subplots_adjust(hspace=0.5,wspace=0.7)
    fig,axs = plt.subplots(8,8,figsize=(16,16),sharex=True)    
    axs_flat=axs.flatten()
    for i in range(0,len(evoked_NREM)):
        axs_flat[i].plot(taxis,evoked_NREM[i])
        axs_flat[i].grid()
        axs_flat[i].axvline(-0.25,ls='--',c='gray')
        axs_flat[i].axvline(0.25,ls='--',c='gray')
    for i in range(len(evoked_NREM),len(axs_flat)):
        axs_flat[i].set_axis_off()
    
    for i in range(0,8):
        axs[-1,i].set_xlabel("Time (seconds)")
    for i in range(0,8):
        axs[i,0].set_ylabel("Deflection (microV)")
    axs_flat[0].set_xlim((-1.5,1.5))
    plt.show()
    plt.savefig("figures/ThalamicSpindle_ERP_unfiltered.png",dpi=300)
    '''
def spindleCollage():
    
    
    taxis,evoked_NREM,df_NREM=getEvokedResponse(getSignificantBands('spindleInGammaChannels'),'NREM',filterEvoked=True)

    
    nEvoked=len(evoked_NREM)
    
    df_NREM['amp'],df_NREM['width'],df_NREM['zscoreExtent']=getAmpWidth(taxis,evoked_NREM)
    sortIndx=np.argsort(df_NREM['width'])



    df_NREM=df_NREM.sort_values("width")
    print(df_NREM.columns)
    print(df_NREM[['pID','ch_name','freqLow','freqHigh']])    
    taxis,evoked_NREM,df_NREM=getEvokedResponse(getSignificantBands('spindleInGammaChannels'),'NREM',filterEvoked=False)        
    evoked_NREM=evoked_NREM[sortIndx]    
    plt.subplots_adjust(hspace=0.5,wspace=0.7)
    fig,axs = plt.subplots(8,8,figsize=(16,16),sharex=False)    
    axs_flat=axs.flatten()
    for i in range(0,len(evoked_NREM)):
        axs_flat[i].plot(taxis,evoked_NREM[i])
        axs_flat[i].grid()
        axs_flat[i].axvline(-0.25,ls='--',c='gray')
        axs_flat[i].axvline(0.25,ls='--',c='gray')
        axs_flat[i].set_xlim((-1.5,1.5))
    for i in range(len(evoked_NREM),len(axs_flat)):
        axs_flat[i].set_axis_off()
    
    for i in range(0,8):
        axs[-2,i].set_xlabel("Time (seconds)")
        
    for i in range(0,8):
        axs[i,0].set_ylabel("Deflection (microV)")
    
    #plt.show()
    plt.savefig("figures/ThalamicSpindle_ERP_unfiltered.png",dpi=300,bbox_inches='tight')
    
def plotThresholdedWidth():
    fig = plt.figure(figsize=(6, 5))
    plt.subplots_adjust(hspace=0.5,wspace=0.7)
    gs = gridspec.GridSpec(1,1)
    #fig.subplots_adjust(wspace=6.0,hspace=0.35)
    

    ax_corr_width=fig.add_subplot(gs[:, :])
    

    plotCorrelationZScoreExtent(ax_corr_width,niter=1e4)
    plt.show()
    
#spindleSingle('p21','R1-R2')
spindleCollage()    

