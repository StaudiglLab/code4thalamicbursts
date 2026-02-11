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


from core import *
from core.helpers import *
from coreFunctions import getBurstRate2D,getSignificantBands,getOverlaps
from burst.plotHelpers import *
from psd.periodicPowerInBand import getAlignedPeriodicPower
from scipy.stats import wilcoxon

import matplotlib.gridspec as gridspec

import matplotlib
sns.set_palette("deep")
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 


#plot ratio of burst rates in bands at the group level
def plotBurstRateRatioGroupLevel(band,ax):
    #read information for the different bands
    
    df_gamma=getSignificantBands(which=band) 
    pID=df_gamma['pID'].values
    pID[pID=='p14_followup']='p14'
    df_gamma['pID']=pID
    uniqPID=np.unique(pID)
    states=['wake','N2','N3','REM']
    
    burstRates_subj=np.zeros((len(uniqPID),len(states)))
    nSubj=len(uniqPID)
    #get subject-level averages
    for iPID in range(0,len(uniqPID)):
        for iState in range(0,len(states)):
            burstRates_subj[iPID,iState]=np.mean(df_gamma['meanBurstRate_%s'%states[iState]][df_gamma['pID']==uniqPID[iPID]].values)
    #logarithm of burst rates        
    burstRates_subj=np.log10(burstRates_subj)

    #plotting
    colors=[sns.color_palette("deep")[0],sns.color_palette("deep")[2],sns.color_palette("deep")[3],sns.color_palette("deep")[1]]
    violin=sns.violinplot(burstRates_subj,palette=colors,ax=ax,cut=0,alpha=0.75,width=0.5)

    for i in range(0,len(uniqPID)):
        ax.plot(np.arange(len(states)),(burstRates_subj[i]),c='gray',lw=1,marker='o',ms=1,zorder=-999)
    start=1.9
    spacing=0.07
    isig=0
    #plotting significant effects
    ncomparision=(len(states)*(len(states)-1))/2
    print("Number of comparisions:",ncomparision)
    icomp=0
    for i in range(0,len(states)):
        for j in range(i+1,len(states)):
            wilcoxtst=wilcoxon(x=burstRates_subj[:,i], y=burstRates_subj[:,j])
            print(band,states[i],states[j],'pvalue=%.4f\tstatistic=%d'%(wilcoxtst.pvalue*ncomparision,wilcoxtst.statistic))
            pvalue=wilcoxtst.pvalue
            if(pvalue<0.05/ncomparision):
                print("Significant effect:",states[i],states[j])
                ax.plot([i,j],[start+spacing*isig,start+spacing*isig],c='black',marker='|')
                isig+=1           
            icomp+=1
    ax.set_ylabel("Burst Rate (/min)")
    ax.set_yticks([0,1,2],["1","10","100"])
    ax.set_yticks([0,1,2],["1","10","100"])
    ax.set_xticks([0,1,2,3],['wake','N2','N3','REM'])
    ax.set_ylim((-0.6,2.25))
    
if(__name__=='__main__'):    
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    ax[0].set_title("Wake- and REM- Specific Oscillation")
    plotBurstRateRatioGroupLevel('gamma',ax[0])
    #plt.savefig("figures/n2N3_gamma.pdf",bbox_inches='tight',dpi=300)
    #plt.clf()
    #ax=plt.subplot(111)
    ax[1].set_title("Spindles")
    plotBurstRateRatioGroupLevel('spindleInGammaChannels',ax[1])
    ax[0].text(0.05,0.95,'(a)',transform=ax[0].transAxes, fontweight='bold')
    ax[1].text(0.05,0.95,'(b)',transform=ax[1].transAxes, fontweight='bold')
    #plt.savefig("figures/n2N3_spindles.pdf",bbox_inches='tight',dpi=300)
    #plt.clf()
    plt.savefig("figures/N2N3.pdf",bbox_inches='tight',dpi=300)