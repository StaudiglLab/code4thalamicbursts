import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)
from burst.coreFunctions import getSignificantBands

import numpy as np
import mne
import pandas as pd

from scipy.interpolate import interp1d
from core import *
from core.helpers import *
from scipy.stats import wilcoxon


def compareSubjLevel(rate1,rate2,pID):
    uniqPID=np.unique(pID)
    rate1Subj=np.zeros(len(uniqPID))
    rate2Subj=np.zeros(len(uniqPID))
    print("Number of subjects for test:%d"%len(uniqPID))
    for i in range(0,len(uniqPID)):
        selmask=pID==uniqPID[i]
        rate1Subj[i]=np.mean(rate1[selmask])
        rate2Subj[i]=np.mean(rate2[selmask])
    return wilcoxon(x=rate1Subj, y=rate2Subj)

def plotIEDRateRatioGroupLevel(ax):
    #read csv file with rates
    dfRates=pd.read_csv("outfiles/IEDrates.csv")
    
    #get channels with the gamma
    df_gamma=getSignificantBands(which='gamma') 
    dfRates=df_gamma.merge(dfRates,on=['pID','ch_name'],validate='1:1')
    pID=dfRates['pID'].values
    pID[pID=='p14_followup']='p14'
    dfRates['pID']=pID
    uniqPID=np.unique(pID)
    
    states=['wake','NREM','REM']
    burstRates_subj=np.zeros((len(uniqPID),len(states)))
    ncomp=3
    print("Pair-wise comparisions:")
    stats=compareSubjLevel(dfRates['rate_NREM'].values,dfRates['rate_REM'].values,pID)
    print("NREM-REM: p(corrected)=%.4f;statistic=%d"%(stats.pvalue*ncomp,stats.statistic))
    stats=compareSubjLevel(dfRates['rate_NREM'].values,dfRates['rate_wake'].values,pID)
    print("NREM-wake: p(corrected)=%.4f;statistic=%d"%(stats.pvalue*ncomp,stats.statistic))

    stats=compareSubjLevel(dfRates['rate_wake'].values,dfRates['rate_REM'].values,pID)
    print("wake-REM: p(corrected)=%.4f;statistic=%d"%(stats.pvalue*ncomp,stats.statistic))
    
    #getting subject-level rates
    for iPID in range(0,len(uniqPID)):
        for iState in range(0,len(states)):
            burstRates_subj[iPID,iState]=np.mean(dfRates['rate_%s'%states[iState]][dfRates['pID']==uniqPID[iPID]])
            
    #print(burstRates_subj)
    burstRates_subj=np.log10(burstRates_subj)
    colors=[sns.color_palette("deep")[0],sns.color_palette("deep")[2],sns.color_palette("deep")[1]]
    violin=sns.violinplot(burstRates_subj,palette=colors,ax=ax,cut=0,alpha=0.75,width=0.5)

    for i in range(0,len(uniqPID)):
        ax.plot(np.arange(len(states)),(burstRates_subj[i]),c='gray',lw=1,marker='o',ms=1,zorder=-999)

    #marking signficance for REM
    ax.plot([0,2],[1.1,1.1],c='black',marker='|')

    ax.set_ylabel("IED Rate (/min)")
    ax.set_yticks([-2,-1,0,1],["0.01","0.1","1","10"])
    ax.set_yticks([-2,-1,0,1],["0.01","0.1","1","10"])
    ax.set_xticks(np.arange(len(states)),states)

fig,axs=plt.subplots(1,1,figsize=((4,3)),sharex=False,sharey=False,layout='constrained')
plotIEDRateRatioGroupLevel(axs)
plt.savefig("figures/IEDRates_Subj.pdf",bbox_inches='tight',dpi=300.0)
