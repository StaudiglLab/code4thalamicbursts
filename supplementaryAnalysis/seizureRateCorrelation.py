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
from scipy.stats import wilcoxon,spearmanr,kendalltau,ttest_ind,mannwhitneyu


#helper function for correlation plots
def plotCorrelationHelper(x,y,ax):
	r=scipy.stats.linregress(x,y)
	spearman=spearmanr(x,y)
	pvaltext='p-value=%.2f'%spearman.pvalue	
	xrnge=np.linspace(np.min(x),np.max(x),100)
	ax.scatter(x,y,label='spearman-r = %.2f\n%s'%(spearman.statistic,pvaltext))
	ax.legend(markerscale=0.0)	
      
#function to plot correlations of seizure rates with burst rates

def getBurstRateCorrelation(ax,which='gamma'):
    df_seizure=pd.read_csv("patientdatafiles/patientinfo.csv")
   
    if(which=='gamma'):
        df_osc=getSignificantBands(which='gamma') 
    else:
        df_osc=getSignificantBands(which='spindleInGammaChannels') 

       
    pID=df_osc['pID'].values
    pID[pID=='p14_followup']='p14'
    df_osc['pID']=pID
    uniqPID=np.unique(pID)
    rate1=(df_osc['meanBurstRate_wake'].values+df_osc['meanBurstRate_REM'].values)/2.

    rate1_subj,rate2_subj=np.zeros(len(uniqPID)),np.zeros(len(uniqPID))
    for i in range(0,len(uniqPID)):
        rate1_subj[i]=np.mean(rate1[pID==uniqPID[i]])
        rate2_subj[i]=df_seizure[df_seizure['pID']==uniqPID[i]]['SeizureFrequency'].values[0]
    plotCorrelationHelper(rate1_subj,rate2_subj,ax)
    if(which=='gamma'):
        ax.set_xlabel("Fast Oscillation Burst Rate (/min)")
    else:
        ax.set_xlabel("Spindle Rate (/min)")       
    ax.set_ylabel("Siezure Frequency (/month)")
    ax.set_yscale("log")
    ax.set_xscale("log")

#function to plot correlation of burst rates with epilepsy types
def getCorrelationWithType(ax,which='gamma',whichTypeGrouping='Bilateral'):
    df_seizure=pd.read_csv("patientdatafiles/patientinfo.csv")
    if(which=='gamma'):
        df_osc=getSignificantBands(which='gamma') 
    else:
        df_osc=getSignificantBands(which='spindleInGammaChannels') 

       
    pID=df_osc['pID'].values
    pID[pID=='p14_followup']='p14'
    df_osc['pID']=pID
    uniqPID=np.unique(pID)
    rate1=(df_osc['meanBurstRate_wake'].values+df_osc['meanBurstRate_REM'].values)/2.

    rate1_subj,isInGroup=np.zeros(len(uniqPID)),np.zeros(len(uniqPID),dtype=bool)
    for i in range(0,len(uniqPID)):
        rate1_subj[i]=np.log10(np.mean(rate1[pID==uniqPID[i]]))
        isInGroup[i]=df_seizure[df_seizure['pID']==uniqPID[i]][whichTypeGrouping].values[0]=='Y'
    print(which,whichTypeGrouping,mannwhitneyu(rate1_subj[isInGroup],rate1_subj[np.logical_not(isInGroup)]))
    #print(np.sum(isInGroup))
    
    sns.boxplot([rate1_subj[isInGroup],rate1_subj[np.logical_not(isInGroup)]],ax=ax)
    
    ax.scatter(1-isInGroup.astype("int"),rate1_subj,marker='o',zorder=999,c='black')
    if(whichTypeGrouping=='Bilateral'):
        ax.set_xticks([0,1],['Bilateral','Unilateral'])
    elif(whichTypeGrouping=='FTLE'):
        ax.set_xticks([0,1],['FTLE','not FTLE'])    
    ax.set_yticks([0,1,2],[1,10,100])
    if(which=='gamma'):
        
        ax.set_ylabel("Fast Oscillation Burst Rate (/min)")
    else:
        
        ax.set_ylabel("Spindle Rate (/min)")     


fig,axs=plt.subplots(3,2,figsize=(10,12))
getBurstRateCorrelation(axs[0,0],which='gamma')
getBurstRateCorrelation(axs[0,1],which='spindle')
axs[0,0].set_title("Wake and REM specific Fast Oscillations ")
axs[0,1].set_title("Spindles")
getCorrelationWithType(axs[1,0],which='gamma',whichTypeGrouping='Bilateral' )
getCorrelationWithType(axs[1,1],which='spindle',whichTypeGrouping='Bilateral' )
getCorrelationWithType(axs[2,0],which='gamma',whichTypeGrouping='FTLE' )
getCorrelationWithType(axs[2,1],which='spindle',whichTypeGrouping='FTLE' )
labels=["(a)","(b)","(c)","(d)","(e)","(f)"]
for i in range(0,6):
    ax=axs.flatten()[i]
    ax.text(0.9,0.92,labels[i],transform=ax.transAxes, fontweight='bold')
plt.savefig("figures/epilepsyType.pdf",bbox_inches='tight')

