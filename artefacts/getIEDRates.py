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



#function to get mask for bad data.
def getIEDRate(pID,
        ch_name,
        sfreq=200.0,
        fullMask=False
        ):

    #read in mask file
    
    IEDmask=np.load(rootdir+"/IEDMask/%s_electrode_%s_IEDmask_convolved.npy"%(pID,ch_name[0]))[int(ch_name[1])-1]#save mask without any dilation
    
    #get the starting index for each contagious segment of IEDs
    startIndx,width=getClustersFromMask(IEDmask)    
    
    #read sleep score file
    taxisSS,sleepScore=readSleepScoreFinal(pID)  
    

    #combining N2 and N3 to NREM with code 2
    sleepScore[sleepScore==3]=2
    

    #extrapolate sleep score into relevent time axes
    sleepScoreRaw=scipy.interpolate.interp1d(taxisSS,sleepScore,bounds_error=False,fill_value=-1,kind='nearest')(np.arange(len(IEDmask))/sfreq)
    sleepScoreAtIED=scipy.interpolate.interp1d(taxisSS,sleepScore,bounds_error=False,fill_value=-1,kind='nearest')(startIndx/sfreq)

    rates=pd.DataFrame()
    
    #get rates for each state
    stateKey={'REM':5,'NREM':2,'wake':0}
    for state,sleepScoreCode in stateKey.items():
        rates.loc[0,'frac_%s'%state]=np.mean(IEDmask[sleepScoreRaw==sleepScoreCode])
        rates.loc[0,'rate_%s'%state]=np.sum(sleepScoreAtIED==sleepScoreCode)/(0.5*np.sum(sleepScore==sleepScoreCode)) #denominator is the total time in the sleep stage, in minutes
    rates.loc[0,'all']=np.mean(IEDmask)

    return rates

def writeAllAverage():
    dfRates=pd.DataFrame(columns=['pID','ch_name','frac_NREM','frac_wake','frac_REM','rate_NREM','rate_wake','rate_REM'])
    i=0
    for pID in cohortForPaper:    
        for ch_name in ['L1-L2','L2-L3','L3-L4','R1-R2','R2-R3','R3-R4']: 
            #skip p26, L electrode
            if(pID=='p26' and ch_name[0]=='L'):
                continue
            dfRates.loc[i]=getIEDRate(pID,ch_name).loc[0]
            dfRates.loc[i,'pID']=pID
            dfRates.loc[i,'ch_name']=ch_name
            i+=1
    
    dfRates.to_csv("outfiles/IEDrates.csv")
    

if __name__ == '__main__':
    writeAllAverage()