import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages

from core import *
from core.helpers import *

from coreFunctions import getPeriEventHist
from numba import njit
import pandas as pd
from burst.coreFunctions import *	
		
def writeEventCorrelationAll(sleepscoreSelect=5):	
			
	df_selected=getSignificantBands('gamma')
	
	print("Number of bands:%d"%len(df_selected))

	periEventAll=None
	saccadeAutoCorrAll=None
	
	for i in range(len(df_selected)):
		indx=df_selected.index[i]
		pID=df_selected.loc[indx,'pID']		
		ch_name=df_selected.loc[indx,'ch_name']
		freqLow=df_selected.loc[indx,'freqLow']
		freqHigh=df_selected.loc[indx,'freqHigh']		
		
		delayT,periEventTrue,saccadeAutoCorr,nSaccade,nBurst=getPeriEventHist(pID=pID,ch_name=ch_name,
								maxDelayInSec=60.0,smoothScaleInMs=1000,
								sleepscoreSelect=sleepscoreSelect,
								freqLowBand=freqLow,freqHighBand=freqHigh,
								minWidthInCycles=5,sfreq=200,
								eyeMovIndexToUse='peakVelocityIndex',
								burstTimeToUse='peak')
						
		
		#create appropriate data structures in the first iteration
		if(i==0):			
			periEventAll=np.zeros((len(df_selected)+1,len(delayT)))
			periEventAll[-1]=delayT
			saccadeAutoCorrAll=np.zeros((len(df_selected)+1,len(delayT)))
			saccadeAutoCorrAll[-1]=delayT
		
		periEventAll[i]=periEventTrue
		saccadeAutoCorrAll[i]=saccadeAutoCorr	

	
	
	
	#write peri event histograms
	if(sleepscoreSelect==5):	
		np.save("outfiles/periEventAll_REM.npy",periEventAll)
		np.save("outfiles/saccadeAutoCorrAll_REM.npy",saccadeAutoCorrAll)	
	elif(sleepscoreSelect==0):
		np.save("outfiles/periEventAll_wake.npy",periEventAll)
		np.save("outfiles/saccadeAutoCorrAll_wake.npy",saccadeAutoCorrAll)	
	
writeEventCorrelationAll(sleepscoreSelect=5)	

