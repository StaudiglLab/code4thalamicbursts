import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import scipy
import pandas as pd

from core import *
from core.helpers import *

from psd.coreFunctions import getPSD


#function to save average Welch power spectrum for each patient, channel, and brain state	
def getAveragePSD():	

	for pID in ['pthal102']:
													
		df=pd.DataFrame()
		for electrode in ['L','R','scalp']:
			taxis,freqs,powerdata,ch_names,sleepScore=getPSD(pID,electrode=electrode)
			df['freqs']=freqs			
						
			NREMmask=np.logical_or(sleepScore==2,sleepScore==3)
			REMmask=sleepScore==5
			wakemask=sleepScore==0		
			
			#averaging over each state
			powerdataNREM=np.nanmean(powerdata[NREMmask],axis=0)
			powerdataREM=np.nanmean(powerdata[REMmask],axis=0)				
			powerdataWake=np.nanmean(powerdata[wakemask],axis=0)

			#saving to dataframe
			for ich in range(len(ch_names)):				
				df['%s_wake'%ch_names[ich]]=powerdataWake[ich]
				df['%s_REM'%ch_names[ich]]=powerdataREM[ich]				
				df['%s_NREM'%ch_names[ich]]=powerdataNREM[ich]			

		df.to_csv("outfiles/%s_psd.csv"%pID)

getAveragePSD()
