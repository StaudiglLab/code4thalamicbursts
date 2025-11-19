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

from matplotlib.backends.backend_pdf import PdfPages

from core import *
from core.helpers import *
from burst.coreFunctions import *
from scipy.ndimage import gaussian_filter1d

#amplitude of bursts
def getAmp_(
		evoked,			#time series with burst profile		
		):
	evokedEnv=np.abs(scipy.signal.hilbert(evoked))	#get envelope for burst profile	
	
	return np.max(evokedEnv)

#get amplitude and width of the evoked responses
def getAmp(taxis,evokeds):
	nEvokeds=len(evokeds)
	amp=np.zeros(nEvokeds)
	for i in range(nEvokeds):
		amp[i]=getAmp_(evokeds[i])
	return amp	#2.355 to get mom-1 to equivalent FWHM
	


def getEvokedResponse(df_detections,state,filterEvoked=True):
	states=np.array(['wake','REM','NREM'])		
	#load evoked responses	
	evoked=np.load(repo.working_tree_dir+"/burstAmp/outfiles/averageBursts_5cycles_minGap0.0sec.npy")
	nBursts=np.load(repo.working_tree_dir+"/burstAmp/outfiles/nBursts_5cycles_minGap0.0sec.npy")	
	evoked=evoked[:,states==state][:,0]
	nBursts=nBursts[:,states==state][:,0]
		
	#select a single frequency band per contact
	df_detections=getSingleFreqBandPerContact(df_detections)
	
	#select evoked responses corresponding to the bands
	taxis=evoked[-1]
	evoked=evoked[:-1][df_detections.index]	
	nBursts=nBursts[df_detections.index]	
	df_detections['nBursts']=nBursts
		
	uniqCounts=len(np.unique(np.column_stack((df_detections['pID'].values.astype("str"),df_detections['ch_name'].values.astype("str"))),axis=0))
	print("Number of bands:%d"%len(evoked))
	if(uniqCounts!=len(df_detections)):
		print("Warning: more than one band per channel")
	
	#convert to microvolts for channels that are in volts
	std=np.std(evoked,axis=1)
	evoked[std<0.01]*=1e6
	
	#filter time series above 8 Hz
	df_detections=df_detections.reset_index()
	if(filterEvoked):
		for i in range(0,len(evoked)):
			evoked[i]=mne.filter.filter_data(evoked[i],sfreq=1.0/(taxis[1]-taxis[0]),l_freq=8,h_freq=None,verbose=False)

	return taxis,evoked,df_detections

