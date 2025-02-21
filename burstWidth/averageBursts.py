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

#actual implementation of width computation for bursts
def getAmpWidth_(taxis,			#time axis
		evoked,			#time series with burst profile
		tlim=2.0,		#time extent to consider for burst profile
		crossingZscore=2,	#z-score threshold for selecting region for moment-2 computation
		smoothTimeScaleInSec=0.05	#smoothing timescale; used to threshold and find region for computation
		):
	evokedEnv=np.abs(scipy.signal.hilbert(evoked))	#get envelope for burst profile
	tmask=np.abs(taxis)<=tlim			#mask for selected time range
	
	evokedEnvSmooth=gaussian_filter1d(evokedEnv,smoothTimeScaleInSec/(taxis[1]-taxis[0]))	#smooth envelope
	
	#z-score smooth envelope; use region beyond tlim to compute mean and standard deviation
	evokedEnvSmooth=(evokedEnvSmooth-np.mean(evokedEnvSmooth[np.logical_not(tmask)]))/np.std(evokedEnvSmooth[np.logical_not(tmask)])
	
	#select time range within tlim
	evokedEnv=evokedEnv[tmask]
	evokedEnvSmooth=evokedEnvSmooth[tmask]
	evoked=evoked[tmask]
	taxis=taxis[tmask]
	
	#determine crossing thresholds on both sides of the burst	
	startMask=np.logical_and(taxis<0,evokedEnvSmooth<crossingZscore)	
	if(np.sum(startMask)):
		startTime=np.max(taxis[startMask])
	else:
		print("Cannot find starting time")
		startTime=taxis[0]
	stopMask=np.logical_and(taxis>0,evokedEnvSmooth<crossingZscore)	
	if(np.sum(stopMask)):
		stopTime=np.min(taxis[stopMask])	
	else:
		print("Cannot find stopping time")
		stopTime=taxis[-1]

	#mask to compute moment-2 over
	momMask=np.logical_and(taxis>=startTime,taxis<=stopTime)

	#compute moment-2
	width=np.sqrt(np.sum(taxis[momMask]**2*evokedEnv[momMask])/np.sum(evokedEnv[momMask]))
	
	return np.max(evokedEnv),width,(stopTime-startTime)

#get amplitude and width of the evoked responses
def getAmpWidth(taxis,evokeds):
	nEvokeds=len(evokeds)
	amp,width,zscoreExtent=np.zeros(nEvokeds),np.zeros(nEvokeds),np.zeros(nEvokeds)
	for i in range(nEvokeds):
		amp[i],width[i],zscoreExtent[i]=getAmpWidth_(taxis,evokeds[i])
	return amp,2.355*width,zscoreExtent	#2.355 to get mom-1 to equivalent FWHM
	


def getEvokedResponse(df_detections,state,filterEvoked=True):
	states=np.array(['wake','REM','NREM'])		
	#load evoked responses	
	evoked=np.load(repo.working_tree_dir+"/burstWidth/outfiles/averageBursts_5cycles_minGap0.0sec.npy")
	nBursts=np.load(repo.working_tree_dir+"/burstWidth/outfiles/nBursts_5cycles_minGap0.0sec.npy")	
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

