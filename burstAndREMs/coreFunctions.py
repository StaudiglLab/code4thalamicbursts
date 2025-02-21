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

from burst.coreFunctions import *
from numba import njit
import pandas as pd


	
@njit
def eventHistogram(event1,event2,maxDelay=100):
	bins=np.arange(-maxDelay,maxDelay)
	hist=np.zeros(len(bins),dtype=float)
	#event1=np.sort(event1)
	#event2=np.sort(event2)
	for i in range(0,len(event1)):
		for j in range(0,len(event2)):
			d=event2[j]-event1[i]
			if(d<-maxDelay):			
				continue
			if(d>maxDelay):
				break	
			hist[d+maxDelay]+=1
	hist/=len(event2)
	return bins,hist
	
		
def getPeriEventHist(pID,
			ch_name,
			maxDelayInSec,		#extent of perievent histogram
			smoothScaleInMs,	#final bin width of perievent histogram
			freqLowBand,		#low edge of frequency band selected for bursts
			freqHighBand,		#high edge of frequency band selected for bursts
			sleepscoreSelect=5,	#sleep score to select
			minWidthInCycles=5,
			sfreq=200.0,
			eyeMovIndexToUse='peakVelocityIndex',
			burstTimeToUse='peak'
			):
		
	#convert to samples
	maxDelay=np.round(maxDelayInSec*sfreq).astype("int")
	smoothScale=np.round(smoothScaleInMs*sfreq/1e3).astype("int")

	#load sleepscore file
	taxisSS,sleepScore=readSleepScoreFinal(pID)
	ssFunc = scipy.interpolate.interp1d(taxisSS,sleepScore, kind='nearest')

	#load bursts
	peakFreq,peakTimeIndx,startTimeIndx,stopTimeIndx,width,sleepScoreAtBurst= getSelectedBursts(pID,ch_name,
					freqLowBand=freqLowBand,
					freqHighBand=freqHighBand,
					minWidthInCycles=minWidthInCycles)
	#select index to use
	if(burstTimeToUse=='peak'):
		burstIndx=peakTimeIndx
	elif(burstTimeToUse=='onset'):
		burstIndx=startTimeIndx
	if(len(burstIndx)==0):
		return None,None
		
	#select bursts at sleep score
	burstIndx=burstIndx[np.logical_and(burstIndx/sfreq>taxisSS[0],burstIndx/sfreq<taxisSS[-1])]	
	sleepScoreAtBurst=ssFunc(burstIndx/sfreq)
	burstIndx=burstIndx[sleepScoreAtBurst==sleepscoreSelect]
	
	
	#rootdir+"/saccadeParams/%s_saccades_alldetections.csv"%pID
	#+"/eyeMovParams/%s_eyeMovEvents_alldetections.csv"%pID
	eyeMovParams=pd.read_csv(rootdir+"/eyeMovParams/%s_eyeMovEvents_alldetections.csv"%pID)
	eyeMovParams=eyeMovParams.sort_values(eyeMovIndexToUse)
	#sort eye movement events
	eyeMovIndex=eyeMovParams[eyeMovIndexToUse].values.astype("int")
	#remove events at edge of recording
	eyeMovIndex=eyeMovIndex[np.logical_and(eyeMovIndex/sfreq>taxisSS[0]+2*maxDelayInSec,eyeMovIndex/sfreq<taxisSS[-1]-2*maxDelayInSec)]
	
	#ensure that eye movement events are far from state transitions 
	sleepScoreAtSaccade=ssFunc(eyeMovIndex/sfreq)
	sleepScoreAtSaccadePre=ssFunc(eyeMovIndex/sfreq-1.5*maxDelayInSec)
	sleepScoreAtSaccadePost=ssFunc(eyeMovIndex/sfreq+1.5*maxDelayInSec)		
	eyeMovIndexSelected=eyeMovIndex[np.logical_and.reduce((sleepScoreAtSaccade==sleepscoreSelect, 
								sleepScoreAtSaccadePre==sleepscoreSelect,
								sleepScoreAtSaccadePost==sleepscoreSelect))]
	
	eyeMovIndex=eyeMovIndex[sleepScoreAtSaccade==sleepscoreSelect]	
	
	print("Number of saccades: %d"%len(eyeMovIndexSelected))
	print("number of bursts: %d"%len(burstIndx))		
	#make sure all indices is sorted
	eyeMovIndex=np.sort(eyeMovIndex)
	burstIndx=np.sort(burstIndx)	
	
	#compute peri event histogram
	delayT,autoCorr=eventHistogram(eyeMovIndexSelected,eyeMovIndex,maxDelay=maxDelay)
	delayT,saccadePeri=eventHistogram(eyeMovIndexSelected,burstIndx,maxDelay=maxDelay)
	
	#blank out central part for eye movement autocorrelation (its unity by default)
	autoCorr[np.abs(delayT/sfreq)<=0.2]=np.nan
	
	#rebin perievent histogram
	
	if(smoothScale>1):
		nBinsAfterSmooth=len(delayT)//smoothScale
		saccadePeri=np.sum(saccadePeri[:nBinsAfterSmooth*smoothScale].reshape(nBinsAfterSmooth,smoothScale),axis=1)
		autoCorr=np.sum(autoCorr[:nBinsAfterSmooth*smoothScale].reshape(nBinsAfterSmooth,smoothScale),axis=1)	
		delayT=np.mean(delayT[:nBinsAfterSmooth*smoothScale].reshape(nBinsAfterSmooth,smoothScale),axis=1)
	delayT=delayT/sfreq
	
	return delayT,1e2*saccadePeri,1e2*autoCorr,len(eyeMovIndexSelected),len(burstIndx)



