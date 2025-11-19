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


	
'''	

'''	


#return filtered data for given channel
def getFilteredData(pID,
		ch_name,
		freqBand, 		#frequency selection [low,high]
		doHilbert=False,	#perform hilbert transform
		envelope=True		#get envelope of hilbert
		):
		
	#load data and mask for the channel
	raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,ch_name[0])).pick(ch_name).load_data()
	maskBad=getBADMask(pID,ch_name[0],combineChannels=False)[int(ch_name[1])-1]
	
	#if no filtering required return data as is
	if(freqBand[0] is None and freqBand[1] is None):
		d=raw.get_data()[0]	
		d[maskBad]=np.nan	
		return raw.times,d
	
	#filter data	
	raw=raw.filter(l_freq=freqBand[0],h_freq=freqBand[1],n_jobs=-1)
		
	if(doHilbert):		
		#apply hilbert transform
		raw.apply_hilbert(envelope=envelope)
		dAmp=raw.get_data()[0]
		dAmp[maskBad]=np.nan
		return raw.times,dfilt,dAmp
	else:
		dfilt=raw.get_data()[0]	
		dfilt[maskBad]=np.nan	
		return raw.times,dfilt
		
#function to get location of peak voltage in the burst
@njit
def findPeakVoltageIndx(d, #time series
			peakIndx, #index of peak of bursts
			window #duration (in samples) to search around reach burst
			):
	peakIndx=peakIndx.astype("int")
	minIndx=np.zeros_like(peakIndx)
	windowByTwo=(window//2).astype("int")
	for i in range(0,len(peakIndx)):
		minIndx[i]=peakIndx[i]-windowByTwo[i]+np.argmax(d[peakIndx[i]-windowByTwo[i]:peakIndx[i]+windowByTwo[i]])
	return minIndx
	
	
#actual implentation of cutting trials
@njit	
def getTrials_(data, 		#length of trials
		indx,		#location of events
		trialLength,	#length of trials (in number of samples)
		minGap		#minimum gap between events (in number of samples)
		):
		
	trials=np.zeros((len(indx),trialLength*2))
	lastPeakIndx=0
	lengthOfData=len(data)
	for i in range(0,len(indx)):
		thisTrial=data[indx[i]-trialLength:indx[i]+trialLength]
		if(np.sum(np.isnan(thisTrial))==0 and indx[i]>lastPeakIndx+minGap):
			trials[i]=thisTrial
			lastPeakIndx=indx[i]			
		else:
			trials[i]=np.nan
	return trials
	
	
			
#Find maximum deflection around each peak power and return trials
def getTrials(pID,
		ch_name,
		freqLowBand,
		freqHighBand,
		trialLengthInSec,
		minWidthInCycles,
		minGapInSec):
		
	peakFreq,peakTimeIndx,startTimeIndx,stopTimeIndx,width,sleepScoreAtBurst=getSelectedBursts(pID=pID,
												ch_name=ch_name,
												freqLowBand=freqLowBand,
												freqHighBand=freqHighBand,
												minWidthInCycles=minWidthInCycles)
										
	#get raw data as well as bandpass filtered data for the channel
	times,d=getFilteredData(pID,ch_name,freqBand=[None,None])
	times,dfilt=getFilteredData(pID,ch_name,freqBand=[freqLowBand,freqHighBand])	
	sfreq=1.0/(times[1]-times[0])
	
	#remove bursts at the edge of the recording
	noEdgeMask=np.logical_and(peakTimeIndx>2*trialLengthInSec*sfreq,peakTimeIndx<len(times)-trialLengthInSec*sfreq)

	peakFreq,peakTimeIndx,startTimeIndx,stopTimeIndx,width,sleepScoreAtBurst=peakFreq[noEdgeMask], peakTimeIndx[noEdgeMask],startTimeIndx[noEdgeMask],stopTimeIndx[noEdgeMask],width[noEdgeMask],sleepScoreAtBurst[noEdgeMask]
	
	
	#get location of peak voltage				
	peakVoltageIndx=findPeakVoltageIndx(d=dfilt,
						peakIndx=peakTimeIndx,
						window=2*sfreq*1.0/peakFreq #search 1 cycle around each burst
						)
						
	
	#get trials around the peakVoltageIndx of bursts
	trialLength=np.round(sfreq*trialLengthInSec).astype("int")
	minGap=int(np.round(minGapInSec*sfreq))
	taxis=np.arange(-trialLength,+trialLength)/sfreq	
	trials=getTrials_(d,peakVoltageIndx,trialLength,minGap)
	
	#select trials without any artefacts
	selmask=np.sum(np.isnan(trials),axis=1)==0		
	print("Number of bursts selected: %d/%d"%(np.sum(selmask),len(selmask)))
	
	peakVoltageIndx,width,sleepScoreAtBurst=peakVoltageIndx[selmask],width[selmask],sleepScoreAtBurst[selmask]
	trials=trials[selmask]
	
	return taxis,trials,peakVoltageIndx,width,sleepScoreAtBurst

	
	
#get average ERP for each of the three brain states
def getEvoked(pID,
		ch_name,
		freqLowBand,
		freqHighBand,
		minWidthInCycles,
		trialLengthInSec,
		minGapInSec
		):
	taxis,trials,peakIndx,width,sleepScoreAtEvent=getTrials(pID=pID,
								ch_name=ch_name,
								freqLowBand=freqLowBand,
								freqHighBand=freqHighBand,
								trialLengthInSec=trialLengthInSec,
								minWidthInCycles=minWidthInCycles,
								minGapInSec=minGapInSec)
	print(trials.shape)
	states=['wake','REM','NREM']	
	sleepScoreSelects={'wake':[0,0],'REM':[5,5],'NREM':[2,3]}
	evoked=np.zeros((3,len(taxis)))
	ntrials=np.zeros(3)
	#iterating over all three states
	for i in range(0,len(states)):
		sleepmask=np.logical_or(sleepScoreAtEvent==sleepScoreSelects[states[i]][0],sleepScoreAtEvent==sleepScoreSelects[states[i]][1])	
		evoked[i]=np.mean(trials[sleepmask],axis=0)
		ntrials[i]=np.sum(sleepmask)
		
	return taxis,evoked,ntrials

	
#function to save the ERPs for all bursts	
def getAveragesAll(minWidthInCycles=5,   #minimum width of individual bursts
		freqLimInHz=4,		#maximum allowed peak frequency of bursts around the peak frequency of band
		trialLengthInSec=4.0,   #length of trials
		minGapInSec=0.0,		#minimum time interval between subsequent bursts that are averaged
		sfreq=200.0
		):
	df_detections=getSignificantBands()
	

		
	sfreq=200.0
	evokedAll=np.zeros((len(df_detections)+1,3,2*np.round(trialLengthInSec*sfreq).astype("int")))
	
	
	
	nBursts=np.zeros((len(df_detections),3))
	iCountGreater=0	
	
	for i in range(0,len(df_detections)):
		freqLow,freqHigh=df_detections.loc[i,'freqLow'],df_detections.loc[i,'freqHigh']
		freqPeak=df_detections.loc[i,'freqPeak']
		if(freqHigh-freqPeak>freqLimInHz):
			freqHigh=freqPeak+freqLimInHz
			iCountGreater+=1
		if(freqPeak-freqLow>freqLimInHz):
			freqLow=freqPeak-freqLimInHz
			iCountGreater+=1
		taxis,evokedAll[i],nBursts[i]= getEvoked(pID=df_detections.loc[i,'pID'], 	
							ch_name=df_detections.loc[i,'ch_name'],
							freqLowBand=freqLow,
							freqHighBand=freqHigh,
							minWidthInCycles=minWidthInCycles,
							trialLengthInSec=trialLengthInSec,
							minGapInSec=minGapInSec
							)
	
	evokedAll[-1]=np.expand_dims(taxis,axis=0) #last row of array contains the taxis	
	print("Fraction of bands whose widths were adjusted:%d/%d"%(iCountGreater,len(df_detections)))
	if(freqLimInHz==np.inf):
		np.save("./outfiles/averageBursts_%dcycles_minGap%.1fsec.npy"%(minWidthInCycles,minGapInSec),evokedAll)
		np.save("./outfiles/nBursts_%dcycles_minGap%.1fsec.npy"%(minWidthInCycles,minGapInSec),nBursts)
	else:
		np.save("./outfiles/averageBursts_%dHz_%dcycles_minGap%.1fsec.npy"%(2*freqLimInHz,minWidthInCycles,minGapInSec),evokedAll)
		np.save("./outfiles/nBursts_%dHz_%dcycles_minGap%.1fsec.npy.npy"%(2*freqLimInHz,minWidthInCycles,minGapInSec),nBursts)				


#getAveragesAll(freqLimInHz=np.inf,minGapInSec=0)



