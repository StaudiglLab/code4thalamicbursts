import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import os
import mne
import scipy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
from numba import njit,jit
import numba as nb
import numpy as np
from scipy.interpolate import interp1d

from core import *



#function to read file with sleep score 
def readSleepScoreFile(fname,scoreSamplingInterval=30.0):
	sleepScore=np.loadtxt(fname,usecols=[0])	
	return np.arange(0,len(sleepScore))*scoreSamplingInterval+scoreSamplingInterval/2.,sleepScore

#function to get location peak deflection
@njit
def getPeakLocations(x,startIndx,stopIndx):
	peaks=np.zeros_like(startIndx)
	for ipeak in range(0,len(peaks)):
		peaks[ipeak]=startIndx[ipeak]+np.argmax(np.abs(x[startIndx[ipeak]:stopIndx[ipeak]]))
	return peaks
	
#function to get clusters in the mask that have True 
@njit
def getClustersFromMask(mask):
	mask=mask.astype("int")
	#if everything is 1
	if(np.mean(mask)==1.0):
		return np.array([0]),np.array([len(mask)])
	#if everything is 0
	elif(np.mean(mask)==0.0):
		return np.array([0]),np.array([0])
		
	ediff=(mask[1:]-mask[:-1])
	startIndx=np.arange(len(ediff))[ediff>0]
	stopIndx=np.arange(len(ediff))[ediff<0]
	
	#if a cluster includes the first point	
	if(len(startIndx)<len(stopIndx) or (len(stopIndx)>0 and startIndx[0]>stopIndx[0])):
		startIndx=np.append([-1],startIndx)
	#if a cluster includes the last point
	if(len(startIndx)>len(stopIndx)):
		stopIndx=np.append(stopIndx,len(ediff))
		
	return startIndx+1,stopIndx-startIndx

#actual implementation of function to get running Median Absolute Deviation
@njit
def runningMAD_(x,			#time series
		windowLength,		#size of window
		nSamplesPerWindow	#number of points to sample per window 
		):
	winByTwo=windowLength//2
	taxis=np.arange(winByTwo,len(x)-winByTwo-1,windowLength//nSamplesPerWindow)
	median=np.zeros(len(taxis))
	mad=np.zeros(len(taxis))
	for i in range(len(taxis)):
		idata=taxis[i]
		xThis=x[idata-winByTwo:idata+winByTwo+1]
		median[i]=np.median(xThis)
		mad[i]=np.median(np.abs(xThis-median[i]))		
	return taxis,median,mad

#function to get running Median Absolute Deviation
def runningMAD(x,			#time series
		windowLength,		#size of window
		nSamplesPerWindow=5	#number of points to sample per window 
		):
	taxis,median,mad=runningMAD_(x,windowLength,nSamplesPerWindow)
	#interpolate median and MAD between sampled points
	median=np.interp(np.arange(len(x)),taxis,median)
	mad=np.interp(np.arange(len(x)),taxis,mad)		
	return median,mad
	
#function to impose additional criteria 
@njit
def isValidSaccade(peaks,startIndx,stopIndx, #peak, start, and stop indices of the events
		deflection,			#voltage deflection time series
		velocity,			#velocity time series
		hasEventMask,			#mask that keeps track of events already detected
		# see getEyeMovParamsFromScalp for description of the following parameters
		durationForLocalRMS,		
		amplitudeThresholdLocal,
		velocityThresholdLocal,
		relativeDeviationThreshold
		):
	
	
	selmaskPeaks=np.zeros(len(peaks),dtype=nb.boolean)
			
	localSNRVelocity=np.zeros(len(peaks))
	localSNRAmplitude=np.zeros(len(peaks))	
								
	for ipeak in range(0,len(peaks)):
			
		ton,toff=startIndx[ipeak],stopIndx[ipeak]
		peakPos=peaks[ipeak]
		#if an event has already been detected in these samples then no further checks required
		if(np.sum(hasEventMask[ton:toff])>0):
			selmaskPeaks[ipeak]=False
			continue
		
		#compute local velocity
		localRMSVelocity=np.sqrt((np.mean(velocity[ton-durationForLocalRMS:ton]**2)+np.mean(velocity[toff:toff+durationForLocalRMS]**2))/2)
		localSNRVelocity[ipeak]=velocity[peakPos]/localRMSVelocity

		#compute maximum of the deflection before and after eye movement event
		amplitude=np.abs(deflection[toff]-deflection[ton])
		deflectionPre=deflection[ton-durationForLocalRMS:ton]
		deflectionPost=deflection[toff:toff+durationForLocalRMS]		
		dev=max(np.max(deflectionPost)-np.min(deflectionPost),np.max(deflectionPre)-np.min(deflectionPre))	
		
		
		#compute local amplitude SNR
		stdPre=np.std(deflectionPre)
		stdPost=np.std(deflectionPost)		
		localSNRAmplitude[ipeak]=amplitude/np.sqrt((stdPre**2+stdPost**2)/2.)
			
			
		if(np.abs(localSNRVelocity[ipeak])>velocityThresholdLocal 		#local velocity threshold
				and np.abs(localSNRAmplitude[ipeak])>amplitudeThresholdLocal 	#local amplitude threshold	
				and dev<relativeDeviationThreshold*amplitude		#maximum deflection threshold
				):
			selmaskPeaks[ipeak]=True
			hasEventMask[ton:toff]=True
		else:
			selmaskPeaks[ipeak]=False
			
	return selmaskPeaks,hasEventMask,localSNRVelocity,localSNRAmplitude

#function for actual implementation 
#see getEyeMovParamsFromScalp for description of input parameters
def _getEyeMovParams(deflection, 	#time series of voltages
			MADwindowInSamples,
			smoothScaleRange,
			minVelocity,
			minWidth,
			maxWidth,
			velocityOnsetOffsetThreshold,
			durationForLocalRMS,
			amplitudeThresholdLocal,
			velocityThresholdLocal,
			relativeDeviationThreshold
			):
	
	#mask that keeps track of eye movement events already detected
	hasEventMask=np.zeros(len(deflection),dtype=bool)
	
	eyeMovParamsAllScales=[]
	scales=np.arange(smoothScaleRange[0],smoothScaleRange[1])
	
	for iScale in range(len(scales)):	
		scale=scales[iScale]
		print("Running scale %d"%scales[iScale])

		#getting velocity via convolution 
		kernal=np.linspace(-1,1,scale)		
		velocity=np.convolve(deflection,kernal,mode='same')
		
		#getting running Median Absolute Deviation and normalizing it by it
		med,mad=runningMAD(velocity,MADwindowInSamples)		
		velocity=(velocity-med)/mad
	
		#get positive deflections
		maskVelThreshPos=velocity>=velocityOnsetOffsetThreshold
		startIndxPos,widthPos=getClustersFromMask(maskVelThreshPos)
		
		#get negative deflections
		maskVelThreshNeg=velocity<=-velocityOnsetOffsetThreshold
		startIndxNeg,widthNeg=getClustersFromMask(maskVelThreshNeg)

		#concat positive and negative deflections
		startIndx=np.append(startIndxPos,startIndxNeg)
		width=np.append(widthPos,widthNeg)	
		stopIndx=startIndx+width
		
		#get locations of peak velocity within the initial selection
		peaks=getPeakLocations(velocity,startIndx,stopIndx)	
		
		#get events that meet initial event selections (width criteria and global velocity threshold)
		maskSel=np.logical_and(np.logical_and(width>minWidth,width<maxWidth),np.abs(velocity[peaks])>minVelocity)
		maskSel=np.logical_and(maskSel,np.logical_not(np.isinf(velocity[peaks])))
		#exclude events at the edge of the recording
		maskSel=np.logical_and(maskSel,np.logical_and(startIndx>durationForLocalRMS+scale,stopIndx<len(deflection)-durationForLocalRMS-scale))
		
		peaks=peaks[maskSel]
		startIndx=startIndx[maskSel]
		stopIndx=stopIndx[maskSel]

		#further refine selection
		selmaskPeaks,hasEventMask,localSNRVelocity,localSNRAmplitude=isValidSaccade(peaks=peaks,startIndx=startIndx,stopIndx=stopIndx,
							deflection=deflection,velocity=velocity,hasEventMask=hasEventMask,
							durationForLocalRMS=durationForLocalRMS,
							relativeDeviationThreshold=relativeDeviationThreshold,
							velocityThresholdLocal=velocityThresholdLocal,
							amplitudeThresholdLocal=amplitudeThresholdLocal)
		
		
		#put all detections into a dataframe
		if(np.sum(selmaskPeaks)==0):
			continue
		eyeMovParams=pd.DataFrame()	
		eyeMovParams['peakVelocity'],eyeMovParams['peakVelocityIndex']=velocity[peaks][selmaskPeaks],peaks[selmaskPeaks]
		eyeMovParams['onsetIndex'],eyeMovParams['offsetIndex']=startIndx[selmaskPeaks],stopIndx[selmaskPeaks]
		eyeMovParams['baseline']=deflection[startIndx[selmaskPeaks]]
		eyeMovParams['amplitude']=deflection[stopIndx[selmaskPeaks]]-eyeMovParams['baseline']
		eyeMovParams['detectionScale']=np.ones(len(eyeMovParams))*scale
		if(len(eyeMovParams)>0):
			eyeMovParamsAllScales.append(eyeMovParams)
		
	#combine detections across different scales
	eyeMovParams=pd.concat(eyeMovParamsAllScales)
	return eyeMovParams



def plotSaccadeRate(saccadeTimes,taxis,sleepScore):
	fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,4),sharex=True)
	ax2=ax.twinx()
	sleepScore[sleepScore>5]=np.nan
	ax.hist(saccadeTimes/3600.,bins=np.arange(np.min(taxis)/3600.,np.max(taxis)/3600.,10/60.))	
	ax2.plot(taxis/3600.,sleepScore,c='black')
	ax2.set_yticks(sleepScoringList,sleepLabels)	
	ax.set_yticks(ticks=ax.get_yticks(),labels=ax.get_yticks()/10.)
	ax.minorticks_on()
	ax.set_ylabel("saccade rate(/min)")
	ax.set_xlabel("Time (hour)")	
	return fig


	
def getEyeMovParamsFromScalp(rawfname, 			#name of filename with raw EEG
			outfname,			#outfilename for csv file with saccade parameters
			sleepscorefile,			#path to file with sleepscore	
			figname=None,			#path to save saccade rates (don't save if None)
			smoothScales=[2,26],		#range of smoothing scales, in samples
			MADwindowInMinutes=5.0,		#window to compute Median Absolute Deviation, in minutes
			velocityThreshold=7,		#peak velocity threshold for initial event selection
			velocityOnsetOffsetThreshold=2, #threshold used to identify onset and offset of events
			minWidthInMs=20,		#minimum duration of eye movement events
			maxWidthInMs=750,		#maximum duration of eye movement events
			velocityThresholdLocal=5,	#local velocity threshold 
			minAmplitude=5,			#local amplitude threshold
			durationForLocalRMSInMs=200,	#duration in milliseconds over which to compute RMS for the above two thresholds.
							#The RMS is computed by taking the following time interval:
							#[EMonset to EMonset-durationForLocalRMSInMs] and [EMoffset to EMoffset+durationForLocalRMSInMs]
			relativeDeviationThreshold=1.0, #events with relative deflection above relativeDeviationThreshold*event_deflection 
							#in the time intervals [EMonset to EMonset-durationForLocalRMSInMs] or 
							#[EMoffset to EMoffset+durationForLocalRMSInMs] are rejected
			sfreqOut=200.0,			#sampling frequency for output timestamps (indices of events are up/down sampled accordingly)
			channelToUse='F7-F8',
			filterData=False):
			
	#load raw data and get eeg channel to use
	raw=mne.io.read_raw(rawfname)	
	times=raw.times	
	sfreq=raw.info['sfreq']
	if(sfreqOut is None):
		sfreqOut=raw.info['sfreq']		
	
	if(channelToUse=='bipolarFrontal'):
		d=raw.get_data('F7')[0]-raw.get_data('F8')[0]
	else:
		d=raw.get_data(channelToUse)[0]		
		
	if(filterData): #not used for thalamus patients
		d=mne.filter.filter_data(d,sfreq=sfreq,l_freq=None,h_freq=40.0)
		d=mne.filter.notch_filter(d, Fs=sfreq,freqs=[25],notch_widths=4.0)
		d=mne.filter.resample(d,down=sfreq/100.)	

		sfreq=100.0
		times=np.arange(0,len(d))/sfreq	
		

	durationForLocalRMS=int(sfreq*durationForLocalRMSInMs/1e3)

	#run actual detection
	eyeMovParams=_getEyeMovParams(deflection=d,
					MADwindowInSamples=MADwindowInMinutes*60*sfreq,
					smoothScaleRange=smoothScales,
					minVelocity=velocityThreshold,
					amplitudeThresholdLocal=minAmplitude,
					minWidth=int(minWidthInMs*sfreq/1e3),
					maxWidth=int(maxWidthInMs*sfreq/1e3),
					velocityOnsetOffsetThreshold=velocityOnsetOffsetThreshold,
					velocityThresholdLocal=velocityThresholdLocal,
					durationForLocalRMS=durationForLocalRMS,
					relativeDeviationThreshold=relativeDeviationThreshold)
	

	print("Number of eye movement events: %d"%len(eyeMovParams))

	
	#read sleep score file to get sleep score at events
	taxisSS,sleepScoreThirtySec=readSleepScoreFile(sleepscorefile)
	
	#plot burst rate figure if asked for
	if(not figname is None):
		fig=plotSaccadeRate(times[eyeMovParams['peakVelocityIndex'].values.astype("int")],taxisSS,sleepScoreThirtySec)
		fig.savefig(figname,bbox_inches='tight')
	#get sleep score at detected events	
	sleepScoreEvents=scipy.interpolate.interp1d(taxisSS,sleepScoreThirtySec,kind='nearest',fill_value=np.nan,bounds_error=False)(times[eyeMovParams['peakVelocityIndex'].astype("int")])
	
	#convert index to output sampling interval
	eyeMovParams['peakVelocityIndex']=np.round((times[eyeMovParams['peakVelocityIndex'].values.astype("int")]-times[0])*sfreqOut).astype("int")
	eyeMovParams['onsetIndex']=np.round((times[eyeMovParams['onsetIndex'].values.astype("int")]-times[0])*sfreqOut).astype("int")	
	eyeMovParams['offsetIndex']=np.round((times[eyeMovParams['offsetIndex'].values.astype("int")]-times[0])*sfreqOut).astype("int")
	
	eyeMovParams.insert(len(eyeMovParams.columns), 'sleepScore',sleepScoreEvents)	
	
	eyeMovParams.to_csv(outfname)



