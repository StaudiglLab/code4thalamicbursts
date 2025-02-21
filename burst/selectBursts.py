import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import scipy
import pandas as pd

from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure

from core import *
from core.helpers import *

@njit
def detectHarmonic(peakFreq,
		startFreq,
		stopFreq,
		peakTime,
		startTime,
		stopTime,
		peakAmp,
		freqSmear=1.0 #accept bursts as harmonic if burst frequency is within freqSmear of half of the frequency
		):
	#assumes everything sorted by startTimeIndx
	isHarmonic=np.zeros(len(peakTime))
	jstart=0
	bufferTime=60 #just for computation purposes
	for i in range(0,len(peakTime)):
		peakFreqByTwo=peakFreq[i]/2
		if(peakFreqByTwo<3): #less than range of frequencies
			continue
		start=startTime[i]
		stop=stopTime[i]
		peakTimeThis=peakTime[i]
		peakAmpThis=peakAmp[i]
				
		while(jstart>0 and stopTime[jstart]>start-bufferTime): 
			jstart-=1
		for j in range(jstart,len(peakTime)):
			if(stopTime[j]<start-bufferTime):
				jstartnew=j
				continue
			elif(startTime[j]>stop): #don't look anymore, all later bursts start after stop
				break
			#condition for harmonic burst
			if(peakAmp[j]>peakAmpThis and startTime[j]<peakTimeThis and stopTime[j]>=peakTimeThis):
				if(np.abs(peakFreqByTwo-peakFreq[j])<freqSmear):
					isHarmonic[i]=1
					break
		jstart=jstartnew				
	return isHarmonic>0
	
@njit
def hasOverlap(start1,stop1,start2,stop2):
	hasoverlap_=(start1<=start2 and start2<=stop1) or (start1<=stop2 and stop2<=stop1) or (start2<=start1 and start1<=stop2)
	return hasoverlap_
@njit
def removeDuplicates(peakFreq,startFreq,stopFreq,peakTime,startTime,stopTime,peakAmp):
	#assumes everything sorted by startTimeIndx
	selmask=np.ones(len(peakFreq))
	jstart=0
	bufferTime=60 #just for computation purposes
	for i in range(0,len(peakTime)):
		pFreq=peakFreq[i]
		start=startTime[i]
		stop=stopTime[i]
		peak=peakTime[i]
		pAmp=peakAmp[i]
		while(jstart>0 and stopTime[jstart]>peak-bufferTime):
			jstart-=1
		for j in range(jstart,len(peakTime)):
			if(stopTime[j]<start-bufferTime or i==j):
				jstartnew=j
				continue
			elif(startTime[j]>stop): #don't look anymore, all later bursts start after stop
				break
			#overlap condition is met if the peak of one burst falls within the extent of another.
			#higher amplitude burst is retained
			if(startTime[j]<=peak and peak<=stopTime[j]):#hasOverlap(start,stop,startTime[j],stopTime[j])):
				if(startFreq[j]<=pFreq and pFreq<=stopFreq[j] and peakAmp[j]>pAmp): #
					selmask[i]=0
					break
		jstart=jstartnew
	return selmask>0
			
@njit
def mergeBursts(peakFreq,
		startFreq,
		stopFreq,
		peakTime,
		startTime,
		stopTime,
		peakAmp,
		timeDistInCycle, #temporal threshold for merging
		freqDist	 #spectral threshold for merging
		):
		
	#assumes everything sorted by startTimeIndx
	timeDistInSec=timeDistInCycle*(1.0/peakFreq)
	selmask=np.ones(len(peakFreq))
	startTimeNew=np.zeros(len(startTime))
	stopTimeNew=np.zeros(len(stopTime))
	jstart=0
	bufferTime=60 #just for computation purposes
	
	for i in range(0,len(peakTime)):
		pFreq=peakFreq[i]
		start=startTime[i]
		stop=stopTime[i]
		peak=peakTime[i]
		pAmp=peakAmp[i]
		timeDist=timeDistInSec[i]
		
		while(jstart>0 and stopTime[jstart]>start-bufferTime): 
			jstart-=1
		for j in range(jstart,len(peakTime)):
			if(stopTime[j]<start-bufferTime or i==j):
				jstartnew=j
				continue
			elif(startTime[j]>stop+timeDist): #don't look anymore, all later bursts start after stop+timeDist
				break
			
			#continue if frequencies of burst not close enough or
			#if candidate burst has higher amplitude (bursts merged into highest amplitude one) 
			#or if candidate burst has already been merged	
			if(np.abs(pFreq-peakFreq[j])>freqDist or pAmp<peakAmp[j] or selmask[j]==0): 
				continue
				
			if(hasOverlap(start,stop,startTime[j],stopTime[j])): #if bursts overlap
				selmask[j]=0
				start=startTime[j] if startTime[j]<start else start
				stop=stopTime[j] if stopTime[j]>stop else stop							
			elif(np.abs(stopTime[j]-start)<=timeDist):        #if earlier burst is close enough					
				selmask[j]=0
				start=startTime[j]
			elif(np.abs(startTime[j]-stop)<=timeDist): #if later burst is close enough							
				selmask[j]=0
				stop=stopTime[j]
		jstart=jstartnew
		startTime[i]=start
		stopTime[i]=stop
	return selmask>0,startTime,stopTime
										
												   
def selectBurstsFromEvents(pID,
			ch_name,
			min_cycle=5,
			thresholdAmp=3,	
			minFreqWidth=2,
			maxFreqWidth=20,
			timeDistInCycleForMerge=1.5,
			freqDistForMerge=1, #in Hz
			freqrange=[8,48.0],
			sfreq=200.0):


	#load bursts and sort by starting time
	data=np.load(rootdir+"/burstFromMorlet/%s_%s.npy"%(pID,ch_name))
	startTimeIndx=data[:,4]
	data=data[np.argsort(startTimeIndx)]
	peakFreq,startFreq,stopFreq,peakTime,startTimeIndx,stopTimeIndx,peakAmp=data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],data[:,6]
		
		
		
		
	#merge bursts
	mergedBurstMask,startTimeMerged,stopTimeMerged=mergeBursts(peakFreq=peakFreq,
										startFreq=startFreq,
										stopFreq=stopFreq,
										peakTime=peakTime,
										startTime=startTimeIndx/sfreq,
										stopTime=stopTimeIndx/sfreq,
										peakAmp=peakAmp,
										timeDistInCycle=timeDistInCycleForMerge,
										freqDist=freqDistForMerge)
		
	peakFreq,startFreq,stopFreq,peakTime,startTimeIndx,stopTimeIndx, peakAmp= peakFreq[mergedBurstMask],startFreq[mergedBurstMask], stopFreq[mergedBurstMask],peakTime[mergedBurstMask],startTimeIndx[mergedBurstMask], stopTimeIndx[mergedBurstMask],peakAmp[mergedBurstMask]
	
	startTimeIndx,stopTimeIndx=(startTimeMerged[mergedBurstMask]*sfreq).astype("int"),(stopTimeMerged[mergedBurstMask]*sfreq).astype("int")
	
	print("Post-Merge fraction: %.2f"%np.mean(mergedBurstMask))		
		
	
	
	#select bursts based on various criteria (ncycles, frequency widths, peak amplitude)

	ncycles=peakFreq*(stopTimeIndx-startTimeIndx)/sfreq
	freqWidth=stopFreq-startFreq
	selmask=peakAmp>thresholdAmp
	selmask=np.logical_and(selmask,np.logical_and(freqWidth>minFreqWidth,freqWidth<maxFreqWidth))	
	selmask=np.logical_and(selmask,ncycles>min_cycle)
	print("Selecting %.2f percent of all bursts"%(100*np.mean(selmask)))
	peakFreq,startFreq,stopFreq,peakTime,startTimeIndx,stopTimeIndx,peakAmp= peakFreq[selmask],startFreq[selmask],stopFreq[selmask],peakTime[selmask],startTimeIndx[selmask],stopTimeIndx[selmask],peakAmp[selmask]
	
	
	#sort by startTimeIndx
	sortIndx=np.argsort(startTimeIndx)
	peakFreq,startFreq,stopFreq,peakTime,startTimeIndx,stopTimeIndx,peakAmp=peakFreq[sortIndx],startFreq[sortIndx],stopFreq[sortIndx], peakTime[sortIndx],startTimeIndx[sortIndx],stopTimeIndx[sortIndx],peakAmp[sortIndx]
	
	
	#detect harmonic and make frequency range selection
	isHarmonic=detectHarmonic(peakFreq,startFreq,stopFreq,peakTime,startTimeIndx/sfreq,stopTimeIndx/sfreq,peakAmp)	   
	selmask=np.logical_not(isHarmonic)
	selmask=np.logical_and(selmask,np.logical_and.reduce((startFreq>freqrange[0],stopFreq<freqrange[1])))
	peakFreq,startFreq,stopFreq,peakTime,startTimeIndx,stopTimeIndx,peakAmp= peakFreq[selmask],startFreq[selmask],stopFreq[selmask],peakTime[selmask],startTimeIndx[selmask],stopTimeIndx[selmask],peakAmp[selmask]
	print("Harmonics fraction: %.2f percent"%np.mean(isHarmonic))
	
	#remove duplicate bursts	
	isNotDuplicate=removeDuplicates(peakFreq,startFreq,stopFreq,peakTime,startTimeIndx/sfreq,stopTimeIndx/sfreq,peakAmp)
	print("Unique fraction: %.2f"%np.mean(isNotDuplicate))
	peakFreq,startFreq,stopFreq,peakTime,startTimeIndx,stopTimeIndx,peakAmp= peakFreq[isNotDuplicate],startFreq[isNotDuplicate],stopFreq[isNotDuplicate],peakTime[isNotDuplicate],startTimeIndx[isNotDuplicate],stopTimeIndx[isNotDuplicate],peakAmp[isNotDuplicate]
		
	
	#determine sleep score at timestamps of selected bursts	
	taxis,sleepScore=readSleepScoreFinal(pID)
	sleepScore=scipy.interpolate.interp1d(taxis,sleepScore,bounds_error=False,fill_value=-1,kind='nearest')(peakTime)	
	
	print("number of bursts detected:%d"%len(peakTime))
	data=np.savetxt(rootdir+"/burstFromMorlet/%s_%s_selected.txt"%(pID,ch_name),np.column_stack(( peakFreq,startFreq,stopFreq,np.round(peakTime*sfreq).astype("int"),startTimeIndx,stopTimeIndx,peakAmp,sleepScore)),fmt='%.2f\t%.2f\t%.2f\t%d\t%d\t%d\t%.3f\t%d',header='peakFreq\tstartFreq\tstopFreq\tpeakTimeIndx\tstartTimeIndx\tstopTimeIndx\tpeakAmp\tsleepScore')

