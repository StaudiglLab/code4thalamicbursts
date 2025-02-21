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
import time
from scipy.signal import find_peaks

from core import *
from core.helpers import *
from coreFunctions import getBurstRate2D

#get bootstrapped realizations of mean burst rate curve
@njit(parallel=True)
def getMeanRealizations(series,niter):
	nsample=series.shape[0]
	nchan=series.shape[1]
	meanRealizations=np.zeros((niter,nchan))
	for i in prange(niter):
		mean=np.zeros(nchan)
		for iSample in range(0,nsample):
			selection=np.random.randint(low=0,high=nsample)
			mean+=series[selection]
		meanRealizations[i]=mean
	return meanRealizations/nsample

#get bootstrap errors on mean burst rates in each frequency band
def getBootstrapError(series,
		peakFreqIndx,   #peak frequency index of bumps
		lowFreqIndx,	#corresponding start frequencies
		highFreqIndx,	#corresponding end frequencies
		niter		#number of bootstrap iterations
		):

	nsample=series.shape[1]
	
	true=np.mean(series,axis=1)
	t=time.time()
	meanRealizations=getMeanRealizations(np.swapaxes(series,0,1),niter)
	t=time.time()-t
	print("Time taken for bootstrap:%.2fsec"%t)
	
	pvaluesLeft=np.zeros(len(lowFreqIndx))
	pvaluesRight=np.zeros(len(lowFreqIndx))	
	pvaluesBoth=np.zeros(len(lowFreqIndx))
	pvaluesMean=np.zeros(len(lowFreqIndx))	
	
	for i in range(0,len(lowFreqIndx)):
		backg=(meanRealizations[:,lowFreqIndx[i]]+meanRealizations[:,highFreqIndx[i]])/2.
		peakRate=meanRealizations[:,peakFreqIndx[i]]
		
		pvaluesLeft[i]=np.mean(peakRate<=meanRealizations[:,lowFreqIndx[i]])
		
		pvaluesRight[i]=np.mean(peakRate<=meanRealizations[:,highFreqIndx[i]])		
		
		pvaluesBoth[i]=np.mean(np.logical_or(peakRate<=meanRealizations[:,lowFreqIndx[i]],
					peakRate<=meanRealizations[:,highFreqIndx[i]])) #this is the one finally used to get significant burst rates
					
		pvaluesMean[i]=np.mean(peakRate<=backg)
	return np.mean(series,axis=1),np.std(meanRealizations,axis=0),pvaluesLeft,pvaluesRight,pvaluesBoth,pvaluesMean


	
	
#function to merge bumps with low prominence
#note "prominence" is used without reference to technical meaning 
def mergePeaks(series,	#mean burst rate
		peaks,	#indices of peaks
		start,	#indices of corresponding start frequencies
		stop,	#indices of corresponding stop frequencies
		prominenceThresh #threshold for merging
		):
		
	prominence=np.zeros(len(peaks))
	for i in range(0,len(peaks)-1):
		seriesThis=series[peaks[i]:peaks[i+1]]
		#prominence is simply the maximum burst rate between the two peaks divided by the minimum. Higher prominence => clearer through
		prominence[i]=np.max(seriesThis)/np.min(seriesThis)
	
	selmask=np.ones(len(peaks),dtype=bool)
	for i in range(0,len(peaks)-1):
		j=i
		maxPeakPos=peaks[i]
		maxPeakVal=series[peaks[i]]
		#while loop ensures more than two peaks are appropriately merged, if conditions are met.
		while(prominence[j]<prominenceThresh and j<len(peaks)-1):
			selmask[j+1]=False
			if(series[peaks[j+1]]>maxPeakVal):
				maxPeakPos=peaks[j+1]
				maxPeakVal=series[maxPeakPos]
			stop[i]=stop[j+1]
			peaks[i]=maxPeakPos
			j=j+1
			
	peaks=peaks[selmask]
	start=start[selmask]
	stop=stop[selmask]
	
	return peaks,start,stop
	
def detectPeaks(freqs,
		burstRateMean,
		absThresh=0.1,			#minimum burst rate at peak 
		relThresh=0.25,			#relative height for determining width
		relativeChangeThresh=1.01,	#second criteria for determinining width
		minWidth=2,			#minimum width of bumps, in Hz
		prominenceThreshForMerging=1.2, #nearby peaks with prominence less than this are merged into one
		):
	

	peaks,t=find_peaks(burstRateMean,height=absThresh)
	
	
	start=np.zeros(len(peaks),dtype=int)
	stop=np.zeros(len(peaks),dtype=int)
	
	#computing slope of burst rate curve; this is used for initial estimation of start frequency and end frequency
	slope=np.ediff1d(burstRateMean,to_end=0.0)
	peaksInSlopePos,t=find_peaks(slope)
	peaksInSlopeNeg,t=find_peaks(-slope)	

	for i in range(0,len(peaks)):
		indx=peaks[i]
		peakValue=burstRateMean[indx]
		start[i]=peakLeft=indx-1
		peakRight=stop[i]=indx+1
		
		#initial start and end estimate obtained from nearest peak in first derivative,if that exists
				
		if(np.sum(peaksInSlopePos<indx)==0):
			peakLeft=start[i]=0
		else:
			peakLeft=start[i]=np.max(peaksInSlopePos[peaksInSlopePos<=indx])	
				
		if(np.sum(peaksInSlopeNeg>indx)==0):
			peakRight=stop[i]=len(slope)-1
		else:
			peakRight=stop[i]=np.min(peaksInSlopeNeg[peaksInSlopeNeg>=indx])
			
		
		#find points on either side where burst rate doesn't change as fast.
		for j in range(peakLeft,1,-1):
			if((burstRateMean[j-2]*relativeChangeThresh<burstRateMean[j])):
				start[i]-=1
			else:
				break		
			
		for j in range(peakRight,len(freqs)-2):
			if((burstRateMean[j+2]*relativeChangeThresh<burstRateMean[j])):
				stop[i]+=1
			else:
				break
		
		#adjust stard and end points if the points determined above have
		#burst rates lower than the relative threshold
		
		while(burstRateMean[start[i]]/peakValue<relThresh):
			start[i]+=1
			
		while(burstRateMean[stop[i]]/peakValue<relThresh):
			stop[i]-=1


	#merge bumps without prominent through
	peaks,start,stop=mergePeaks(burstRateMean,peaks,start,stop,prominenceThresh=prominenceThreshForMerging)
	
	#apply width criteria
	selmask=np.logical_and(stop-peaks>=minWidth,peaks-start>=minWidth)
	
	start=start[selmask]
	stop=stop[selmask]
	peaks=peaks[selmask]
	return peaks,start,stop


#detect oscillations in channel along with producing diagnostic plots

def detectOscillation(pID,
			ch_name,
			sfreq=200.0,
			smoothWindowInHz=2, #smooth burst rate curve by gaussian kernal of FWHM smoothWindowInHz
			niter=int(1e6)	#number of iterations to determine p-value
			):
	#get 2D burst rate for the channel
	taxis,freqs,burstRate,sleepScore=getBurstRate2D(pID,ch_name,sfreq=sfreq,smoothWindowInHz=smoothWindowInHz)

	#prepare matplotlib figure for diagnostic plot
	
	fig, axes = plt.subplot_mosaic("AAAABB",figsize=(12,4))
	fig.suptitle("%s %s"%(pID,ch_name))
	fig.subplots_adjust(wspace=1)
	axes['A'].imshow(burstRate,origin='lower',extent=(taxis[0]/3600.,(taxis[-1]+30.)/3600.,freqs[0],freqs[-1]),aspect='auto',vmax=5)
	ax2=axes['A'].twinx()
	plotStandardHypnogram(taxis,sleepScore,ax2,c='gray')
	axes['A'].set_xlabel("Time (hr)")
	axes['A'].set_ylabel("Frequency (Hz)")	 
	ax2=axes['B']

	freqlist={}
	
	#iterate over three states
	for istate,state in zip(np.arange(3),['wake','REM','NREM']):
		if(state=='wake'):
			sleepmask=sleepScore==0  
		elif(state=='REM'):
			sleepmask=sleepScore==5
		elif(state=='NREM'):
			sleepmask=np.logical_or(sleepScore==2,sleepScore==3)	  

		#get all peaks in the mean burst rate curve
		peakFreqIndx,lowFreqIndx,highFreqIndx=detectPeaks(freqs=freqs,burstRateMean=np.mean(burstRate[:,sleepmask],axis=1))
		
		#bootstrap and get pvalues
		mean,std,pvaluesLeft,pvaluesRight,pvaluesBoth,pvaluesMean= getBootstrapError(burstRate[:,sleepmask],peakFreqIndx,lowFreqIndx,highFreqIndx,niter=niter)
		
		#convert indices to actual frequencies
		peakFreq,lowFreq,highFreq=freqs[peakFreqIndx],freqs[lowFreqIndx],freqs[highFreqIndx]
		
		freqlist[state]=np.column_stack((peakFreq,lowFreq,highFreq,pvaluesLeft,pvaluesRight,pvaluesBoth,pvaluesMean))
		
		#add to diagnostic plot
		for i in range(0,len(peakFreq)):
			if(pvaluesBoth[i]<1e-3): #only for plotting diagnostics, pvalue threshold for final use elsewhere.
				ax2.plot([lowFreq[i],highFreq[i]],[-0.2-.1*istate,-0.2-0.1*istate],c='C%d'%istate,marker='|')
		ax2.plot(freqs,mean,label=state,c='C%d'%istate)
		ax2.grid()
		ax2.fill_between(freqs,mean-std,mean+std,fc='C%d'%istate,alpha=0.5)
	ax2.legend()
	ax2.set_xlabel("Frequency (Hz)")
	ax2.set_ylabel("Burst rate (/min)")

	return fig,freqlist
	


	
	
