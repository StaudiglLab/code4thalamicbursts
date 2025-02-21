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
from scipy.signal import find_peaks


from core import *
from core.helpers import *

import pandas as pd
from joblib import Parallel, delayed
from numba import prange


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


	
#get errors on mean burst rate
def getBootstrapErrorOnCurve(series, #2D burst rate
			niter=10000, #number of iterations for bootstrapping
			pvalue=0.01  #quantiles are based on this
			):
	nsample=series.shape[1]
	meanRealizations=getMeanRealizations(np.swapaxes(series,0,1),niter)
	q=np.quantile(meanRealizations,q=[pvalue/2.,1-pvalue/2.],axis=0)
	return np.mean(series,axis=1),np.std(meanRealizations,axis=0),q[0],q[1]	

#check for overlapping bands and for overlapping bands find which band (narrowest one)
def getOverlaps(df_detections):
	nRows=len(df_detections)
	
	hasOverlap={'REM':np.zeros(nRows,dtype=bool),'wake':np.zeros(nRows,dtype=bool),'NREM':np.zeros(nRows,dtype=bool)}
	
	uniqueSelectionMask=np.ones(len(df_detections),dtype=bool)
	#iterating over all bands
	for i in range(0,nRows):
		indx_i=df_detections.index[i]	
		low1,high1,peak1=df_detections.loc[indx_i,'freqLow'],df_detections.loc[indx_i,'freqHigh'],df_detections.loc[indx_i,'freqPeak']			
		for state in ['REM','wake','NREM']:
			#same pID, channel name
			selmask=np.logical_and(df_detections['pID']==df_detections['pID'][indx_i],df_detections['ch_name']==df_detections['ch_name'][indx_i])
			#brain state to look for in overlap
			selmask=np.logical_and(selmask,df_detections['state']==state)			
			if(np.sum(selmask)==0):
				continue
			#iterate over bands for same pID, channel, and specific brain state
			df_selected=df_detections[selmask]
			for j in range(0,len(df_selected)):
				indx=df_selected.index[j]
				low2,high2,peak2= df_selected.loc[indx,'freqLow'],df_selected.loc[indx,'freqHigh'],df_selected.loc[indx,'freqPeak']
				#overlap is when peak of one falls within the extent of another band
				if((low1<=peak2 and peak2<=high1) or (low2<=peak1 and peak1<=high2)):
					#found overlap
					hasOverlap[state][i]=True
					#select narrower band or for equal widths, any one of them (doesn't matter)
					if(high2-low2<high1-low1):
						uniqueSelectionMask[i]=False
					elif(high2-low2==high1-low1 and indx_i<indx):
						uniqueSelectionMask[i]=False
					break
	#add overlap information to dataframe
	df_detections['hasOverlap_REM'] =	hasOverlap['REM']
	df_detections['hasOverlap_NREM']=	hasOverlap['NREM']
	df_detections['hasOverlap_wake']=	hasOverlap['wake']
	df_detections['uniqueSelection']=	uniqueSelectionMask
	return df_detections
				
#return bursts in given band
def getSelectedBursts(pID,
			ch_name,
			freqLowBand,
			freqHighBand,
			minWidthInCycles=5,
			samplingFreq=200.0):	
	#load burst information
	peakFreq,startFreq,stopFreq,peakTimeIndx,startTimeIndx,stopTimeIndx,peakAmp,sleepScoreAtBurst=np.loadtxt(rootdir+"/burstFromMorlet/%s_%s_selected.txt"%(pID,ch_name),unpack=True)
	#impose width and frequency selection
	width=(stopTimeIndx-startTimeIndx)/samplingFreq
	widthInCycles=peakFreq*width
	selmask=np.logical_and(np.logical_and(peakFreq<=freqHighBand,peakFreq>=freqLowBand),widthInCycles>=minWidthInCycles)
	return peakFreq[selmask],peakTimeIndx[selmask].astype("int"),startTimeIndx[selmask].astype("int"),stopTimeIndx[selmask].astype("int"),width[selmask],sleepScoreAtBurst[selmask].astype("int")


	

		
		
#function to determine 2D burst rate
def getBurstRate2D(pID,ch_name,smoothWindowInHz,sfreq=200.0):
	taxis,sleepScore=readSleepScoreFinal(pID)
	#load burst positions	
	peakFreq,startFreq,stopFreq,peakTimeIndx,startTimeIndx,stopTimeIndx,peakAmp,sleepScoreAtBurst=np.loadtxt(rootdir+"/burstFromMorlet/%s_%s_selected.txt"%(pID,ch_name),unpack=True)
	
	freqaxis=np.arange(8.25,45,0.5)
	
	#bin every 30seconds to get burst 2D rate
	burstRate,xedge,freqs=np.histogram2d(peakTimeIndx/sfreq,peakFreq,bins=(np.append(taxis,[taxis[-1]+30]),freqaxis))
	burstRate=burstRate.T*2 #per min
	freqs=(freqs[1:]+freqs[:-1])/2.	
	
	#smooth burst rate by gaussian kernal
	if(not smoothWindowInHz is None):
		sigma=(smoothWindowInHz/2.355)/(freqaxis[1]-freqaxis[0])
		burstRate=scipy.ndimage.gaussian_filter(burstRate,(sigma,0),mode='nearest',truncate=5.0)

	return taxis,freqs,burstRate,sleepScore


#function to get burst density in bands
def getMeanBurstDensity(pID,
		ch_name,
		states, #state in which burst was detected
		freqLow, # lower edge of band
		freqHigh, #higher edge of band
		sfreq=200.0
		):
		
	peakBurstRateInBand=np.zeros(len(states))
	burstRateInBand=np.zeros(len(states))	
	burstRateStates={'wake':np.zeros(len(states)),'REM':np.zeros(len(states)),'NREM':np.zeros(len(states))}
	
	#get 2D burst rate (without spectral smoothing to make sure one is only counting bursts within the band)
	taxis,freqs,burstRate,sleepScore=getBurstRate2D(pID,ch_name,sfreq=sfreq,smoothWindowInHz=None)
	
	#get mean burst rate curve in each brain state
	burstRateMean={'wake':np.mean(burstRate[:,sleepScore==0],axis=1),
			'REM':np.mean(burstRate[:,sleepScore==5],axis=1),
			'NREM':np.mean(burstRate[:,np.logical_or(sleepScore==2,sleepScore==3)],axis=1)}
	
	#get burst densities
	for i in range(0,len(peakBurstRateInBand)):				
		for state in ['wake','REM','NREM']:
			rateThis=burstRateMean[state][np.logical_and(freqs>=freqLow[i],freqs<=freqHigh[i])]
			burstRateStates[state][i]=np.sum(rateThis)
			if(state==states[i]):
				peakBurstRateInBand[i]=np.max(rateThis)
				burstRateInBand[i]=np.sum(rateThis)
				
	return 	peakBurstRateInBand,burstRateInBand,burstRateStates    
	


#get dataframe containing frequency bands with significant burst rates
def getSignificantBands(which='all', #selection
			infile=repo.working_tree_dir+"/burst/outfiles/bursts_detectedFrequencies_selected.txt"	
		):
	df_freqs=pd.read_csv(infile,sep=' ')	
	if(which=='all'):
		df_freqs=getOverlaps(df_freqs)
		return df_freqs
	elif(which=='allUnique'):
		df_freqs=getOverlaps(df_freqs)	
		selmask=df_freqs['uniqueSelection']==True
		df_freqs=df_freqs[selmask]
		return df_freqs
	elif(which=='gamma'):
		df_freqs=df_freqs[np.logical_and(df_freqs['freqPeak']>=19,np.logical_or(df_freqs['state']=='REM',df_freqs['state']=='wake'))]	
		df_detections=getOverlaps(df_freqs)		
		selmask1=df_freqs['uniqueSelection']==True
		#both REM and wake
		selmask2=np.logical_or(np.logical_and(df_freqs['state']=='REM', df_freqs['hasOverlap_wake']==True),
					np.logical_and(df_freqs['state']=='wake',df_freqs['hasOverlap_REM']==True))
		#higher burst rate in REM and wake
		selmask3=np.logical_and(df_freqs['meanBurstRate_wake']>df_freqs['meanBurstRate_NREM'],
					df_freqs['meanBurstRate_REM']>df_freqs['meanBurstRate_NREM'])
		df_freqs=df_freqs[np.logical_and.reduce((selmask1,selmask2,selmask3))]
                #select higher frequency band
		df_freqs=getSingleFreqBandPerContact(df_freqs)
		return df_freqs
	elif(which=='gammaREMCorrelated'):
		df_freqs=pd.read_csv(repo.working_tree_dir+"/burstAndREMs/outfiles/REM_burstSaccadeCrossCorr.txt",sep=' ')	
		df_freqs.set_index('Unnamed: 0',inplace=True)
		df_freqs=df_freqs[df_freqs['crossCorrCoeff_pvalue']<=9e-4]
		return df_freqs
	elif(which=='spindle'):
		#select bands in spindle frequency range
		df_freqs=df_freqs[np.logical_and.reduce((df_freqs['freqPeak']>=11,df_freqs['freqPeak']<=17,df_freqs['state']=='NREM'))]
		return df_freqs
	elif(which=='spindleInGammaChannels'):	
		df_freqs=getSignificantBands(which='spindle')
		
		#select pIDs and channels where gamma in REM/wake is also detected
		df_freqs_gamma=getSignificantBands(which='gamma')
		pIDGamma=df_freqs_gamma['pID'].values
		chnameGamma=df_freqs_gamma['ch_name'].values
		selmaskCommon=np.zeros(len(df_freqs),dtype=bool)
		for i in range(0,len(df_freqs)):
			indx=df_freqs.index[i]
			if(np.sum(np.logical_and(df_freqs.loc[indx,'pID']==pIDGamma,df_freqs.loc[indx,'ch_name']==chnameGamma))):
				selmaskCommon[i]=True
				
		return df_freqs[selmaskCommon]
	elif(which=='spindleInGammaREMChannels'):	
		df_freqs=getSignificantBands(which='spindle')
		
		#select pIDs and channels where gamma in REM/wake is also detected
		df_freqs_gamma=getSignificantBands(which='gammaREMCorrelated')
		pIDGamma=df_freqs_gamma['pID'].values
		chnameGamma=df_freqs_gamma['ch_name'].values
		selmaskCommon=np.zeros(len(df_freqs),dtype=bool)
		for i in range(0,len(df_freqs)):
			indx=df_freqs.index[i]
			if(np.sum(np.logical_and(df_freqs.loc[indx,'pID']==pIDGamma,df_freqs.loc[indx,'ch_name']==chnameGamma))):
				selmaskCommon[i]=True
				
		return df_freqs[selmaskCommon]

#select a single frequency band per contact when more than one exists
def getSingleFreqBandPerContact(df_detections,
				func=np.argmax #which band to select; replace with np.argmin for lower one
				):
	pID_ch=df_detections['pID'].values+df_detections['ch_name'].values
	freqPeak=df_detections['freqPeak'].values
	selmask=np.ones(len(pID_ch),dtype=bool)
	for i in range(0,len(pID_ch)):
		sameMask=pID_ch==pID_ch[i]
		if(np.sum(sameMask)>1):
			freqs=freqPeak[sameMask]
			selmask[sameMask]=np.arange(len(freqs))==func(freqs)
	df_detections=df_detections[selmask]
	return df_detections
