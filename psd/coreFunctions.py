import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt

from core import *
from core.helpers import *

	
#function to read psd from disk, average over chunkSizeInMin and return averaged psd
def getPSD(pID,
	electrode,
	outputSamplingInterval=60.0,
	flim=None,
	returnSleepScore=True
	):
	print("Reading: %s electrode-%s"%(pID,electrode))
	power=mne.time_frequency.read_spectrum(rootdir+"/psd/%s_electrode-%s_psd.h5"%(pID,electrode))	
	epochLengthInSamples=(power.events[1,0]-power.events[0,0])
	epochLengthInSec=(epochLengthInSamples/power.info['sfreq'])
	print("Length of epochs: %.2f sec"%epochLengthInSec)

	
	
	#mark epochs with bad data	
	badMaskEpochs=getBADMask(pID,electrode,outsfreq=1.0/epochLengthInSec,combineChannels=False)
	badMaskEpochs=np.swapaxes(badMaskEpochs,0,1)
	powerDataFull=power._data
	
	print("Bad data fraction:%.3f"%np.nanmean(badMaskEpochs))
	
	#trimming arrays to ensure equal length
	if(len(badMaskEpochs)<len(powerDataFull)):
		powerDataFull=powerDataFull[:len(badMaskEpochs)]
	else:
		badMaskEpochs=badMaskEpochs[:len(powerDataFull)]	
	powerDataFull[badMaskEpochs]=np.nan
	'''
	badMask=getBADMask(pID,electrode,combineChannels=False)
	badMask=np.swapaxes(badMask,0,1)
	powerDataFull=power._data
	#badMask=getCommonBadMask(pID)
	nFFT=500
	badMask=np.mean(badMask[:(badMask.shape[0]//nFFT)*nFFT].reshape((len(badMask)//nFFT,nFFT,badMask.shape[1])),axis=1)>0
	print(pID,"bad data fraction",np.nanmean(badMask))
	if(len(badMask)<len(powerDataFull)):
		powerDataFull=powerDataFull[:len(badMask)]
	else:
		badMask=badMask[:len(powerDataFull)]	
	powerDataFull[badMask]=np.nan
	'''
	
	
	#averaging epochs
	nAve=int(np.round(outputSamplingInterval/epochLengthInSec))	
	nChunksNew=len(powerDataFull)//nAve
	psdDataFull=np.nanmean(powerDataFull[:nChunksNew*nAve].reshape((nChunksNew,nAve,len(power.ch_names),len(power.freqs))),axis=1)
	if(not flim is None):
		freqMask=np.logical_and(power.freqs>flim[0],power.freqs<flim[1])
	else:
		freqMask=np.ones(len(power.freqs),dtype=bool)

	taxis=np.arange(0,nChunksNew)*outputSamplingInterval
	
	if(returnSleepScore):
		taxisSS,sleepScore=readSleepScoreFinal(pID)
		#interpolating sleepscore to get sleepscore at psd epochs
		sleepScore = scipy.interpolate.interp1d(taxisSS,sleepScore, kind='nearest',fill_value=-1,bounds_error=False)(taxis)
		return taxis,power.freqs[freqMask],psdDataFull[:,:,freqMask],np.array(power.ch_names),sleepScore
	else:
		return taxis,power.freqs[freqMask],psdDataFull[:,:,freqMask],np.array(power.ch_names),power.info
	
#read Morlet from file and average
def readMorlet(pID,
		ch_name,
		inputSamplingInterval=1,
		outputSamplingInterval=60,
		n_cycle=10
		):

	mwt=np.load(rootdir+"/morlet/%s_%s_morlet_n_cycle_%d_%ds.npy"%(pID,ch_name,n_cycle,inputSamplingInterval))
	freqs=np.load(rootdir+"/morlet/%s_electrode-%s_morlet_freqs.npy"%(pID,ch_name[0]))	
	
	integrateFactor=int(outputSamplingInterval//inputSamplingInterval)
	
	#mark epochs with bad data		
	badMask=getBADMask(pID,
			ch_name[0],
			outsfreq=1/inputSamplingInterval,
			combineChannels=False)[int(ch_name[1])-1]
	mwt[:,badMask>0]=np.nan
			
	#average morlet to outputsampling interval
	nSampNew=int(mwt.shape[1]//integrateFactor)
	mwt=np.nanmean(mwt[:,:nSampNew*integrateFactor].reshape((mwt.shape[0],nSampNew,integrateFactor)),axis=2)	
	
	
	return freqs,np.abs(mwt)


	
def getMorletPSD(pID,
		electrode,
		outputSamplingInterval=60,
		n_cycle_morlet=10,
		flim=[8,48],
		returnSleepScore=True):
		
	#call readMorlet function for each channel and combine them together into one array
	ch_names=['%s1-%s2'%(electrode,electrode),'%s2-%s3'%(electrode,electrode),'%s3-%s4'%(electrode,electrode)]
	mwtAll=None
	for ch_name in ch_names:
		freqs,mwt=readMorlet(pID,ch_name,outputSamplingInterval=outputSamplingInterval,n_cycle=n_cycle_morlet)
		freqmask=np.logical_and(freqs>=flim[0],freqs<=flim[1])
		freqs=freqs[freqmask]
		if(mwtAll is None):
			mwtAll=np.expand_dims(mwt[freqmask].T,axis=1)
		else:
			mwtAll=np.append(mwtAll,np.expand_dims(mwt[freqmask].T,axis=1),axis=1)
	#interpolate sleepscore to outputSamplingInterval		
	taxisFull=np.arange(mwtAll.shape[0])*outputSamplingInterval	
	taxis,sleepScore=readSleepScoreFinal(pID)
	sleepScoreFull=scipy.interpolate.interp1d(taxis,sleepScore,bounds_error=False, fill_value=-1,kind='nearest')(taxisFull)			
	return taxisFull,freqs,mwtAll,ch_names,sleepScoreFull



