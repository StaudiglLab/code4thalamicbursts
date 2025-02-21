import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special as sc
sc.seterr(all='ignore')

from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure
from numba import njit
from joblib import Parallel, delayed

from core import *
from core.helpers import *
from psd.waveletTransform import compute_wavelet_transform



#morlet transform data
def getMorlet(pID,ch_name,freqs,n_cycle):
        if(ch_name[0]=='L' or ch_name[0]=='R'):
        	#for thalamic electrodes
                electrode=ch_name[0]       
                raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,electrode)).pick("eeg")
                ch_names=np.array(raw.ch_names)
                raw.pick(ch_name).load_data()
                data=raw.get_data()[0]
                badMask=np.load(rootdir+"/IEDMask/%s_electrode_%s_IEDmask_convolved.npy"%(pID,electrode))[ch_names==ch_name][0]
                taxis=raw.times
                sfreq=raw.info['sfreq']
        else:
        	#for scalp data
                taxis,data,badMask,sfreq=getScalpData(pID,ch_name,ref='Cpz')
                
	#setting masked datapoints to NaN 
        data[badMask]=np.nan
        mwt = compute_wavelet_transform(data, fs=sfreq, n_cycles=n_cycle, freqs=freqs)	
        return taxis.copy(),freqs,mwt

#function to find temporal and spectral width around each peak that has been detected
@njit
def findWidthAroundPeaks(data, #full 2D morlet amplitudes
			peakPos, #indices of peaks
			peakAmp, #amplitudes of peaks
			thresholdAbs, #absolute threshold
			thresholdRel #threshold relative to peak
			):
        startFreqIndx=np.zeros(len(peakPos))
        stopFreqIndx=np.zeros(len(peakPos))
        startTimeIndx=np.zeros(len(peakPos))
        stopTimeIndx=np.zeros(len(peakPos))
        nfreqs=data.shape[0]
        ntimes=data.shape[1]
        
        for i in range(0,len(peakPos)):
                peakFreq=peakPos[i,0]
                peakTime=peakPos[i,1]
        	#choose threshold: either peak*thresholdRel or thresholdAbs which ever is larger
        	
                thresholdThis=max(peakAmp[i]*thresholdRel,thresholdAbs)
                
                #search for start frequency
                freqStart=peakFreq
                while(freqStart>0):
                        if(data[freqStart,peakTime]>thresholdThis        			 #continue if threshold is not crossed
                        		and data[freqStart-1,peakTime]<data[freqStart,peakTime]  #continue if amplitude is still decreasing
                        		):
                                freqStart-=1
                        else:
                                break
                                
                #search for stop frequency                                
                freqStop=peakFreq             
                while(freqStop<nfreqs-1):
                        if(data[freqStop,peakTime]>thresholdThis 					 #continue if threshold is not crossed
                        		and data[freqStop+1,peakTime]<data[freqStop,peakTime]):		 #continue if amplitude is still decreasing
                                freqStop+=1
                        else:
                                break
                                
                #search for start time                                                        
                timeStart=peakTime        
                while(timeStart>0):
                        if(data[peakFreq,timeStart]>thresholdThis 					#continue if threshold is not crossed
                        		and data[peakFreq,timeStart-1]<data[peakFreq,timeStart]):	#continue if amplitude is still decreasing
                                timeStart-=1
                        else:
                                break
                                
                                
                #search for start time                                                                        
                timeStop=peakTime             
                while(timeStop<ntimes-1):
                        if(data[peakFreq,timeStop]>thresholdThis 					#continue if threshold is not crossed
                        		and data[peakFreq,timeStop+1]<data[peakFreq,timeStop]):		#continue if amplitude is still decreasing
                                timeStop+=1
                        else:
                                break
                                
                #save indices to array
                startFreqIndx[i]=freqStart
                stopFreqIndx[i]=freqStop
                startTimeIndx[i]=timeStart
                stopTimeIndx[i]=timeStop
        return startFreqIndx,stopFreqIndx,startTimeIndx,stopTimeIndx
        
        
def getMorletBurstsFromPeaks(pID,
			ch_name,
			fmin=3, 	#start frequency
			fmax=49,	#stop frequency
			delf=0.5,	#frequency steps
			threshold=2,	#threshold for peaks (x median)
			thresholdRel=0.1, #threshold relative to peak, for determining width
			n_cycle_mwt=10, #number of morlet cycles
			saveToFile=True):

	freqs=np.arange(fmin,fmax,delf)	
	
	#compute morlet amplitude
	taxis,freqs,mwt=getMorlet(pID=pID,ch_name=ch_name,freqs=freqs,n_cycle=n_cycle_mwt)	
	mwt=np.abs(mwt)
	
	#normalize by median
	mwt/=np.nanmedian(np.abs(mwt),keepdims=True)
	
	
	
	#find peaks in 2D morlet that exceed threshold
	neighborhood = generate_binary_structure(2,2)
	local_max = maximum_filter(mwt, footprint=neighborhood)==mwt
	local_max[local_max]=mwt[local_max]>threshold
	freqs_2d,taxis_2d=np.meshgrid(np.arange(len(freqs)),np.arange(len(taxis)),indexing='ij')
	peakFreq,peakTime=freqs_2d[local_max],taxis_2d[local_max]
	
	#exclude peaks at edge frequencies
	
	selmask=np.logical_and(peakFreq>1,peakFreq<len(freqs)-1)	
	peakAmp=mwt[local_max][selmask]
	peakFreq=peakFreq[selmask]
	peakTime=peakTime[selmask]
	
	#get burst boundaries
	startFreqIndx,stopFreqIndx,startTimeIndx,stopTimeIndx=findWidthAroundPeaks(mwt,np.column_stack((peakFreq,peakTime)),peakAmp
								,thresholdAbs=threshold,thresholdRel=thresholdRel)
	
	#convert indices to actual frequencies and actual times
	peakFreq,peakTime=freqs[peakFreq],taxis[peakTime]	
	startFreq,stopFreq=freqs[startFreqIndx.astype("int")],freqs[stopFreqIndx.astype("int")]
	
	
	if(saveToFile):
		np.save(rootdir+"/burstFromMorlet/%s_%s.npy"%(pID,ch_name),np.column_stack((peakFreq,startFreq,stopFreq,peakTime,startTimeIndx,stopTimeIndx,peakAmp)))
			
	return np.column_stack((peakFreq,startFreq,stopFreq,peakTime,startTimeIndx,stopTimeIndx,peakAmp))

	
	

