import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from waveletTransform import compute_wavelet_transform
from core import *
from core.helpers import *

	
#function to compute wavelet transform, perform temporal smoothing and write to disk
def writeAverageMorletToDisk(pID, 
	electrode,
	chunkSizeInSec=1, #temporal smoothing before writing to disk
	n_cycles_morlet=[5,10,15],
	fmin=5, 
	fmax=48.0,
	delf=0.5 #spacing along frequency in Hz
	):
		

	raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,electrode)).pick("eeg")
	
	taxis=raw.times
	
	#size to smooth over in number of samples
	chunkSize=int(chunkSizeInSec*raw.info['sfreq'])
	
	nSampleTotalAfterAvg=len(taxis)//chunkSize
	
	freqs=np.arange(fmin,fmax,0.5,dtype=np.float64)
	
	
	nMid=len(freqs)//2 
	
	#save frequency axis
	np.save(rootdir+"/morlet/%s_electrode-%s_morlet_freqs.npy"%(pID,electrode),freqs)		
	for i in range(0,len(raw.ch_names)):
		print("Running morlet transform for %s, channel %s"%(pID,raw.ch_names[i]))
		data=raw.get_data(picks=raw.ch_names[i])[0]
		for n_cycle in n_cycles_morlet:
			#computing morlet transform in two parts, splitting the frequency axis in half, just for saving memory
			
			#first part
			mwt = compute_wavelet_transform(data, fs=np.float64(raw.info['sfreq']), n_cycles=np.int32(n_cycle), freqs=freqs[:nMid])
			
			#average first part along time dimension		
			mwt=mwt[:,:chunkSize*nSampleTotalAfterAvg].reshape(len(freqs[:nMid]),nSampleTotalAfterAvg,chunkSize)
			mwtAvg=np.nanmean(np.abs(mwt),axis=2)	
			del mwt	
			
			#second part	
			mwt = compute_wavelet_transform(data, fs=np.float64(raw.info['sfreq']), n_cycles=np.int32(n_cycle), freqs=freqs[nMid:])
			
			#average and append second part
			mwt=mwt[:,:chunkSize*nSampleTotalAfterAvg].reshape(len(freqs[nMid:]),nSampleTotalAfterAvg,chunkSize)			
			mwtAvg=np.append(mwtAvg,np.nanmean(np.abs(mwt),axis=2),axis=0)	
			np.save(rootdir+"/morlet/%s_%s_morlet_n_cycle_%d_1s.npy"%(pID,raw.ch_names[i],n_cycle),mwtAvg)
			del mwt
	




		
#loop over all subjects and write morlet transform
for pID in cohortForPaper:	
	for el in ['L','R']:
		writeAverageMorletToDisk(pID,el)	

