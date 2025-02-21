import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from core import *
from core.helpers import *


import warnings
warnings.filterwarnings("ignore")

	
def getPSD(pID,
	electrode,
	fmin,
	fmax,
	resamplingRate=200.0, #resample first
	n_fft=500, #number of samples over which to compute fft in Welch method.
	):	
	
	raw=mne.io.read_raw('C:/Aditya/thalamus-census/'+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,electrode),preload=True)	
	
	if(electrode=='scalp'): #for scalp data ensure reference is Cpz
		if(pID=='p26' or pID=='p20'): #patients with Fp1 bad
			ch_names=['F7','F8','T7','P7','T8','P8','O1','O2']
		else:
			ch_names=['Fp1','Fp2','F7','F8','T7','P7','T8','P8','O1','O2']  
		raw.pick(np.append(ch_names,['Cpz']))
		raw,t=mne.set_eeg_reference(raw, ref_channels=['Cpz'],copy=False) #rereferencing to Cpz
		raw.pick(ch_names)
		
		
	raw.info['subject_info']=None
	taxis=raw.times
	
	
	raw.pick('eeg')
	
	raw.resample(resamplingRate)	
	
	chunkFFTDuration=n_fft*(1.0/raw.info['sfreq'])
	print("Computing PSD on %.3f s blocks"%chunkFFTDuration)	
	
	#split entire data into chunks of duration chunkFFTDuration	 
	epochs=mne.make_fixed_length_epochs(raw, duration=chunkFFTDuration) 
	ch_names=np.array(raw.ch_names)

	#compute psd on each epoch of length chunkFFTDuration
	psd=epochs.compute_psd(fmin=fmin,fmax=fmax,verbose=False,n_jobs=-1,method='welch',n_fft=n_fft)	

	return psd
	

def savePSD(subjects,electrodes,fmin=1,fmax=90):	
	for pID in subjects:
		for iEl in range(len(electrodes)):
			epochsSpec=getPSD(pID,electrodes[iEl],fmin=fmin,fmax=fmax)	
			epochsSpec.save(rootdir+"/psd/%s_electrode-%s_psd.h5"%(pID,electrodes[iEl]),overwrite=True)

#loop over all subjects and compute PSD			
savePSD(subjects=['p26','pthal102'],electrodes=['L','R','scalp'])	


