import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import numpy as np
import mne
import pandas as pd


from core import *
from core.helpers import *

#function to get mask for bad data.
def getIEDMask(pID,
		electrode,
		thresh=[5,20] #first threshold for identification of IEDs
				    #second threshold for identification of bad epochs of data
		):
        #load data
        raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,electrode)).pick("eeg")
        raw.load_data()
        
        
        if(electrode=='scalp'): #for scalp data ensure reference is Cpz
                
                if(pID=='p26' or pID=='p20'): #patients with Fp1 bad
                        ch_names=['F7','F8','T7','P7','T8','P8','O1','O2']
                else:
                        ch_names=['Fp1','Fp2','F7','F8','T7','P7','T8','P8','O1','O2']
                        
                raw.pick(np.append(ch_names,['Cpz']))
                raw,t=mne.set_eeg_reference(raw, ref_channels=['Cpz'],copy=False) #rereferencing to Cpz
                raw.pick(ch_names)
                
        #calculate envelopes of filtered and unfiltered data     
        taxis=raw.times
        dhilbertUnfilt=raw.copy().apply_hilbert(envelope=True).get_data() #envelope of unfiltered data        		
        dhilbert=raw.filter(l_freq=55.0,h_freq=None,n_jobs=-1).apply_hilbert(envelope=True).get_data() #envelope of data filtered above 55 Hz
        
        #dividing by median
        dhilbert/=np.median(np.abs(dhilbert),axis=1,keepdims=True)   # axis 0 is channel and axis 1 is time; median over time
        dhilbertUnfilt/=np.median(np.abs(dhilbertUnfilt),axis=1,keepdims=True)
        
        #mark data as bad if either of the conditions are met
        IEDMask=np.logical_or(dhilbert>thresh[0],dhilbertUnfilt>thresh[1]) 
      
        return IEDMask,raw.info['sfreq']
	
def writeIEDMask(pID,electrode,
	thresh=[5,20], #first threshold for identification of IEDs
			#second threshold for identification of bad epochs of data
	expandMaskInSec=1 #amount by which to dilate mask around each bad point
	):

	IEDMask,sfreq=getIEDMask(pID,electrode,thresh=thresh)	
	np.save(rootdir+"/IEDMask/%s_electrode_%s_IEDmask.npy"%(pID,electrode),IEDMask) #save mask without any dilation

	#dilate mask such that samples within +/- expandMaskInSec are also masked out	
	expandMaskSamples=int(expandMaskInSec*sfreq)
	for i in range(0,len(IEDMask)):
		IEDMask[i]=np.convolve(IEDMask[i],np.ones(expandMaskSamples),mode='same')>0 
			
	np.save(rootdir+"/IEDMask/%s_electrode_%s_IEDmask_convolved.npy"%(pID,electrode),IEDMask) #save mask with dilation


#looping over all subjects			
for subID in cohortForPaper:				
	writeIEDMask(subID,'scalp',thresh=[5,20])
	#writeIEDMask(subID,'R',thresh=[5,20])
	#writeIEDMask(subID,'L',thresh=[5,20])
	

