import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git
repo = git.Repo('.', search_parent_directories=True)
import sys
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from numba import njit, prange

from core import *
from core.helpers import *



from psd.waveletTransform import compute_wavelet_transform


#function to load scalp data
def getScalpData(pID,ch_name_select,ref='Cpz'):
        
        #select consistently available channels
        if(pID=='p26' or pID=='p20'):
                ch_names_allsel=['F7','F8','T7','P7','T8','P8','O1','O2']
        else:
                ch_names_allsel=['Fp1','Fp2','F7','F8','T7','P7','T8','P8','O1','O2']
       
       	#load data and rereference         
        raw=mne.io.read_raw('C:/Aditya/thalamus-census'+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,'scalp')).pick("eeg")
        if(ref!='average'):
                ch_names_allsel=np.append(np.array(ch_names_allsel),[ref])
                ref=[ref]
                
        raw.pick(ch_names_allsel).load_data()
        mne.set_eeg_reference(raw, ref_channels=ref,copy=False)
        raw.pick(ch_name_select)
        
        #mask data only if more than half of the scalp contacts have artefacts
        badMask=np.load(rootdir+"/IEDMask/%s_electrode_scalp_IEDmask_convolved.npy"%(pID))
        badMask=np.mean(badMask,axis=0)>0.5
        d=raw.get_data()[0]
        
        return raw.times,d,badMask,raw.info['sfreq']

#perform morlet transform of entire data        
def getMorlet(pID,ch_name,freqs,n_cycles,reference='Cpz'):
        if(ch_name[0]=='L' or ch_name[0]=='R'):
                electrode=ch_name[0]       
                raw=mne.io.read_raw('C:/Aditya/thalamus-census'+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,electrode)).pick("eeg")
                ch_names=np.array(raw.ch_names)
                raw.pick(ch_name).load_data()
                d=raw.get_data()[0]
                badMask=np.load(rootdir+"/IEDMask/%s_electrode_%s_IEDmask_convolved.npy"%(pID,electrode))[ch_names==ch_name][0]
                taxis=raw.times
                sfreq=raw.info['sfreq']
        else:
                taxis,d,badMask,sfreq=getScalpData(pID,ch_name,reference)
        
        d[badMask]=np.nan
        mwt = compute_wavelet_transform(d, fs=sfreq, n_cycles=n_cycles, freqs=freqs)	
        return taxis.copy(),freqs,mwt

	
def getConnectivityScalp(pID,
			ch_names_thal, 	#list of iEEG channels
			ch_names_scalp,	#list of scalp channels
			reference='Cpz',
			fmin=8,
			fmax=48,
			fdelta=1.0,
			n_cycles=10,
			fs=200.0
			):	
        freqs=np.arange(fmin,fmax,fdelta)
       
        print("Scalp channels:",ch_names_scalp)
        print("Thalamus channels:",ch_names_thal)
        
        #read sleepscore and upsample
        raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_L_eeg.fif'%(pID,pID))
        taxisSS,sleepScore=readSleepScoreFinal(pID)        
        sleepScore=scipy.interpolate.interp1d(taxisSS,sleepScore,bounds_error=False,fill_value=-1,kind='nearest')(raw.times)

        states=['wake','REM','NREM']
        nanFrac=np.zeros((len(ch_names_thal),len(ch_names_scalp)))
        for iChScalp in range(len(ch_names_scalp)):
        	#do morlet on scalp channel
                taxis,freqs,mwtScalp=getMorlet(pID=pID,ch_name=ch_names_scalp[iChScalp],
                				freqs=freqs,n_cycles=n_cycles,reference=reference)
                for iChThal in range(len(ch_names_thal)):
                       	#do morlet on thalamic channel
                        taxis,freqs,mwtThal=getMorlet(pID=pID,ch_name=ch_names_thal[iChThal],freqs=freqs,n_cycles=n_cycles)
                     
                        for state in states:
                                if(state=='wake'):
                                        sleepmask=sleepScore==0
                                elif(state=='NREM'):
                                        sleepmask=np.logical_or(sleepScore==3,sleepScore==2)
                                elif(state=='REM'):
                                        sleepmask=sleepScore==5
				#get imaginary component of correlation
                                mwtThalSelect=mwtThal[:,sleepmask]
                                mwtScalpSelect=np.conj(mwtScalp[:,sleepmask])
                                imagCij=np.imag(mwtThalSelect*mwtScalpSelect)
                                nanFrac[iChThal,iChScalp]=np.mean(np.isnan(imagCij))
                                print(ch_names_scalp[iChScalp],ch_names_thal[iChThal],state,mwtThalSelect.shape,nanFrac[iChThal,iChScalp])
                              	
                              	#compute pli and wpli
                                pli=np.nanmean(np.sign(imagCij),axis=1)
                                wpli=np.nanmean(imagCij,axis=1)/np.nanmean(np.abs(imagCij),axis=1)
				#compute average morlet spectrum
                                thalMean=np.nanmean(np.abs(mwtThalSelect),axis=1)
                                scalpMean= np.nanmean(np.abs(mwtScalpSelect),axis=1)
                               
                                #save to file
                                np.savetxt(rootdir+"/connectivity/pli_%s_%s-%s_%s_ref%s.txt"%(pID,ch_names_thal[iChThal],ch_names_scalp[iChScalp],state,reference),
                                           np.column_stack((freqs,pli,wpli,thalMean,scalpMean)),
                                           header='freqs \t pli \t dpli \t meanpower(thalamus)\t meanpower(scalp)\tpli')
                               
        #save nanfraction to file
        np.save(rootdir+"/connectivity/nfrac_%s.npy"%pID,nanFrac)

def getSingleSubj(pID,reference='Cpz'):
        if(pID=='p26' or pID=='p20'):
        	ch_names=['F7','F8','T7','P7','T8','P8','O1','O2']
        else:
          	ch_names=['Fp1','Fp2','F7','F8','T7','P7','T8','P8','O1','O2']      

        getConnectivityScalp(pID,ch_names_thal=['R1-R2','R2-R3','R3-R4','L1-L2','L2-L3','L3-L4'],ch_names_scalp=ch_names,reference='Cpz')


#iterate over cohort
cohort=['p03','p05','p09','p14','p14_followup',
			'p16','p18','p20','p21','p22','p30',
			'pthal101','pthal103','pthal104','pthal106']

for pID in cohort[2:]:
        getSingleSubj(pID,reference='Cpz')


