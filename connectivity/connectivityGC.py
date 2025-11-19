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
from mne_connectivity import spectral_connectivity_time
#'C:/Aditya/thalamus-census'
#function to load scalp data
def getScalpData(pID,ch_name_select,ref='Cpz'):
        
        #select consistently available channels
        if(pID=='p26' or pID=='p20'):
                ch_names_allsel=['F7','F8','T7','P7','T8','P8','O1','O2']
        else:
                ch_names_allsel=['Fp1','Fp2','F7','F8','T7','P7','T8','P8','O1','O2']
       
       	#load data and rereference         
        raw=mne.io.read_raw('C:/Aditya/thalamus-census'  +'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,'scalp')).pick("eeg")
        if(ref!='average'):
                ch_names_allsel=np.append(np.array(ch_names_allsel),[ref])
                ref=[ref]
                
        raw.pick(ch_names_allsel).load_data()
        mne.set_eeg_reference(raw, ref_channels=ref,copy=False)
        raw.pick(ch_name_select)
        
        #mask data only if more than half of the scalp contacts have artefacts
        badMask=np.load(rootdir+"/IEDMask/%s_electrode_scalp_IEDmask_convolved.npy"%(pID))
        badMask=np.mean(badMask,axis=0)>0.5
        d=raw.get_data()
        
        return raw.times,d,badMask,raw.info['sfreq']

#perform morlet transform of entire data    
# 
def getEpochs(pID,ch_name,chunkSizeInSec=10,reference='Cpz'):
        if(ch_name[0]=='L' or ch_name[0]=='R'):
                electrode=ch_name[0]      
                raw=mne.io.read_raw('C:/Aditya/thalamus-census' +'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,electrode)).pick("eeg")
                ch_names=np.array(raw.ch_names)
                raw.pick(ch_name).load_data()
                d=raw.get_data()
                badMask=np.load(rootdir+"/IEDMask/%s_electrode_%s_IEDmask_convolved.npy"%(pID,electrode))[ch_names==ch_name][0]
                taxis=raw.times
                sfreq=raw.info['sfreq']
        else:
                taxis,d,badMask,sfreq=getScalpData(pID,ch_name,reference)
        
        chunkSizeInSamples=int(chunkSizeInSec*sfreq)
        nChunks=int(d.shape[1]//chunkSizeInSamples)
        print(d.shape)
        if(pID=='p21'):
                badMask[:int(4e5)]=True
                badMask[int(1.329e7):]=True
       
        print(d.shape)
        d=d[:,:nChunks*chunkSizeInSamples].reshape((d.shape[0],nChunks,chunkSizeInSamples))
        print(d.shape)
        d=np.swapaxes(d,0,1)
        
        print(d.shape)
        badMask=np.mean(badMask[:nChunks*chunkSizeInSamples].reshape((nChunks,chunkSizeInSamples)),axis=1)>0
        print(np.mean(np.abs(d),axis=0))
        
        return taxis[:nChunks*chunkSizeInSamples:chunkSizeInSamples],d,np.logical_not(badMask),sfreq

def getGC(dataSeed,dataTarget,freqs,n_cycles,sfreq):
    nSeed=dataSeed.shape[1]
    nTarget=dataTarget.shape[1]
    data=np.append(dataSeed,dataTarget,axis=1)
    del dataSeed,dataTarget
    seed_ind=[]
    target_ind=[]

    #create mapping for seeds and targets
    for iSeed in range(nSeed):
            for iTarget in range(nTarget):
                    seed_ind.append([iSeed])
                    target_ind.append([nSeed+iTarget])
                    seed_ind.append([nSeed+iTarget])
                    target_ind.append([iSeed])

    indices=(seed_ind,target_ind)
   
    con_gc_ab=spectral_connectivity_time(data,
                        method=['gc','gc_tr'],
                        indices=indices,
                        mode='cwt_morlet',average=False,
                        freqs=freqs,n_cycles=n_cycles,sfreq=sfreq,n_jobs=50,verbose=False)
  
    return con_gc_ab,np.array(seed_ind).flatten(),np.array(target_ind).flatten()
def getConnectivityScalp(pID,
			ch_names_thal, 	#list of iEEG channels
			ch_names_scalp,	#list of scalp channels
			reference='Cpz',
			fmin=8,
			fmax=48,
			fdelta=1.0,
			n_cycles=10,
			fs=200.0,
                        chunkSizeInSec=10.0
			):	
        freqs=np.arange(fmin,fmax,fdelta)
       
        print("Scalp channels:",ch_names_scalp)
        print("Thalamus channels:",ch_names_thal)
        
        #read sleepscore and upsample
        taxisSS,sleepScore=readSleepScoreFinal(pID)        
        sleepScoreFunc=scipy.interpolate.interp1d(taxisSS,sleepScore,bounds_error=False,fill_value=-1,kind='nearest')

        states=['wake','REM','NREM']
        #get epochs scalp channel
        taxis,epochsScalp,selmaskScalp,sfreq=getEpochs(pID=pID,ch_name=ch_names_scalp,
                                        reference=reference,chunkSizeInSec=chunkSizeInSec)
        print(epochsScalp.shape)
        for iChThal in range(len(ch_names_thal)):
                #get epochs for thalamic channel
                ch_names_all=np.append(ch_names_thal[iChThal],ch_names_scalp)
                print(ch_names_all)
                taxis,epochsThal,selmaskThal,sfreq=getEpochs(pID=pID,ch_name=ch_names_thal[iChThal],chunkSizeInSec=chunkSizeInSec)
                #combine selmasks
                selmask=np.logical_and(selmaskScalp,selmaskThal)
                epochsThal=epochsThal[selmask]
                print("Selecting %d out of %d epochs"%(np.sum(selmask),len(selmask)))
                con_gc,indSeed,indTarget=getGC(epochsThal,epochsScalp[selmask],freqs=freqs,n_cycles=n_cycles,sfreq=sfreq)
                del epochsThal
                sleepScore=sleepScoreFunc(taxis[selmask])
                
                for state in states:
                        if(state=='wake'):
                                sleepmask=sleepScore==0
                        elif(state=='NREM'):
                                sleepmask=np.logical_or(sleepScore==3,sleepScore==2)
                        elif(state=='REM'):
                                sleepmask=sleepScore==5
                        #get statewise average
                        df_avg=pd.DataFrame()
                        df_avg['freqs']=np.array(con_gc[0].freqs)
                        for iCh in range(0,len(indSeed)):
                                df_avg['gc_%s->%s'%(ch_names_all[indSeed[iCh]],ch_names_all[indTarget[iCh]])]=np.mean(con_gc[0].get_data()[sleepmask][:,iCh],axis=0)  
                                df_avg['gctr_%s->%s'%(ch_names_all[indSeed[iCh]],ch_names_all[indTarget[iCh]])]=np.mean(con_gc[1].get_data()[sleepmask][:,iCh],axis=0)  
                                #save to file
                        df_avg.to_csv(rootdir+"/connectivity/gc_%s_%s_%s_ref%s.csv"%(pID,ch_names_thal[iChThal],state,reference))
                del con_gc     

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

for pID in cohort:
        getSingleSubj(pID,reference='Cpz')


