import numpy as np
from numba import njit,jit,prange
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline

from seaborn import violinplot
from .params import *
import scipy


#function to get referenced scalp data
def getScalpData(pID,ch_name_select,ref='Cpz'):
        if(pID=='p26' or pID=='p20'):
                ch_names_allsel=['F7','F8','T7','P7','T8','P8','O1','O2']
        else:
                ch_names_allsel=['Fp1','Fp2','F7','F8','T7','P7','T8','P8','O1','O2']        
                
        raw=mne.io.read_raw('C:/Aditya/thalamus-census/'+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,'scalp')).pick("eeg")
        if(ref!='average'):
                ch_names_allsel=np.append(np.array(ch_names_allsel),[ref])
                ref=[ref]
        raw.pick(ch_names_allsel).load_data()
        raw,t=mne.set_eeg_reference(raw, ref_channels=ref,copy=False)
        raw.pick(ch_name_select)
        badMask=np.load(rootdir+"/IEDMask/%s_electrode_scalp_IEDmask_convolved.npy"%(pID))
        badMask=np.mean(badMask,axis=0)>0.5
        d=raw.get_data()[0]
        return raw.times,d,badMask,raw.info['sfreq']
        
        
#standard hypnogram plot
def plotStandardHypnogram(taxis,
		sleepscore,
		ax, #matplotlib axes to draw on
		c='black', #line colour
		combineNREM=False #combine N1 and N2
		):
	#interpolate artefact scores, just for better plot vizualization
	
	sScoreNew=scipy.interpolate.interp1d(taxis[sleepscore<=5],sleepscore[sleepscore<=5],kind='nearest',bounds_error=False,fill_value=-1)(taxis)
	
	#fixing order to plot the hypnogram
	sleepscore=sScoreNew.copy()
	sScoreNew[sleepscore==0]=4
	sScoreNew[sleepscore==5]=3
	sScoreNew[sleepscore==1]=2	
	sScoreNew[sleepscore==2]=1
	if(combineNREM):
		sScoreNew[sleepscore==3]=1	
	else:
		sScoreNew[sleepscore==3]=0
		
		
	ax.plot(taxis/3600.0,sScoreNew,c=c)
	sScoreNew[np.logical_not(sScoreNew==3)]=np.nan
	ax.plot(taxis/3600.0,sScoreNew,c=c,lw=4)	
	ax.set_xlabel("Time (hour)")
	
	ax.minorticks_on()
	ax.tick_params(axis='y', which='minor', left=False)
	if(combineNREM):
		ax.set_yticks(np.array([2,3,4,5])-1,['NREM','N1','REM','Wake'])
		ax.set_ylim((0.5,4.5))
	else:
		ax.set_yticks(np.array([1,2,3,4,5])-1,['N3','N2','N1','REM','Wake'])	
		ax.set_ylim((-0.5,4.5))
		
		
#read sleepscore from standard file
def readSleepScoreFinal(pID,
			scoreSamplingInterval=30.0
			):
	
	sleepScore=np.loadtxt(rootdir+"/sleepscore/final/%s_sleepscore.txt"%pID,usecols=[0])	
	return np.arange(0,len(sleepScore))*scoreSamplingInterval+scoreSamplingInterval/2.,sleepScore
	
	
#function to dilate mask
@njit
def expandMaskFunc(mask,sample):
	maskExpanded=np.zeros_like(mask)
	for i in range(sample,len(mask)-sample):	
		if(mask[i]):
			maskExpanded[i-sample:i+sample]=True
	maskExpanded[:sample]=True
	maskExpanded[-sample:]=True
	return maskExpanded

#function to return mask for bad data
def getBADMask(pID,
		electrode,
		sfreq=200.0,
		outsfreq=None,
		expandMaskInSec=1,
		combineChannels=True):
	
	#read mask at raw sampling
	IEDMask=np.load(rootdir+"/IEDMask/%s_electrode_%s_IEDmask.npy"%(pID,electrode))
	taxis=np.arange(IEDMask.shape[1])/sfreq
	
	#if asked for, combine channels in the electrode to get a single mask for all channels (e.g. bad if any channel is bad)
	if(combineChannels):
		IEDMask=np.sum(IEDMask,axis=0,keepdims=True)>0
	
	#dilate mask
	if(not expandMaskInSec is None):
		expandMaskSamples=int(expandMaskInSec*sfreq)
		for i in range(0,len(IEDMask)):
			IEDMask[i]=expandMaskFunc(IEDMask[i],expandMaskSamples)	
			
	#initial 9 hours on left electrode for p09 has recording issues. Manually marking it here.
	if(pID=='p09' and electrode=='L'):
		IEDMask[:,taxis<33000]=True	
	
	#average mask to outsfreq, if asked for.
	if(not outsfreq is None):
		intfactor=int(np.round(sfreq/outsfreq))
		nSampNew=IEDMask.shape[1]//intfactor
		IEDMask=np.mean(IEDMask[:,:nSampNew*intfactor].reshape(IEDMask.shape[0],nSampNew,intfactor),axis=2)>0
	return IEDMask
	
	



		
