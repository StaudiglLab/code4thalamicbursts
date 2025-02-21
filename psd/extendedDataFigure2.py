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
import seaborn as sns
import matplotlib

from core import *
from core.helpers import *
from periodicPowerInBand import getAlignedPeriodicPower
from burst.coreFunctions import *

import matplotlib.gridspec as gridspec




matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 	

def plotPowerOnScalp():
	#get dataframe containing frequency bands
	df=getSignificantBands('gamma')

	#groups of electrodes
	groups={'(A) Thalamus (iEEG)':['Thalamus'],'(B) Frontal (scalp)':['Fp1','Fp2','F7','F8'],'(C) Central (scalp)':['T7','P7','T8','P8'],'(D) Occipital (scalp)':['O1','O2']}	
	states=['wake','REM','NREM']
	
	groupNames=list(groups.keys())	
	
	#get PSD on thalamus
	freqs,powerThal=getAlignedPeriodicPower(df,ch_name_sel='Thalamus')
	
	#create structure for all PSDs
	averagePowers={}
	averagePowerGroups=np.zeros((len(groupNames),powerThal.shape[0],powerThal.shape[1],powerThal.shape[2]))		
	averagePowerGroups[0]=powerThal
	
	#read power on scalp electrode groups
	for iGroup in range(1,len(groupNames)):
		ch_names=groups[groupNames[iGroup]]
		averagePowers=np.zeros((len(ch_names),powerThal.shape[0],powerThal.shape[1],powerThal.shape[2]))
		#average over all electrodes in each group
		for iChan in range(len(ch_names)):
			freqs,averagePowers[iChan]=getAlignedPeriodicPower(df,ch_name_sel=ch_names[iChan])
		averagePowerGroups[iGroup]=np.nanmean(averagePowers,axis=0)
		
		
	#normalize (doesn't change scale across states and frequencies)
	averagePowerGroups=averagePowerGroups/np.std(averagePowerGroups[:,:,:,:],axis=(2,3),keepdims=True)

	nSubj=averagePowerGroups.shape[1]
	fig,axs=plt.subplots(1,4,figsize=(16,4),sharey=True,sharex=True)
	plt.subplots_adjust(bottom=0.15)
	
	#plot group level means and S.E.M
	for i in range(0,len(groupNames)):
		mean=np.mean(averagePowerGroups[i],axis=0)
		std=np.std(averagePowerGroups[i],axis=0)/np.sqrt(nSubj)
		
		for iState in range(0,len(states)):
		
			axs[i].plot(freqs,mean[iState],label=states[iState])
			axs[i].fill_between(freqs,mean[iState]-std[iState],mean[iState]+std[iState],alpha=0.5)
		if(i==0):
			axs[i].legend()
		axs[i].set_title(groupNames[i])
		axs[i].set_ylabel("Power (a.u.)")
		axs[i].set_xlabel("Relative Frequency (Hz)")	
		axs[i].set_xlim((-6,6))
		axs[i].set_xticks([-5,0,5],[r"f$_\mathrm{osc}$-5 Hz", "f$_\mathrm{osc}$","f$_\mathrm{osc}$+5 Hz"])
		axs[i].minorticks_on()

	plt.savefig("figures/edf2.pdf",bbox_inches='tight',dpi=300.0)	
	
plotPowerOnScalp()	
