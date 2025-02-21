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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib import ticker
#from scipy.stats import PermutationMethod
from fooof import FOOOF
from core import *
from core.helpers import *
import matplotlib.gridspec as gridspec
from psd.coreFunctions import getPSD
import matplotlib

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 	
	

def fooof(freqs,spec,peak_width_limits=[2, 25]):	
	fg = FOOOF(peak_width_limits=peak_width_limits)#peak_width_limits=[1, 12], min_peak_height=0.05, max_n_peaks=3)
	fg.fit(freqs,spec)
	aperiodicModel=fg.get_model(space='linear',component='aperiodic')
	return spec-aperiodicModel

# get subject level 1/f subtracted average power spectra
def getAlignedPeriodicPower(df, #dataframe with list of bands
			flim=[8,48],	#frequency limit for 1/f fitting
			ch_name_sel='Thalamus' #which channel (thalamus or scalp)
			):
	#commong frequency axes to interpolate aligned spectra to
	freqAxisCommon=np.arange(-6,6.1,0.25)
	
	powerCommon=np.zeros((len(df),3,len(freqAxisCommon)))
	states=['wake','REM','NREM']
	print("Number of frequency bands:",len(df))
	
	i=0
	for indx in df.index:
		if(ch_name_sel =='Thalamus'):
			ch_name=df.loc[indx,'ch_name']
		else:
			ch_name=ch_name_sel
		#channels that are not available in p20 and p26
		if((ch_name=='Fp1' or ch_name=='Fp2') and (df.loc[indx,'pID']=='p20' or df.loc[indx,'pID']=='p26')):
			continue
			
		
		freqLow=df.loc[indx,'freqLow']
		freqHigh=df.loc[indx,'freqHigh']	
		freqPeak=df.loc[indx,'freqPeak']
		
		#load time-average power spectra
		df_psd=pd.read_csv(repo.working_tree_dir+"/psd/outfiles/%s_psd.csv"%df.loc[indx,'pID'])	
		freqs=df_psd['freqs'].values
		freqmask=np.logical_and(freqs>flim[0],freqs<=flim[1])
		freqs=freqs[freqmask]
		
		#subtract aperiodic and align to common frequency axes
		for istate in range(len(states)):	
			spec=df_psd['%s_%s'%(ch_name,states[istate])].values[freqmask]
			spec=fooof(freqs,spec)
			powerCommon[i,istate]=np.interp(freqAxisCommon,freqs-freqPeak,spec)
			
		i+=1
	
	
	#combine reimplantation epoch with original epoch in p14 (because subject-level average)
	pIDs=df['pID'].values
	pIDs[pIDs=='p14_followup']='p14'
	uniqPID=np.unique(df['pID'])
	
	#get a single spectra for each subject
	powerGroup=np.zeros((len(uniqPID),3,len(freqAxisCommon)))
	for iPID in range(0,len(uniqPID)):
		powerGroup[iPID]=np.mean(powerCommon[uniqPID[iPID]==df['pID']],axis=0)
	return freqAxisCommon,powerGroup


	
