import os
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)


import os
import mne
import scipy
import matplotlib.pyplot as plt
from scipy.signal import medfilt,find_peaks
import pandas as pd
from numba import njit,jit
import numpy as np
from scipy.signal import medfilt
import matplotlib.gridspec as gridspec

from core import *
from core.helpers import *

#function toget deflections in scalp EEG during eye movements in REM sleep
def getTopoAtSaccades(pID):
	#load eye movement events
	eyeMovParams=pd.read_csv(rootdir+"/eyeMovParams/%s_eyeMovEvents_alldetections.csv"%pID)
	eyeMovParams=eyeMovParams.sort_values(by='peakVelocityIndex')
	#select events during REM sleep
	selmask=eyeMovParams['sleepScore']==5
	ton=eyeMovParams['onsetIndex'].values[selmask].astype("int")
	toff=eyeMovParams['offsetIndex'].values[selmask].astype("int")
	peakVel=eyeMovParams['peakVelocity'].values[selmask]
	#load raw file
	raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,'scalp'),preload=False)
	ch_names=raw.ch_names	
	
	#save deflections to file
	df=pd.DataFrame()
	for i in range(0,len(ch_names)):
		d=raw.get_data(picks=[ch_names[i]])
		deflection=(d[0,toff]-d[0,ton])*np.sign(peakVel)
		df[ch_names[i]]=deflection
	df.to_csv(rootdir+"/eyeMovParams/%s_deflections.csv"%pID,index=False)
	
#plot topo for a given patient	
def plotTopoMap(pID,
		ax		#matplotlib axes to draw on
		):
		
	#load deflections
	df=pd.read_csv(rootdir+"/eyeMovParams/%s_deflections.csv"%pID)
	ch_names=np.array(df.columns)
	
	#rereference to Cpz
	data=df.to_numpy()-np.expand_dims(df['Cpz'].values,axis=1)
	data=data
	
	#get mean topo across all events
	topomean=np.mean(data,axis=0)
	if(np.std(topomean)<1e-3):
		topomean*=1e6 #converting to microvolt for data in volts

	#get montage and plot	
	pos = mne.channels.make_standard_montage("standard_1020").get_positions()['ch_pos']
	posArray=np.zeros((len(ch_names),2))
	for i in range(0,len(ch_names)):
		if(ch_names[i]=='Cpz'):
			ch_names[i]='CPz'
		posArray[i]=pos[ch_names[i]][:2]
	im,cn=mne.viz.plot_topomap(topomean,posArray,show=False,axes=ax,border=0,vlim=[-80,80])	
	
	return im



'''
#save deflections
for pID in cohortForPaper:
	getTopoAtSaccades(pID)
	
	
	


'''
#plot topo
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(4, 6, height_ratios = [1,1,1,0.1])
newlabels=['pp1','pp2','pp3','pp4','pp5','pp5 (reimplant)','pp6','pp7','pp8','pp9','pp10','pp11','pp12','pp13','pp14','pp15','pp16','pp17']		
for i in range(0,len(cohortForPaper)):
	ax=fig.add_subplot(gs[i//6,i%6])
	ax.set_title(newlabels[i])
	im=plotTopoMap(cohortForPaper[i],ax)

plt.colorbar(im,cax=fig.add_subplot(gs[3,2:4]),label='average deflection (microVolts)',orientation='horizontal')
plt.savefig("figures/edf5.pdf",bbox_inches='tight')

