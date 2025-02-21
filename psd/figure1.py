import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy


import matplotlib.gridspec as gridspec
from matplotlib import ticker
from PIL import Image

from core import *
from core.helpers import *
from coreFunctions import getMorletPSD


import matplotlib
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 


def singlePSD(pID,
	ch_name,
	ax_ps, #axes to draw power spectra on
	ax_ss, #axes to draw sleep score on
	tcrop=None):
	
	#read morlet amplitude and divide by median
	taxis,freqs,powerdata,ch_names,sleepScore=getMorletPSD(pID,ch_name[0],n_cycle_morlet=15)	
	powerdata/=np.nanmedian(powerdata)
	
	#read starting timestamp from original EDF
	raw=mne.io.read_raw(rawfileName(pID),preload=False,verbose=False)
	tstart=raw.info['meas_date'].hour*3600.+raw.info['meas_date'].minute*60.0+raw.info['meas_date'].second

	#crop power spectra for plot
	if(not tcrop is None):
	    selmask=np.logical_and(taxis/3600.0>tcrop[0],taxis/3600.0<=tcrop[1])
	    taxis=taxis[selmask]
	    sleepScore=sleepScore[selmask]
	    powerdata=powerdata[selmask]
	taxis=taxis+tstart
	
	#select channel
	ch_names=np.array(ch_names)
	powerdata=powerdata[:,ch_names==ch_name][:,0]
	
	#call helper function to plot hypnogram
	plotStandardHypnogram(taxis,sleepScore,ax_ss,c='black',combineNREM=True)
	
	
	
	taxis=taxis/3600.
	powerSel=powerdata[:,freqs>10]
	nanMask=np.logical_not(np.logical_or(np.isinf(powerSel),np.isnan(powerSel)))
	if(np.sum(nanMask)==0):
		return ax
	
	#determine range on colour scale, saturate a bit on either side for better vizualization
	vmin=np.quantile(powerSel[nanMask],0.02)
	vmax=np.quantile(powerSel[nanMask],0.999)
	powerdata[np.logical_or(np.isinf(powerdata),np.isnan(powerdata))]=vmin-1
	im=ax_ps.imshow(powerdata.T,aspect='auto',origin='lower',vmin=vmin,vmax=vmax,extent=(taxis[0],taxis[-1],freqs[0],freqs[-1]),cmap='inferno')
	
	xticks=np.arange(np.round(taxis[0]),np.round(taxis[-1]),2)
	labels=[]
	for i in range(0,len(xticks)):
		labels.append('%02d'%(xticks[i]%24))
	ax_ps.set_xticks(xticks,labels)	
	ax_ps.minorticks_on()
	
	ax_ss.set_xticks(xticks,labels)	

	ax_ss.set_xlim(taxis[0],taxis[-1])
	ax_ps.set_xlim(taxis[0],taxis[-1])
	ax_ss.set_xlabel("Local Time (hr)")	
	ax_ps.set_xlabel("Local Time (hr)")			

	return im

def plotFigure1():
	fig = plt.figure(figsize=(8, 6))
	gs = gridspec.GridSpec(4,40,height_ratios=[1,1.5,0.05,0.5])
	
	#plot example electrode trajectory
	slice1 = np.asarray(Image.open('exampleElectrode/slice1.png'))
	slice2 = np.asarray(Image.open('exampleElectrode/slice2.png'))
	slice3 = np.asarray(Image.open('exampleElectrode/slice3.png'))
	#ax_text0=fig.add_subplot(gs[0, 1])
	ax_slice1=fig.add_subplot(gs[0, 5:13])
	ax_slice2=fig.add_subplot(gs[0, 15:23])
	ax_slice3=fig.add_subplot(gs[0, 25:33])
	ax_slice1.imshow(slice1,interpolation='bicubic')
	#ax_slice1.set_xlim(
	ax_slice2.imshow(slice2,interpolation='bicubic')
	ax_slice3.imshow(slice3,interpolation='bicubic')
	ax_slice1.set_axis_off()
	ax_slice2.set_axis_off()
	ax_slice3.set_axis_off()
	ax_slice1.set_title("(A) electrode trajectory",loc='left',fontdict={'fontweight':'bold','fontsize':8})


	#plot thalamic field potential and sleep score.
	ax_ps=fig.add_subplot(gs[1, :-1])
	ax_ss=fig.add_subplot(gs[3, :-1])
	ax_cb=fig.add_subplot(gs[1, -1])	
	ax_ps.set_title("(B) thalamic field potential",loc='left',fontdict={'fontweight':'bold','fontsize':8})
	ax_ss.set_title("(C) sleep architecture",loc='left',fontdict={'fontweight':'bold','fontsize':8})
	ax_ps.set_ylabel("Frequency (Hz)")
	im=singlePSD('p21','R1-R2',ax_ps,ax_ss,tcrop=[0.9,18.4])
	plt.colorbar(im,cax=ax_cb,label='Morlet Amplitude (a.u.)',location='right')	
	plt.savefig("figures/figure1.png",bbox_inches='tight',dpi=600)		

plotFigure1()	
