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

from core import *
from core.helpers import *
from averageBursts import *
from burst.coreFunctions import *
import matplotlib.gridspec as gridspec

import matplotlib

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 

#get burst widths for all profiles
def getAllBurstWidths():
	#load evoked during REM sleep
	taxis,evoked_REM,df_REM=getEvokedResponse(getSignificantBands('gammaREMCorrelated'),'REM')
	df_REM['amp'],df_REM['width'],df_REM['zscoreExtent']=getAmpWidth(taxis,evoked_REM)
	
	#load evoked during wakefullness	
	taxis,evoked_wake,df_wake=getEvokedResponse(getSignificantBands('gammaREMCorrelated'),'wake')	
	df_wake['amp'],df_wake['width'],df_wake['zscoreExtent']=getAmpWidth(taxis,evoked_wake)	
	
	#combine REM and wake responses
	evoked_combined=(evoked_REM+evoked_wake)/2.
	df_combined=df_REM.copy()
	df_combined['amp'],df_combined['width'],df_combined['zscoreExtent']=getAmpWidth(taxis,evoked_combined)	
	
	#load spindle evoked
	taxis,evoked_NREM,df_NREM=getEvokedResponse(getSignificantBands('spindleInGammaChannels'),'NREM')
	df_NREM['amp'],df_NREM['width'],df_NREM['zscoreExtent']=getAmpWidth(taxis,evoked_NREM)

	#merge dataframes on pIDs and contacts
	df_merged=df_NREM.merge(df_combined,left_on=['pID','ch_name'],right_on=['pID','ch_name'],validate='m:m')

	return df_wake,df_REM,df_NREM,df_merged

#bootstrap to get p-value
def calculatePValue(x,y,niter):
	true_r=np.mean((x-x.mean())*(y-y.mean()))/(x.std()*y.std())
	rvals=np.zeros(int(niter))
	nPoints=len(x)	
	for iIter in range(niter):
		select=np.random.randint(low=0,high=nPoints,size=nPoints)
		xthis=x[select]
		ythis=y[select]		
		rvals[iIter]=np.mean((xthis-xthis.mean())*(ythis-ythis.mean()))/(xthis.std()*ythis.std())
	print("Number of points: %d"%nPoints)
	print("True r: %.3f"%true_r)
	print("Mean r value: %.3f"%np.mean(rvals))
	print("Error on r value: %.3f"%np.std(rvals))
	print("Pvalue for positive correlation: %.2e"%np.mean(rvals<0)) 
	return np.mean(rvals<0)
	
#plot regression with pvalue
def plotCorrelationHelper(x,y,ax,niter=int(1e5)):
	r=scipy.stats.linregress(x,y)
	pearson=scipy.stats.pearsonr(x,y,alternative='greater')
	pvalue=calculatePValue(x,y,niter=niter)
	if(pvalue<=1/niter):
		pvaltext=r'p-value<10$^{-%d}$'%np.log10(niter)
	elif(pvalue<1e-4):
		pvaltext='p-value<%d x$10^{-4}$'%(pvalue/1e-4)
	else:
		pvaltext='p-value=%.4f'%pvalue	
	xrnge=np.linspace(np.min(x),np.max(x),100)
	ax.plot(xrnge,r.slope*xrnge+r.intercept,ls='--',c='black',label='r = %.2f\n%s'%(pearson.statistic,pvaltext))
	ax.legend()	
		
#plot correlations
def plotCorrelation(ax_width,
			ax_amp,	
			whichWidth='width',
			whichCorrelation='fosc_spindle'
			):
	
	#get dataframes with widths
	df_wake,df_REM,df_NREM,df_merged=getAllBurstWidths()
	
	#choose variable according to selected correlation
	if(whichCorrelation=='fosc_spindle'):
		width_1,amp_1=df_merged['%s_x'%whichWidth].values,df_merged['amp_x'].values
		width_2,amp_2=df_merged['%s_y'%whichWidth].values,df_merged['amp_y'].values
	elif(whichCorrelation=='fosc_state'):
		width_1,amp_1=df_REM['%s'%whichWidth].values,df_REM['amp'].values
		width_2,amp_2=df_wake['%s'%whichWidth].values,df_wake['amp'].values	
			
	#set reimplantation epoch to have same pID
	pIDs=df_merged['pID'].values.copy()
	pIDs[pIDs=='p14_followup']='p14'
	uniqPID=np.unique(df_merged['pID'])

	#scatter plot 
	markers=['o',"v","^","<",">","8","s","p","P","h","X","D","H","d"]
	c=['C0','C3','C4','C5','C6','C7']
	for i in range(0,len(uniqPID)):
		pIDmask=df_merged['pID']==uniqPID[i]
		ax_width.scatter(width_1[pIDmask],width_2[pIDmask],marker=markers[i%5],c=c[i%6],s=20)
		ax_amp.scatter(amp_1[pIDmask],amp_2[pIDmask],marker=markers[i%5],c=c[i%6],s=20)

	print("----")	
	print("width correlation")
	print("----")	
	plotCorrelationHelper(width_1,width_2,ax_width)
	
	print("----")	
	print("amp correlation")
	print("----")	
	plotCorrelationHelper(amp_1,amp_2,ax_amp)


	if(whichCorrelation=='fosc_spindle'):
		ax_amp.set_xlabel(r"Spindle Amplitude ($\mu$V)")
		ax_amp.set_ylabel(r"Fast Oscillation Amplitude ($\mu$V)")
		ax_width.set_xlabel(r"Spindle Width (equivalent FWHM, in sec)")
		ax_width.set_ylabel(r"Fast Oscillation Width (equivalent FWHM, in sec)")
	elif(whichCorrelation=='fosc_state'):
		ax_amp.set_xlabel(r"Fast Oscillation Amplitude, REM ($\mu$V)")
		ax_amp.set_ylabel(r"Fast Oscillation Amplitude, wake ($\mu$V)")
		ax_width.set_xlabel("Fast Oscillation width, REM\n(eqv. FWHM, in sec)")
		ax_width.set_ylabel(r"Fast Oscillation width, wake (eqv. FWHM, in sec)")

#plot example bursts
def plotSingleAverage(pIDs,		#list of pIDs
			ch_names,	#list of corresponding
			axs_gamma,	#axes to draw fast oscillatory bursts on
			axs_spindle	#axes to draw spindles on
			):
	#load evoked responses
	taxis,evoked_wake,df_wake=getEvokedResponse(getSignificantBands('gammaREMCorrelated'),'wake')
	taxis,evoked_REM,df_REM=getEvokedResponse(getSignificantBands('gammaREMCorrelated'),'REM')
	taxis,evoked_NREM,df_NREM=getEvokedResponse(getSignificantBands('spindleInGammaChannels'),'NREM')
	#combine wake and REM	
	evoked_gamma=(evoked_REM+evoked_wake)/2.
	
	#loop over example patients
	for i in range(0,len(pIDs)):
		#select response for current pID and channel
		evoked_gamma_this=evoked_gamma[np.logical_and(df_wake['pID']==pIDs[i],df_wake['ch_name']==ch_names[i])]
		evoked_spindle_this=evoked_NREM[np.logical_and(df_NREM['pID']==pIDs[i],df_NREM['ch_name']==ch_names[i])]		
	
		#plot evoked responses with envelope
		axs_gamma[i].plot(taxis,evoked_gamma_this[-1],c='C1',lw=0.5)
		evokedEnvGamma=np.abs(scipy.signal.hilbert(evoked_gamma_this[-1]))
		axs_gamma[i].plot(taxis,evokedEnvGamma,ls='--',c='C1',lw=0.5)
				
		evokedEnvSpindle=np.abs(scipy.signal.hilbert(evoked_spindle_this[0]))
		axs_spindle[i].plot(taxis,evoked_spindle_this[0],c='C2',lw=0.5)
		axs_spindle[i].plot(taxis,evokedEnvSpindle,ls='--',c='C2',lw=0.5)

		
		axs_spindle[i].set_xlim((-1,1))
		axs_gamma[i].set_xlim((-1,1))	
		axs_spindle[i].set_ylim((-12.5,12.5))
		axs_gamma[i].set_ylim((-6.5,6.5))
		axs_gamma[i].set_ylabel(r"Amp ($\mu$V)")
		axs_spindle[i].set_ylabel(r"Amp ($\mu$V)")	
		axs_gamma[i].minorticks_on()
		axs_spindle[i].minorticks_on()
	

	
	axs_gamma[-1].set_xlabel("Time (sec)")
	axs_spindle[-1].set_xlabel("Time (sec)")	

def plotFigure4():
	fig = plt.figure(figsize=(14, 5))
	plt.subplots_adjust(hspace=0.5,wspace=0.7)
	gs = gridspec.GridSpec(4,6)
	ax_spindle_example=[]
	ax_gamma_example=[]
	for i in range(0,4):	
		ax_spindle_example.append(fig.add_subplot(gs[i,0]))
		ax_gamma_example.append(fig.add_subplot(gs[i,1]))

	ax_spindle_example[0].set_title("(A) Spindle",loc='left',fontdict={'fontweight':'bold','fontsize':10})
	ax_gamma_example[0].set_title("(B) Fast Oscillation",loc='left',fontdict={'fontweight':'bold','fontsize':10})

	ax_corr_amp=fig.add_subplot(gs[:, 2:4])
	ax_corr_width=fig.add_subplot(gs[:, 4:])
	
	#plot example bursts
	plotSingleAverage(['p03','pthal106','p21','p18']
			,['R3-R4','R1-R2','L2-L3','R1-R2'],ax_gamma_example,ax_spindle_example)
	
	
	ax_corr_amp.set_title("(C) Amplitude Correlation",loc='left',fontdict={'fontweight':'bold','fontsize':10})
	ax_corr_width.set_title("(D) Width Correlation",loc='left',fontdict={'fontweight':'bold','fontsize':10})

	#plot correlations
	plotCorrelation(ax_corr_width,ax_corr_amp,whichWidth='width',whichCorrelation='fosc_spindle')
	
	plt.savefig("figures/figure4.pdf",bbox_inches='tight',dpi=600.0)

def plotWakeREMCorrelation():
	fig = plt.figure(figsize=(8, 4))
	plt.subplots_adjust(hspace=0.5,wspace=0.7)
	gs = gridspec.GridSpec(4,4)

	ax_corr_amp=fig.add_subplot(gs[:, 0:2])
	ax_corr_width=fig.add_subplot(gs[:, 2:])
	
	ax_corr_amp.set_title("(A) Amplitude Correlation",loc='left',fontdict={'fontweight':'bold','fontsize':10})
	ax_corr_width.set_title("(B) Width Correlation",loc='left',fontdict={'fontweight':'bold','fontsize':10})

	plotCorrelation(ax_corr_width,ax_corr_amp,whichWidth='width',whichCorrelation='fosc_state')
	ax_corr_width.set_ylim(0.05,0.95)
	plt.savefig("figures/edf1.pdf",bbox_inches='tight',dpi=300)

def plotThresholdedWidth():

	
	fig = plt.figure(figsize=(6, 5))
	plt.subplots_adjust(hspace=0.5,wspace=0.7)
	gs = gridspec.GridSpec(1,1)

	ax_corr_width=fig.add_subplot(gs[:, :])
	figdummy,axsdummy = plt.subplots()	#dummy axes to draw amplitude correlation on
	plotCorrelation(ax_corr_width,axsdummy,whichWidth='zscoreExtent',whichCorrelation='fosc_spindle')
	fig.savefig("figures/edf6.pdf",bbox_inches='tight',dpi=300)
	

plotFigure4()		
plotWakeREMCorrelation()
plotThresholdedWidth()
