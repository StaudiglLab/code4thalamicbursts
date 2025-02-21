import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from core import *
from core.helpers import *
from coreFunctions import getBurstRate2D,getSignificantBands,getOverlaps
from plotHelpers import *
from psd.periodicPowerInBand import getAlignedPeriodicPower

import matplotlib.gridspec as gridspec

import matplotlib
sns.set_palette("deep")
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 


#plot summary of detected frequency bands
def plotSummaryDetection(ax_Wake, 	#axes to draw histogram of wake detections
			ax_REM,		#axes to draw histogram of REM detections
			ax_NREM		#axes to draw histogram of NREM detections
			):
	#load all significant bands	
	df_freqs=getSignificantBands(which='all')		
	states=df_freqs['state'].values
	peakFreq=df_freqs['freqPeak'].values


	statelabel=['wake','REM','NREM']
	axs=[ax_Wake,ax_REM,ax_NREM]
	
	#draw histogram for state
	for i in range(0,3):
		axs[i].hist(peakFreq[states==statelabel[i]],bins=np.logspace(np.log2(10),np.log2(42),15,base=2),fc='C%d'%i,label=statelabel[i])
		axs[i].set_ylim((0,50))
		axs[i].minorticks_on()
		axs[i].legend()		
		axs[i].set_ylabel("Number of contacts")		
	axs[2].set_xlabel("Peak Frequency of\n Oscillatory Band (Hz)")

#plot ratio of burst rates in bands
def plotBurstRateRatio(ax_ratioWake,ax_ratioREM):
	df_uniq=getSignificantBands(which='allUnique') #get all unique bands
	print("Number of unique bands:%d"%len(df_uniq))
	
	peakFreq=df_uniq['freqPeak'].values
	
	#ratio of NREM to wake and REM burst
	ratioWake=np.log10(df_uniq['meanBurstRate_wake'].values/df_uniq['meanBurstRate_NREM'].values)
	ratioREM=np.log10(df_uniq['meanBurstRate_REM'].values/df_uniq['meanBurstRate_NREM'].values)
	
	#scatter plot of ratios
	ax_ratioWake.scatter(peakFreq,ratioWake,s=1,label='Wake',c='black',zorder=1000)	
	ax_ratioREM.scatter(peakFreq,ratioREM,s=1,label='Wake',c='black',zorder=1000)		
	
	#set yticks to reflect log scale
	ax_ratioWake.set_yticks([-1,0,1],["0.1","1","10"])
	ax_ratioREM.set_yticks([-1,0,1],["0.1","1","10"])
	
	#set labels and horizontal lines
	ax_ratioWake.set_ylabel("Ratio Burst Rate, Wake to NREM")	
	ax_ratioREM.set_ylabel("Ratio Burst Rate, REM to NREM")	
	ax_ratioWake.axhline(0,ls='--',c='black')	
	ax_ratioREM.axhline(0,ls='--',c='black')	
	
	
	#bin frequency bands
	ratioWakeBinned=[]
	ratioREMBinned=[]	
	binWidth=5 # 5 Hz bins
	start=10  # starting at 10 Hz
	
	for freqBin in np.arange(start,36,binWidth):
		ratioWakeBinned.append(ratioWake[np.logical_and(peakFreq>=freqBin,peakFreq<freqBin+binWidth)])
		ratioREMBinned.append(ratioREM[np.logical_and(peakFreq>=freqBin,peakFreq<freqBin+binWidth)])

	#draw boxplots
	ax_ratioWake.boxplot(ratioWakeBinned,positions=np.arange(start+binWidth/2.0,40,binWidth),patch_artist=True,widths=binWidth-0.75,medianprops = dict(color='C0'),capprops = dict(color='C0'),whiskerprops = dict(color='C0'),boxprops  = dict(color='C0',facecolor='C0',alpha=0.75),whis=(2.5,97.5),showfliers=False)
	
	ax_ratioREM.boxplot(ratioREMBinned,positions=np.arange(start+binWidth/2.0,40,binWidth),patch_artist=True,widths=binWidth-0.75,medianprops = dict(color='C1'),capprops = dict(color='C1'),whiskerprops = dict(color='C1'),boxprops  = dict(color='C1',facecolor='C1',alpha=0.75),whis=(2.5,97.5),showfliers=False)
	
	#set location of xticks
	ax_ratioWake.set_xticks(np.arange(10,41,5),np.arange(10,41,5))
	ax_ratioREM.set_xticks(np.arange(10,41,5),np.arange(10,41,5))
	
	#labels and minortick adjustments
	ax_ratioWake.minorticks_on()
	ax_ratioWake.tick_params(axis='y', which='minor', left=False)
	
	ax_ratioREM.minorticks_on()
	ax_ratioREM.tick_params(axis='y', which='minor', left=False)
	
	ax_ratioREM.set_xlabel("Peak Frequency\n of Oscillatory Band (Hz)")	
	

#plot ratio of burst rates in bands at the group level
def plotBurstRateRatioGroupLevel(band,ax):
	df_gamma=getSignificantBands(which=band) 
	
	pID=df_gamma['pID'].values
	pID[pID=='p14_followup']='p14'
	df_gamma['pID']=pID
	uniqPID=np.unique(pID)
	states=['wake','NREM','REM']
	burstRates_subj=np.zeros((len(uniqPID),len(states)))

	
	for iPID in range(0,len(uniqPID)):
		for iState in range(0,len(states)):
			burstRates_subj[iPID,iState]=np.mean(df_gamma['meanBurstRate_%s'%states[iState]][df_gamma['pID']==uniqPID[iPID]])
	#print(burstRates_subj)
	burstRates_subj=np.log10(burstRates_subj)
	colors=[sns.color_palette("deep")[0],sns.color_palette("deep")[2],sns.color_palette("deep")[1]]
	violin=sns.violinplot(burstRates_subj,palette=colors,ax=ax,cut=0,alpha=0.75,width=0.5)

	for i in range(0,len(uniqPID)):
		ax.plot(np.arange(3),(burstRates_subj[i]),c='gray',lw=1,marker='o',ms=1,zorder=-999)
	ax.set_ylabel("Burst Rate (/min)")
	ax.set_yticks([0,1],["1","10"])
	ax.set_yticks([0,1],["1","10"])
	ax.set_xticks([0,1,2],states)
	ax.set_ylim((-0.3,2))
	
				
#plot subject-level average aligned power spectra
def plotAlignedPSD(ax_osc,ax_spindle):

	#load aligned power spectra
	freqs,powerThalGamma=getAlignedPeriodicPower(getSignificantBands(which='gamma'),ch_name_sel='Thalamus')
	freqs,powerThalSpindle=getAlignedPeriodicPower(getSignificantBands(which='spindleInGammaChannels'),ch_name_sel='Thalamus')	
		
	states=['wake','REM','NREM']
	#normalize power spectra by standard deviation 
	#single standard deviation per subject 
	#to ensure the power spectra for different states stay on the same scale
	
	powerThalGamma=powerThalGamma/np.std(powerThalGamma,axis=(1,2),keepdims=True)
	powerThalSpindle=powerThalSpindle/np.std(powerThalSpindle,axis=(1,2),keepdims=True)
	
	##
	#fast oscillations
	##
	
	#get subject level mean and S.E.M
	nSubj=powerThalGamma.shape[0]
	print("Number of subjects:%d"%nSubj)
	mean=np.mean(powerThalGamma,axis=0)
	std=np.std(powerThalGamma,axis=0)/np.sqrt(nSubj)
	
	#plot mean and SEM for each state
	for iState in range(len(states)):
		ax_osc.plot(freqs,mean[iState],label=states[iState])
		ax_osc.fill_between(freqs,mean[iState]-std[iState],mean[iState]+std[iState],alpha=0.75)	
	
	
	ax_osc.text(0.02,0.90,r"N$_\mathrm{subjects}=%d$"%nSubj,transform=ax_osc.transAxes)

	

	ax_osc.set_ylabel("Power (a.u.)")
	ax_osc.set_xlabel("Frequency relative to fast oscillation (Hz)")
	ax_osc.set_ylim((-0.3,3.5))
	ax_osc.set_xlim((-6,6))	
	ax_osc.set_xticks([-5,0,5],[r"f$_\mathrm{osc}$-5 Hz", r"f$_\mathrm{osc}$",r"f$_\mathrm{osc}$+5 Hz"])
	ax_osc.axvline(0,ls='--',c='gray')
	ax_osc.minorticks_on()	
	
	##
	#spindles
	##
	
	mean=np.mean(powerThalSpindle,axis=0)
	std=np.std(powerThalSpindle,axis=0)/np.sqrt(nSubj)
	for iState in range(len(states)):
		ax_spindle.plot(freqs,mean[iState],label=states[iState])
		ax_spindle.fill_between(freqs,mean[iState]-std[iState],mean[iState]+std[iState],alpha=0.75)	
	leg=ax_spindle.legend(loc='upper right')
	for line in leg.get_lines():
		line.set_linewidth(4.0)
	ax_spindle.set_ylabel("Power (a.u.)")
	ax_spindle.set_xlabel("Frequency relative to spindle (Hz)")	
	ax_spindle.set_xlim((-6,6))
	ax_spindle.set_xticks([-5,0,5],[r"f$_\mathrm{spindle}$-5 Hz", r"f$_\mathrm{spindle}$",r"f$_\mathrm{spindle}$+5 Hz"])

	ax_spindle.axvline(0,ls='--',c='gray')
	ax_spindle.text(0.02,0.90,r"N$_\mathrm{subjects}=%d$"%nSubj,transform=ax_spindle.transAxes)
	ax_spindle.minorticks_on()

	
	
def plotFigure2(pID,ch_name,trange):



	

	#creating matplotlib panels for the plots
	fig = plt.figure(figsize=(14, 13))
	gs = fig.add_gridspec(10, 14, height_ratios = [3.5,1.0,1.8,0.4,1,1,1,1,1,1])
	fig.subplots_adjust(wspace=0.15,hspace=0.35)	
	
	#axes for example plots	
	ax_hyp=fig.add_subplot(gs[1, :11])
	ax_burstRate2D=fig.add_subplot(gs[0, :11])
	ax_burstRate1D=fig.add_subplot(gs[2, :11])	
	ax_burstRateFreq=fig.add_subplot(gs[0, 11:])	
	ax_burstRate2D.set_title("(A) Burst detection in example patient",loc='left',fontdict={'fontweight':'bold','fontsize':10})
	
	#axes for histograms
	ax_freqHistWake=fig.add_subplot(gs[4:6, :2])
	ax_freqHistREM=fig.add_subplot(gs[6:8, :2])						
	ax_freqHistNREM=fig.add_subplot(gs[8:10, :2])	
	ax_freqHistWake.set_title("(B) Frequency distri-\nbution across cohort",loc='left',fontdict={'fontweight':'bold','fontsize':10})
	
	#axes for the burst density ratios
	gs_sub=gs[4:, 3:7].subgridspec(2, 1, hspace=0.3)
	axs = gs_sub.subplots()
	ax_ratioWake=axs[0]
	ax_ratioREM=axs[1]														
	ax_ratioWake.set_title("(C) Burst rates across cohort",loc='left',fontdict={'fontweight':'bold','fontsize':10})	
	
	
	#axes for plotting group statistics
	gs_sub=gs[4:, 8:10].subgridspec(2, 1, hspace=0.3)
	axs = gs_sub.subplots()
	ax_subjBR_fosc=axs[0]
	ax_subjBR_spindle=axs[1]
	ax_subjBR_fosc.set_title("(D) Subject-level\nBurst Rates",loc='left',fontdict={'fontweight':'bold','fontsize':10})
	
	
	#axes for plotting aligned average spectra
	gs_sub=gs[4:, 11:].subgridspec(2, 1, hspace=0.3)
	axs = gs_sub.subplots()
	ax_avgPSD_fosc=axs[0]
	ax_avgPSD_spindle=axs[1]
	ax_avgPSD_fosc.set_title("(E) 1/f Subtracted\nAverage Power",loc='left',fontdict={'fontweight':'bold','fontsize':10})		
	
			
		
	
	#plotting example burst rate														
	im=plot2DBurstRate(pID=pID,
				ch_name=ch_name,
				trange=trange,
				ax=ax_burstRate2D)
	ax_burstRate2D.set_xlabel("")		
	ax_burstRate2D.minorticks_on()		
	ax_inset = ax_burstRate2D.inset_axes([0.04, 0.9, 0.15, 0.05])
	cb=plt.colorbar(im,cax=ax_inset, orientation='horizontal')
	cb.set_label('Burst Rate (/min)', color='white')
	cb.ax.xaxis.set_tick_params(color='white',labelcolor='white')
	cb.outline.set_edgecolor('white')
	
	
	#hyponogram for example 
	plotHypnoLocalTime(pID=pID,
				trange=trange,
				ax=ax_hyp)
	ax_hyp.set_xlabel("")	
	
	#burst rate as a function of frequency 
	plotBurstRateWithBands(pID=pID,
				ch_name=ch_name,
				ax=ax_burstRateFreq)
	ax_burstRateFreq.yaxis.tick_right()
	ax_burstRateFreq.yaxis.set_ticks_position('both')
	ax_burstRateFreq.yaxis.set_label_position("right")
	
	#plot 1D burst rate as a function of time
	df_freqs=getSignificantBands(which='all')
	df_selectedChannel=df_freqs[np.logical_and.reduce((df_freqs['pID']==pID,df_freqs['ch_name']==ch_name,df_freqs['uniqueSelection']==True))]	
	plot1DBurstRate(pID=pID,
			ch_name=ch_name,
			trange=trange,
			freqrange=np.column_stack((df_selectedChannel['freqLow'].values,df_selectedChannel['freqHigh'].values)),
			linecolors=['C1','C2'],ax=ax_burstRate1D)
	ax_burstRate1D.set_ylim((-0.1,4.5))
	ax_burstRate1D.set_xlabel("Local Time (hour)")		
	
	
	#draw histograms of all detected frequencies
	plotSummaryDetection(ax_Wake=ax_freqHistWake,
				ax_REM=ax_freqHistREM,
				ax_NREM=ax_freqHistNREM
				)	
	
	#plot burst ratio
	plotBurstRateRatio(ax_ratioWake=ax_ratioWake,ax_ratioREM=ax_ratioREM)	
	
	
	#plot burst ratio subject level
	plotBurstRateRatioGroupLevel('gamma',ax_subjBR_fosc)	
	plotBurstRateRatioGroupLevel('spindleInGammaChannels',ax_subjBR_spindle)	
	ax_subjBR_fosc.text(0.15,0.92,"fast oscillations",transform=ax_subjBR_fosc.transAxes)
	ax_subjBR_spindle.text(0.3,0.92,"spindles",transform=ax_subjBR_spindle.transAxes)

	
	ax_subjBR_spindle.text(0.02,0.05,r"N$_\mathrm{subjects}=14$",transform=ax_subjBR_spindle.transAxes,fontsize=8)	
	ax_subjBR_fosc.text(0.02,0.03,r"N$_\mathrm{subjects}=14$",transform=ax_subjBR_fosc.transAxes,fontsize=8)
	#plot average aligned PSD 
	plotAlignedPSD(ax_avgPSD_fosc,ax_avgPSD_spindle)
	
	plt.savefig("figures/figure2.pdf",bbox_inches='tight',dpi=600)


plotFigure2('p21','R1-R2',trange=[0.9,18.4])	
