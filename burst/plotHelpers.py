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
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure
from scipy.interpolate import interp1d

from core import *
from core.helpers import *
from coreFunctions import getBurstRate2D,getBootstrapErrorOnCurve,getSignificantBands
from psd.waveletTransform import compute_wavelet_transform


#extract small segment of morlet transform for illustration purposes
def extractMorletSegment(
			pID,		
			ch_name,
			tstart,		#start point
			duration,	#duration of cutout
			fmin=7,
			fmax=49,
			n_cycles=10,
			fs=200.0			
			):
			
	raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,ch_name[0])).pick(ch_name).load_data()
	
	#crop data
	raw.crop(tmin=tstart,tmax=tstart+duration)
	
	taxis=raw.times	
	freqs=np.arange(fmin,fmax,0.5)
	data=raw.get_data()[0]
	
	#compute wavelet transform for given segment of data
	mwt = compute_wavelet_transform(data, fs=raw.info['sfreq'], n_cycles=n_cycles, freqs=freqs)	
	np.save("outfiles/Example_mwt_%s_%s_%.1f_%.1f.npy"%(pID,ch_name,tstart,duration),mwt)
	np.save("outfiles/Example_taxis_%s_%s_%.1f_%.1f.npy"%(pID,ch_name,tstart,duration),taxis)	
	
#plot small segment of morlet transform with bursts overlaid for illustration purposes
def plotMorletSegment(ax,  		#matplotlib axes to draw on
			pID,		
			ch_name,
			tstart,		#start point
			duration,	#duration of cutout
			tmax,		#maximum time to plot
			fmin=7,
			fmax=49,
			n_cycles=10,
			fs=200.0,
			titletext=''
			,plotcbar=True,
			plotDetections=True	#overlay bursts that have been detected
			):
			

	#bufferTime is the extent on the edges which would have NaNs due to insufficient number of cycles available
	bufferTime=n_cycles/fmin
	freqs=np.arange(fmin,fmax,0.5)

	
	#load wavelet transform for the given segment of data
	mwt=np.load("outfiles/Example_mwt_%s_%s_%.1f_%.1f.npy"%(pID,ch_name,tstart,duration))
	taxis=np.load("outfiles/Example_taxis_%s_%s_%.1f_%.1f.npy"%(pID,ch_name,tstart,duration))	
	
	#plot morlet transformed data, exclude bufferTime on either sides
	im=ax.imshow(np.abs(mwt)*1e3,extent=(taxis[0]-bufferTime,taxis[-1]-bufferTime,freqs[0],freqs[-1]),origin='lower',aspect='auto',cmap='inferno')
	ax.set_xlim(0,tmax)
	ax.set_ylim(fmin+0.1,fmax-0.5)
	ax.set_xlabel("Time (seconds)")
	ax.set_ylabel("Frequency (Hz)")
	
	#plot title
	ax.text(0.1,fmax-4,titletext,color='white',fontsize=14,fontweight ='bold')

	#load and overlay bursts on the morlet spectrogram
	peakFreq,startFreq,stopFreq,peakTimeIndx,startTimeIndx,stopTimeIndx,peakAmp,sleepScoreAtBurst=np.loadtxt(rootdir+"/burstFromMorlet/%s_%s_selected.txt"%(pID,ch_name),unpack=True)
	peakTime,startTime,stopTime=peakTimeIndx/fs,startTimeIndx/fs,stopTimeIndx/fs
	selmask=np.logical_and(startTime>tstart,stopTime<tstart+duration)
	peakTime,startTime,stopTime,peakFreq,startFreq,stopFreq= peakTime[selmask],startTime[selmask],stopTime[selmask],peakFreq[selmask],startFreq[selmask],stopFreq[selmask]
	
	if(plotDetections):	
		ax.errorbar(peakTime-tstart-bufferTime,peakFreq,xerr=[peakTime-startTime,stopTime-peakTime],yerr=[peakFreq-startFreq,stopFreq-peakFreq],fmt='o',c='white', capsize=4, elinewidth=0.5,ms=1,lw=0.25)
	

	#plot colorbar
	if(plotcbar):
		plt.colorbar(im,ax=ax,label='Amplitude (a.u.)')
	ax.minorticks_on()	

	
	

		
# plot brain-state-averaged burst rate curves
def plotBurstRateWithBands(pID,
		ch_name,
		ax,		#matplotlib axes to plot on
		sfreq=200.0,
		smoothWindowInHz=2.0
		):
	#get 2D burst rate
	taxis,freqs,burstRate,sleepScore=getBurstRate2D(pID=pID,
							ch_name=ch_name,
							sfreq=sfreq,
							smoothWindowInHz=smoothWindowInHz)
	#get significant bands for the particular channel
	df_freqs=getSignificantBands()	
	df_freqs=df_freqs[np.logical_and(df_freqs['pID']==pID,df_freqs['ch_name']==ch_name)].reset_index(drop=True)

	#plot frequency averaged
	for istate,state in zip(np.arange(3),['wake','REM','NREM']):
		if(state=='wake'):
			sleepmask=sleepScore==0  
		elif(state=='REM'):
			sleepmask=sleepScore==5
		elif(state=='NREM'):
			sleepmask=np.logical_or(sleepScore==2,sleepScore==3)	  


		mean,std,low,high=getBootstrapErrorOnCurve(burstRate[:,sleepmask])
		
		#select bands in current brain state
		dfThisState=df_freqs[df_freqs['state']==state].reset_index()
		
		#draw markers for significant bands
		for i in range(0,len(dfThisState)):
			ax.plot([-0.2-.3*istate,-0.2-0.3*istate],[dfThisState.loc[i,'freqLow'],dfThisState.loc[i,'freqHigh']],c='C%d'%istate,marker='_')
		
		#draw mean curves with errorbands
		ax.plot(mean,freqs,label=state,c='C%d'%istate)
		ax.fill_betweenx(freqs,low,high,fc='C%d'%istate,alpha=0.5)
	
	ax.set_ylim((freqs[0],freqs[-1]))
	ax.legend()
	ax.minorticks_on()	
	ax.set_ylabel("Frequency (Hz)")
	ax.set_xlabel("Burst rate (/min)")

#plot hypnogram with actual recording timestamp
def plotHypnoLocalTime(pID,
			trange, #time range to plot
			ax	#matplotlib axes to plot on
			):
	#load timestamp

	if(pID=='p21'):
		tstart=47321.0
	else:
		print("Error: Hard coded version, can only be run for p21")
		sys.exit()
	#raw=mne.io.read_raw(rawfileName(pID),preload=False,verbose=False)	
	#tstart=(raw.info['meas_date'].hour*3600.+raw.info['meas_date'].minute*60.0+raw.info['meas_date'].second)
	
	
	#load sleepscore
	taxis,sleepScore=readSleepScoreFinal(pID)
	taxis+=tstart
	#plot hyponogram
	plotStandardHypnogram(taxis=taxis,sleepscore=sleepScore,ax=ax,combineNREM=True)
	
	#relabel xticks to 24h labels
	xticks=np.arange(np.round(tstart/3600.+trange[0]),np.round(tstart/3600.+trange[-1]),2)
	labels=[]
	for i in range(0,len(xticks)):
		labels.append('%02d'%(xticks[i]%24))
	ax.set_xticks(xticks,labels)	
	
	ax.minorticks_on()
	ax.tick_params(axis='y', which='minor', left=False)
	
	ax.set_xlim(trange[0]+tstart/3600.,trange[-1]+tstart/3600.)
	ax.set_ylim((0,5))
	
	ax.set_xlabel("Local Time (hour)")

#plot burst rate as a function of time for given frequency bands
def plot1DBurstRate(pID,
		ch_name,
		trange, 	#time ranges
		freqrange,	#frequency bands
		linecolors,	#line colours to use
		ax,		#matplotlib axes to draw on
		sfreq=200.0):
	#load timestamp information
	#raw=mne.io.read_raw(rawfileName(pID),preload=False,verbose=False)
	#tstart=(raw.info['meas_date'].hour*3600.+raw.info['meas_date'].minute*60.0+raw.info['meas_date'].second)
	
	if(pID=='p21'):
		tstart=47321.0
	else:
		print("Error: Hard coded version, can only be run for p21")
		sys.exit()
		
		
	#load 2D burst rate
	taxis,freqs,burstRate,sleepScore=getBurstRate2D(pID=pID,ch_name=ch_name,sfreq=sfreq,smoothWindowInHz=None)
	taxis+=tstart
	
	#calculate and plot normalized burst rates for each band
	for ifreq in range(len(freqrange)):
		rate=np.mean(burstRate[np.logical_and(freqs>=freqrange[ifreq,0],freqs<=freqrange[ifreq,1])],axis=0)
		rate/=np.mean(rate)
		ax.plot(taxis/3600.,rate,c=linecolors[ifreq],label='%.1f - %.1f Hz'%(freqrange[ifreq,0],freqrange[ifreq,1]))
	
	ax.legend(ncols=2)
	
	ax.set_ylabel("normalized rate (/min)")	 

	
	#set x axes limit and relabel to put 24h time ticks
	ax.set_xlim(trange[0]+tstart/3600.,trange[-1]+tstart/3600.)	
	xticks=np.arange(np.round(tstart/3600.+trange[0]),np.round(tstart/3600.+trange[-1]),2)
	labels=[]
	for i in range(0,len(xticks)):
		labels.append('%02d'%(xticks[i]%24))
	ax.set_xticks(xticks,labels)	
	
	ax.minorticks_on()

	
#plot single channel 2D burst rate		
def plot2DBurstRate(pID,
		ch_name,
		trange,		 	#time range to plot
		ax,			#matplotlib axes to plot on
		sfreq=200.0,		#sampling frequency
		smoothWindowInHz=2.0 	#frequency smoothing
		):
	
	
	#get 2D burst rate
	taxis,freqs,burstRate,sleepScore=getBurstRate2D(pID=pID,
							ch_name=ch_name,
							sfreq=sfreq,
							smoothWindowInHz=smoothWindowInHz
							)
	#set start of recording from the raw file, for plot in paper	
	if(pID=='p21'):			
		#raw=mne.io.read_raw(rawfileName(pID),preload=False,verbose=False)
		#tstart=(raw.info['meas_date'].hour*3600.+raw.info['meas_date'].minute*60.0+raw.info['meas_date'].second)
		tstart=47321.0
		#print(tstart)
		taxis+=tstart
	
	#plot burst rate
	im=ax.imshow(burstRate,origin='lower',extent=(taxis[0]/3600.,(taxis[-1]+30.)/3600.,freqs[0],freqs[-1]),aspect='auto',cmap='inferno',vmax=10)
	ax.set_ylabel("Frequency (Hz)")	 
	
	#set x axes limit and relabel to put 24h time ticks, for plot in paper
	if(pID=='p21'):	
		ax.set_xlim(trange[0]+tstart/3600.,trange[-1]+tstart/3600.)
		
		xticks=np.arange(np.round(tstart/3600.+trange[0]),np.round(tstart/3600.+trange[-1]),2)
		labels=[]
		for i in range(0,len(xticks)):
			labels.append('%02d'%(xticks[i]%24))
		ax.set_xticks(xticks,labels)	
	
	ax.minorticks_on()
	return im
