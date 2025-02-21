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
import matplotlib.gridspec as gridspec


from core import *
from core.helpers import *
from burst.coreFunctions import *
from coreFunctions import *
from psd.waveletTransform import compute_wavelet_transform


import matplotlib.gridspec as gridspec

import matplotlib

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 

#bootstrap to standard deviation on correlation coefficients
def crossCorrShuffle(saccadeAutoCorr,eventHist,nShuffle=int(1e4)):
	notNanMask=np.logical_not(np.isnan(saccadeAutoCorr))
	xaxis=np.arange(len(saccadeAutoCorr))
	saccadeAutoCorr=np.interp(xaxis,xaxis[notNanMask],saccadeAutoCorr[notNanMask])
	eventHist_zscored=(eventHist-np.mean(eventHist))/np.std(eventHist)
	saccadeAutoCorr_zscored=(saccadeAutoCorr-np.mean(saccadeAutoCorr))/np.std(saccadeAutoCorr)
	
	trueCorr=np.mean(eventHist_zscored*saccadeAutoCorr_zscored)
	shuffledDist=np.zeros(nShuffle)
	nSamp=len(eventHist_zscored)
	for i in range(0,nShuffle):
		sel=np.random.randint(0,nSamp-1,nSamp)	
		eventHistThis=eventHist_zscored[sel]
		saccadeAutoCorrThis=saccadeAutoCorr_zscored[sel]
		eventHistThis=(eventHistThis-eventHistThis.mean())/eventHistThis.std()
		saccadeAutoCorrThis=(saccadeAutoCorrThis-saccadeAutoCorrThis.mean())/saccadeAutoCorrThis.std()		
		shuffledDist[i]=np.mean(eventHistThis*saccadeAutoCorrThis)
	return trueCorr,np.std(shuffledDist)


#function to compute morlet transform for small segments and write to file
#used to plot examples
def extractExampleTrace(
			pID,		
			ch_name,
			tstart,		#start point
			duration,	#duration of cutout
			fmin=7,
			fmax=49,
			n_cycles=10,
			fs=200.0,
			):
	
	raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_%s_eeg.fif'%(pID,pID,ch_name[0])).pick(ch_name).load_data()
	taxisOriginal=raw.times	
	#crop data
	raw.crop(tmin=tstart,tmax=tstart+duration)
	
	taxis=raw.times	
	freqs=np.arange(fmin,fmax,0.5)
	data=raw.get_data()[0]
	
	badMask= getBADMask(pID,ch_name[0],combineChannels=False)[int(ch_name[1])-1]
	badMask=badMask[np.logical_and(taxisOriginal>=tstart,taxisOriginal<=tstart+duration)]
	data=raw.get_data()[0]
	data[badMask]=np.nan
	
	
	#compute wavelet transform for given segment of data
	mwt = compute_wavelet_transform(data, fs=raw.info['sfreq'], n_cycles=n_cycles, freqs=freqs)	
	
	
	mwt=np.abs(mwt)
	mwt=mwt/np.nanmedian(mwt)
	np.save("outfiles/Example_mwt_%s_%s.npy"%(pID,ch_name),mwt)
	
	
	raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_F7-F8_eeg.fif'%(pID,pID)).crop(tmin=tstart,tmax=tstart+duration).load_data()
	#raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_scalp_eeg.fif'%(pID,pID)).crop(tmin=tstart,tmax=tstart+duration).load_data()

	d=raw.get_data('F7-F8')[0]
	
	np.save("outfiles/Example_F7F8_%s.npy"%(pID),d)
	
	np.save("outfiles/Example_taxis_%s.npy"%(pID),raw.times)
#plot small sengment of morlet transform for illustration purposes

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
			):
	
	
	#bufferTime is the extent on the edges which would have NaNs due to insufficient number of cycles available
	bufferTime=n_cycles/fmin
	
	taxis=np.load("outfiles/Example_taxis_%s.npy"%(pID))	
	mwt=np.load("outfiles/Example_mwt_%s_%s.npy"%(pID,ch_name))
	freqs=np.arange(fmin,fmax,0.5)
	
	maxval=np.nanmax(mwt)	
	im=ax.imshow(mwt,extent=(taxis[0]-bufferTime,taxis[-1]-bufferTime,freqs[0],freqs[-1]),origin='lower',aspect='auto',cmap='inferno')

	ax.set_ylim(fmin+0.1,fmax-0.5)
	ax.set_xlabel("Time (seconds)")
	ax.set_ylabel("Frequency (Hz)")
	ax.text(0.1,fmax-4,titletext,color='white',fontsize=14,fontweight ='bold')
	ax.minorticks_on()
	if(plotcbar):
		plt.colorbar(im,ax=ax,label='Amplitude (a.u.)')
	return bufferTime,im,maxval
	
#function to plot example traces	
def plotExampleTrace(pID,ch_name,tstart,duration,tmax,ax,title='',fs=200.0):
	eyeMovParams=pd.read_csv(rootdir+"/eyeMovParams/%s_eyeMovEvents_alldetections.csv"%pID)
	selmask=eyeMovParams['sleepScore'].values.astype("int")==5
	tonIndx=eyeMovParams['peakVelocityIndex'].values[selmask]
	ton=tonIndx/fs
	mask=np.logical_and(ton>tstart,ton<=tstart+duration)
	tonIndx=tonIndx[mask]
	ton=ton[mask]-tstart


	ax[0].set_title(title,loc='left',fontdict={'fontweight':'bold','fontsize':10})
	
	#plot morlet segment
	bufferTime,im,maxval=plotMorletSegment(ax=ax[0],pID=pID,ch_name=ch_name,
							tstart=tstart,duration=duration,tmax=tmax,
							titletext='',plotcbar=False)	
	
	#draw colorbar
	cb=plt.colorbar(im,cax=ax[2],location='left',fraction=0.05)		
	cb.set_label(label='amplitude (a.u.)',size=8)
	cb.set_ticks(np.arange(0,maxval,1))
	cb.ax.tick_params(labelsize=8)

	#load F7-F8 traces
	#raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_F7-F8_eeg.fif'%(pID,pID)).crop(tmin=tstart,tmax=tstart+duration).load_data()
	#raw=mne.io.read_raw(rootdir+'/data/rereferenced/%s/raw_%s_electrode_scalp_eeg.fif'%(pID,pID)).crop(tmin=tstart,tmax=tstart+duration).load_data()

	d=np.load("outfiles/Example_F7F8_%s.npy"%(pID))
	taxis=np.load("outfiles/Example_taxis_%s.npy"%(pID))	
	#raw.get_data('F7-F8')[0]#-raw.get_data('F8')[0]	
	
	#mark eye movement events
	for t in ton:
		ax[1].axvline(t-bufferTime,ls='-',c='gray',lw=0.5)		

	ax[1].plot(taxis-bufferTime,d*1e6)
	ax[1].set_xlabel("Time (sec)")
	ax[1].set_ylabel("F7-F8 (microV)")	
	ax[1].set_xlim((0,tmax))
	ax[0].set_xlim((0,tmax))	
	

#plot example peri event histogram
def plotPeriEventExample(pID,
			ch_name,
			ax	#matplotlib axes to draw on
			):
			
	#figure out the correct location of pID and channel
	df_detections=getSignificantBands(which='gamma')
	selmask=np.logical_and(df_detections['pID']==pID,df_detections['ch_name']==ch_name)
	df_sel=df_detections[selmask]
	
	

	#compute perievent histogram
	delayT,eventHist,saccadeAutoCorr,nREMs,nBurst=getPeriEventHist(pID=pID,ch_name=ch_name,
					maxDelayInSec=60.0,smoothScaleInMs=1000,
					freqLowBand=df_sel['freqLow'].values[0],freqHighBand=df_sel['freqHigh'].values[0])
	
	#get bootstrapped errors for displaying in panel
	notNanMask=np.logical_not(np.isnan(saccadeAutoCorr))
	trueCoeff,stdCoeff=crossCorrShuffle(saccadeAutoCorr,eventHist)
	
	saccadeAutoCorr[np.abs(delayT)<=1]=np.nan
	ax.plot(delayT,(eventHist),zorder=9999,lw=3,c='C1')

	ax.text(0.02,0.85,r"N$_\mathrm{EM}$=%d"%nREMs+"\n"+ r"r=$%.2f\pm%.2f$"%(trueCoeff,stdCoeff),transform=ax.transAxes,fontsize=8)	
	ax.axvline(0,ls='--',c='black',lw=0.5)			

	#plot saccade autocorrelation in a twin axes
	ax2=ax.twinx()
	ax2.plot(delayT,saccadeAutoCorr,c='gray',lw=2)							
	ax.set_ylabel(r"Burst Probability ($\%$)")
	ax2.set_ylabel("rapid EM Probability")	
	ax2.spines["right"].set_edgecolor('gray')
	ax2.yaxis.label.set_color('gray')
	ax.set_xlabel("Time relative rapid EM (sec)")
	ax.set_zorder(999) 
	ax.patch.set_visible(False)
	ax.set_xlim((delayT[0],delayT[-1]))
	
	
#plot statistics summary
def plotStatsSummary(ax,state='REM',pvalThresh=9e-4):
	df_REM=pd.read_csv("./outfiles/%s_burstSaccadeCrossCorr.txt"%state,sep=' ')

       
	
	#load variables and sort by frequency
	pID=df_REM['pID'].values	
	ch_name=df_REM['ch_name'].values		
	uniqPID=np.unique(pID)	
	freqPeak=df_REM['freqPeak'].values
	sortIndx=np.argsort(freqPeak)	
	pval=df_REM['crossCorrCoeff_pvalue'].values[sortIndx]
	coeff=df_REM['crossCorrCoeff'].values[sortIndx]		
	shuffleLow=df_REM['crossCorrCoeffShuffle_low'].values[sortIndx]
	shuffleHigh=df_REM['crossCorrCoeffShuffle_high'].values[sortIndx]
	pID=pID[sortIndx]
	ch_name=ch_name[sortIndx]	
	freqPeak=freqPeak[sortIndx]
	pID_ch_name=pID+ch_name
	uniqPID=np.unique(pID[pval<=pvalThresh])	

	index=np.arange(len(freqPeak))+1
	#plot range of null values
	ax.errorbar(index,coeff*0,yerr=[np.abs(shuffleLow),shuffleHigh],marker=None,ms=5,alpha=0.5)	
	ax.axhline(0,ls='--',c='black')
	
	#plot significant bands in different colour

	significantMask=pval<=pvalThresh
	print("Number of significant bands:%d/%d"%(np.sum(significantMask),len(significantMask)))
	print("Number of contacts:%d/%d"%(len(np.unique(pID_ch_name[significantMask])),len(np.unique(pID_ch_name))))
	ax.errorbar(index,coeff,fmt='o',ms=5,alpha=0.5)		
	ax.errorbar(index[significantMask],coeff[significantMask],fmt='o',c='C2',ms=5)	
	ax.set_xlim(0.1,index[-1]+0.9)
	ax.set_xlabel("Contact #")
	ax.set_ylabel("correlation coefficient")


def plotAveragePeriEvent(ax,state='REM',pvalThresh=9e-4):
	
	#load pvalues and select significant bands
	df_REM=pd.read_csv("./outfiles/%s_burstSaccadeCrossCorr.txt"%state,sep=' ')

	
	
	selmask=df_REM['crossCorrCoeff_pvalue']<pvalThresh
	df_REM=df_REM[selmask]
	
	#load peri event histograms
	periEventAll=np.load("outfiles/periEventAll_%s.npy"%state)
	saccadeAutoCorrAll=np.load("outfiles/saccadeAutoCorrAll_%s.npy"%state)	
	
	delay=periEventAll[-1]
	periEventAll=periEventAll[:-1][selmask]
	saccadeAutoCorrAll=saccadeAutoCorrAll[:-1][selmask]	
	print("Dimensions of perievent array:",periEventAll.shape)
	
	#get subject level averages
	pID=df_REM['pID'].values
	pID[pID=='p14_followup']='p14'
	uniqPID=np.unique(pID)
	
	periEventSubj=np.zeros((len(uniqPID),len(delay)))
	saccadeAutoCorrSubj=np.zeros((len(uniqPID),len(delay)))	
	for i in range(0,len(uniqPID)):
		periEventSubj[i]=np.mean(periEventAll[pID==uniqPID[i]],axis=0)
		saccadeAutoCorrSubj[i]=np.mean(saccadeAutoCorrAll[pID==uniqPID[i]],axis=0)		

	#correct baseline
	saccadeAutoCorrSubj=100*(saccadeAutoCorrSubj/np.median(saccadeAutoCorrSubj[:,np.abs(delay)>15.0],axis=1,keepdims=True)-1.0)
	periEventSubj=100*(periEventSubj/np.median(periEventSubj[:,np.abs(delay)>15.0],axis=1,keepdims=True)-1.0)	
	

	
	#get means and S.E.Ms
	nsubj=len(periEventSubj)
	meanPeriEvent=np.mean(periEventSubj,axis=0)	
	semPeriEvent=np.std(periEventSubj,axis=0)/np.sqrt(nsubj)
	
	meanAuto=np.mean(saccadeAutoCorrSubj,axis=0)	
	semAuto=np.std(saccadeAutoCorrSubj,axis=0)/np.sqrt(nsubj)
	
	
	notNanMask=np.logical_not(np.isnan(meanAuto))
	trueCoeff,stdCoeff=crossCorrShuffle(meanAuto,meanPeriEvent)
	#print(trueCoeff,stdCoeff)
	
	#plot mean and SEMs
	
	ax.plot(delay,meanPeriEvent,lw=3,c='C3')
	ax.fill_between(delay,meanPeriEvent-semPeriEvent,meanPeriEvent+semPeriEvent,fc='C3',alpha=0.5)		
	ax.axvline(0,ls='--',c='black',lw=0.5)			
	
	
	ax2=ax.twinx()
	ax.text(0.02,0.80,r"N$_\mathrm{subjects}$=%d"%nsubj+"\n"+ r"r=$%.2f\pm%.2f$"%(trueCoeff,stdCoeff),fontsize=12,transform=ax.transAxes)	
	ax2.plot(delay,meanAuto,c='gray',lw=2)	
	ax2.fill_between(delay,meanAuto-semAuto,meanAuto+semAuto,fc='gray',alpha=0.5)	
	ax.set_ylabel(r"Burst Probability (relative baseline, in $\%$)")
	ax2.set_ylabel(r"Rapid EM Probability (relative Baseline, in $\%$)")
	
	ax2.spines["right"].set_edgecolor('gray')
	ax2.yaxis.label.set_color('gray')
	ax.set_xlabel("Time relative rapid EM (sec)")
	ax.set_zorder(999) 
	ax.patch.set_visible(False)
	ax.set_xlim((-60,60))
	ax.minorticks_on()
	ax2.minorticks_on()

	#plt.show()	
	
		
def plotFigure3():	
		
	fig = plt.figure(figsize=(14, 12))
	gs = gridspec.GridSpec(5, 18, height_ratios = [0.7,0.7,1,0.01,1.2])
	fig.subplots_adjust(wspace=6.0,hspace=0.35)	
	
	
	#plot example-1
	ax_morlet_example1=fig.add_subplot(gs[0, 0:5])
	ax_EOG_example1=fig.add_subplot(gs[1, 0:5])
	ax_perievent_example1=fig.add_subplot(gs[2, 0:5])
	ax_buff1=fig.add_subplot(gs[0, 5])
	ax_morlet_example1.set_ylabel("Frequency (Hz)")		
	plotExampleTrace('p21','L2-L3',52360+815+4,30,27,[ax_morlet_example1,ax_EOG_example1,ax_buff1],title='(A) Example pp-1')
	plotPeriEventExample('p21','L2-L3',ax_perievent_example1)
	
	#plot example-2	
	ax_morlet_example2=fig.add_subplot(gs[0, 6:11])
	ax_EOG_example2=fig.add_subplot(gs[1, 6:11])
	ax_perievent_example2=fig.add_subplot(gs[2, 6:11])	
	ax_buff2=fig.add_subplot(gs[0, 11])
	plotExampleTrace('p03','R3-R4',58840+432,30,25,[ax_morlet_example2,ax_EOG_example2,ax_buff2],title='(B) Example pp-2')	
	plotPeriEventExample('p03','R3-R4',ax_perievent_example2)
	
	#plot example-3		
	ax_morlet_example3=fig.add_subplot(gs[0, 12:17])
	ax_EOG_example3=fig.add_subplot(gs[1, 12:17])
	ax_perievent_example3=fig.add_subplot(gs[2, 12:17])		
	ax_buff3=fig.add_subplot(gs[0, 17])
	plotExampleTrace('pthal103','R3-R4',87500+140,24,20,[ax_morlet_example3,ax_EOG_example3,ax_buff3],title='(C) Example pp-3')
	plotPeriEventExample('pthal103','R3-R4',ax_perievent_example3)
	


	#plot summary of statistics
	ax_stats_REM=fig.add_subplot(gs[4, 0:8])	
	ax_stats_REM.set_title("(D) Correlation between burst and rapid eye movements",loc='left',fontdict={'fontweight':'bold','fontsize':10})
	plotStatsSummary(ax_stats_REM,state='REM')	
	
	
	#plot subject level curves
	ax_avgperi_REM=fig.add_subplot(gs[4, 9:17])
	ax_avgperi_REM.set_title("(E) Average subject-level probabilities (REM)",loc='left',fontdict={'fontweight':'bold','fontsize':10})	
	plotAveragePeriEvent(ax_avgperi_REM,state='REM')	
	
	plt.savefig("figures/figure3.png",bbox_inches='tight',dpi=600)
	


plotFigure3()

'''
NOTE: This section requires access to full raw data
extractExampleTrace('p03','R3-R4',58840+432,30)
extractExampleTrace('p21','L2-L3',52360+815+4,30)
extractExampleTrace('pthal103','R3-R4',87500+140,24)
'''
