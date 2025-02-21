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


from core import *
from core.helpers import *
from burst.coreFunctions import getBurstRate2D,getOverlaps,getMeanBurstDensity
from scipy.stats import spearmanr,binned_statistic

#add burst densities to the dataframe
def addBurstDensity(df_freqs):
	uniqPID=np.unique(df_freqs['pID'].values)
	uniqCh=np.unique(df_freqs['ch_name'].values)
	
	peakBurstRateInBand,burstRateInBand=np.zeros(len(df_freqs)),np.zeros(len(df_freqs))
	burstRateStates={'wake':np.zeros(len(df_freqs)),'REM':np.zeros(len(df_freqs)),'NREM':np.zeros(len(df_freqs))}	
	
	#loop over all subjects and channels
	for i in range(0,len(uniqPID)):
		for ch_name in uniqCh:
			selmask=np.logical_and(df_freqs['pID']==uniqPID[i],df_freqs['ch_name']==ch_name)
			if(np.sum(selmask)==0):
				continue
			#select the bands that have been detected in the channel
			df_this=df_freqs[selmask].reset_index(drop=True)
			
			#call function to get the burst densities
			peakBurstRateInBand[selmask],burstRateInBand[selmask],burstRateStates_= getMeanBurstDensity(uniqPID[i],ch_name,df_this['state'].values,df_this['freqLow'].values,df_this['freqHigh'].values)
			burstRateStates['wake'][selmask]=burstRateStates_['wake']
			burstRateStates['NREM'][selmask]=burstRateStates_['NREM']
			burstRateStates['REM'][selmask]=burstRateStates_['REM']
	
	#add all rates to dataframe		
	df_freqs['peakBurstRate']=peakBurstRateInBand
	df_freqs['meanBurstRate']=burstRateInBand
	for state in ['wake','REM','NREM']:
		df_freqs['meanBurstRate_%s'%state]=burstRateStates[state]
	return df_freqs
	
	

#function to check if burst rates of candidate harmonic bands are correlated
def checkForHarmonic(pID,
		ch_name, 
		freqsLow,  #low edge of frequency bands, list with [candidate main, candidate harmonic]
		freqsHigh, #high edge of frequency bands, list with [candidate main, candidate harmonic]
		sleepscoreSelect, #sleep score of bands
		sfreq=200.0,	#sampling frequency
		pvalThresh=1e-3	#pvalue threshold for correlation
		):
	#load burst timestamps
	peakFreq,startFreq,stopFreq,peakTimeIndx,startTimeIndx,stopTimeIndx,peakAmp,sleepScoreAtBurst=np.loadtxt(rootdir+"/burstFromMorlet/%s_%s_selected.txt"%(pID,ch_name),unpack=True)
	#load sleep score
	timeSS,sleepScore=readSleepScoreFinal(pID)
	ssFunc = scipy.interpolate.interp1d(timeSS,sleepScore, kind='nearest')	
	
	#select candidate 
	selmaskMain=np.logical_and.reduce((peakFreq<=freqsHigh[0],peakFreq>=freqsLow[0]))
	selmaskHarmonic=np.logical_and.reduce((peakFreq<=freqsHigh[1],peakFreq>=freqsLow[1]))
	burstIndxMain=peakTimeIndx[selmaskMain]
	burstIndxHarmonic=peakTimeIndx[selmaskHarmonic]	
	
	#get burst rates in 30second time windows
	tBins=np.arange(timeSS[0],timeSS[-1],30)
	#print(pID,ch_name,freqsLow,freqsHigh,sleepscoreSelect)
	rateMain,t=np.histogram(burstIndxMain/sfreq,bins=tBins)
	rateHarmonic,t=np.histogram(burstIndxHarmonic/sfreq,bins=tBins)
	tBins=(tBins[:-1]+tBins[1:])/2
	
	#select bins at specified brain state
	sleepmask=np.logical_or(ssFunc(tBins)==sleepscoreSelect[0],ssFunc(tBins)==sleepscoreSelect[1])
	rateMain=rateMain[sleepmask]
	rateHarmonic=rateHarmonic[sleepmask]
	
	#spearman r 
	resRate=spearmanr(rateMain,rateHarmonic)
	
	#if positively correlated return true
	if(resRate.statistic>0 and resRate.pvalue<pvalThresh):
		return True
	else:
		return False


#function to detect if a band is at harmonic frequency of another band detected
def getHarmonicMask(df_freqs):
	
	freqLowAll=df_freqs['freqLow'].values
	freqHighAll=df_freqs['freqHigh'].values

	sleepscoreSelect={'wake':[0,0],'REM':[5,5],'NREM':[2,3]}
	selmaskHarmonic=np.zeros(len(df_freqs),dtype=bool)
	
	iInitHarmonic=0
	#loop over all bands
	
	for i in range(0,len(df_freqs)):
		indx=df_freqs.index[i]
		#get information for current band
		stateThis=df_freqs.loc[indx,'state']
		chThis=df_freqs.loc[indx,'ch_name']
		pIDThis=df_freqs.loc[indx,'pID']
		freqThis=df_freqs.loc[indx,'freqPeak']
		freqLowThis=df_freqs.loc[indx,'freqLow']
		freqHighThis=df_freqs.loc[indx,'freqHigh']
		peakBurstRateThis=df_freqs.loc[indx,'peakBurstRate']
		
		#mask of likely main frequencies of primary harmonic
		likelyMainFrequencies=np.logical_and.reduce((df_freqs['pID']==pIDThis,	 #same patient
							df_freqs['ch_name']==chThis,	#same channel
							df_freqs['state']==stateThis,	#same brain state
							np.abs(freqThis/2.0-df_freqs['freqPeak'].values)<=1.0, 	#at half the frequency (within 1 Hz error)
							peakBurstRateThis<df_freqs['peakBurstRate'])		#with higher burst rate
							)	
							
		#mask of likely main frequencies of secondary harmonic						
		likelyMainFrequenciesForThirdHarmonic=np.logical_and.reduce((df_freqs['pID']==pIDThis,	#same patient
							df_freqs['ch_name']==chThis,			#same channel	
							df_freqs['state']==stateThis,			#same brain state
							np.abs(freqThis/3.0-df_freqs['freqPeak'].values)<=1.0,	#at 1/3rd the frequency (within 1 Hz error)
							peakBurstRateThis<df_freqs['peakBurstRate'])		#with higher burst rate
							)
							
		#mask of likely frequencies of primary harmonic, for cases where the current one is a secondary harmonic (additional safety net)													
		likelyPrimaryHarmonicForThirdHarmonic=np.logical_and.reduce((df_freqs['pID']==pIDThis,	#same patient
							df_freqs['ch_name']==chThis,			#same channel
							df_freqs['state']==stateThis,			#same brain state
							np.abs(2*freqThis/3.0-df_freqs['freqPeak'].values)<=1.0, #at 2/3rd the frequency (within 1 Hz error)
							peakBurstRateThis<df_freqs['peakBurstRate'])	#with higher burst rate
							)
		
		
		#if there is one like likely main frequency				
		isPrimaryHarmonic=np.sum(likelyMainFrequencies)==1
		#if there is both one likely main frequency and one likely primary harmonic; added insurance to false detections.				
		isSecondaryHarmonic=np.sum(likelyMainFrequenciesForThirdHarmonic)==1 and np.sum(likelyPrimaryHarmonicForThirdHarmonic)==1
		if(isPrimaryHarmonic or isSecondaryHarmonic):
			if(isPrimaryHarmonic):
				freqLowCandidate=freqLowAll[likelyMainFrequencies][0]
				freqHighCandidate=freqHighAll[likelyMainFrequencies][0]
			else:
				freqLowCandidate=freqLowAll[likelyMainFrequenciesForThirdHarmonic][0]
				freqHighCandidate=freqHighAll[likelyMainFrequenciesForThirdHarmonic][0]
				print(pIDThis,chThis)
			iInitHarmonic+=1

			#check if burst rates correlate
			selmaskHarmonic[i]=checkForHarmonic(pIDThis,
					chThis,
					freqsLow=[freqLowCandidate,freqLowThis],
					freqsHigh=[freqHighCandidate,freqHighThis],
					sleepscoreSelect=sleepscoreSelect[stateThis])
			
			
				
	print("Number of harmonic bands initially detected:%d"%iInitHarmonic)
	print("Number of harmonic bands detected with correlated burst rates:%d"%np.sum(selmaskHarmonic))
	return selmaskHarmonic

	
def getSelection(infile,outfile,maxWidth=30,pvalThresh=7e-5):
	

	df_freqs=pd.read_csv(infile,sep=' ')
	selmask=np.ones(len(df_freqs),dtype=bool)	
	
	#removing left electrodes of p26; electrode not recorded; channels contain junk
	selmask[np.logical_and(df_freqs['pID']=='p26',df_freqs['ch_name']=='L1-L2')]=False
	selmask[np.logical_and(df_freqs['pID']=='p26',df_freqs['ch_name']=='L2-L3')]=False
	selmask[np.logical_and(df_freqs['pID']=='p26',df_freqs['ch_name']=='L3-L4')]=False	
	df_freqs=df_freqs[selmask]
	print("Number of bands before significance test: %d"%len(df_freqs))	
	
		
	#select significant bands
	selmask=df_freqs['pvaluesBoth']<pvalThresh		
	df_freqs=df_freqs[selmask]
	print("Number of significant bands: %d"%len(df_freqs))	
	
	#add burst density to dataframe
	df_freqs=addBurstDensity(df_freqs)	
		
	#remove bands likely to be harmonics
	selmaskHarmonic=getHarmonicMask(df_freqs)	
	print("List of harmonic bands:")
	print(df_freqs[selmaskHarmonic][["pID","ch_name","state","freqLow","freqHigh"]])
	df_freqs=df_freqs[np.logical_not(selmaskHarmonic)].reset_index(drop=True)
	print("Number of bands after harmonic rejection: %d"%len(df_freqs))	
		
	
		
	df_freqs.to_csv(outfile,sep=' ',index=False)
	
#uncomment the following lines to get list of significant bands					
#getSelection('outfiles/bursts_detectedFrequencies.txt',"outfiles/bursts_detectedFrequencies_selected.txt",pvalThresh=7e-5)

#getSelection('outfiles/bursts_detectedFrequencies_scalp.txt',"outfiles/bursts_detectedFrequencies_scalp_selected.txt",pvalThresh=7e-5)
