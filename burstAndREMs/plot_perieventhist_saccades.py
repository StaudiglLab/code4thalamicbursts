import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.backends.backend_pdf import PdfPages

from core import *
from core.helpers import *
from burst.coreFunctions import *	

from coreFunctions import getPeriEventHist
from numba import njit
import pandas as pd


def plotBurstProbability(sleepscoreSelect,outname,doIEDMasking=True,maxDelayInSec=30.0,plotCorrected=False):
	
	pdf=PdfPages(outname)

	df_selected=getSignificantBands('gamma')

	
	
	ch_names=['L1-L2','L2-L3','L3-L4','R1-R2','R2-R3','R3-R4']
	for pID in cohortForGammaStudy:
		fig = plt.figure(layout='constrained', figsize=(12,12))
		fig.suptitle("%s"%(pID))
		axs=fig.subplots(nrows=3,ncols=2,sharex=True, sharey=False)
		
		
		for ich in range(len(ch_names)):
			ax=axs[ich%3,ich//3]	
			ax.set_title("%s"%(ch_names[ich]),fontsize=8)	
			df_this=df_selected[np.logical_and(df_selected['pID']==pID,df_selected['ch_name']==ch_names[ich])]

			df_this=df_this.sort_values(by='freqPeak',ascending=False)
			df_this=df_this.reset_index(drop=True)

			
			ax2=ax.twinx()
			for iFreq in range(0,len(df_this)):		
				ch_name=df_this.loc[iFreq,'ch_name']	
				freqLow,freqHigh=df_this.loc[iFreq,'freqLow'],df_this.loc[iFreq,'freqHigh']			
				
				delayT,periEventTrue,saccadeAutoCorr,nSaccade,nBurst=getPeriEventHist(pID=pID,ch_name=ch_name,maxDelayInSec=maxDelayInSec,
					smoothScaleInMs=1000,sleepscoreSelect=sleepscoreSelect,
					freqLowBand=freqLow,freqHighBand=freqHigh,minWidthInCycles=3,sfreq=200,
					eyeMovIndexToUse='peakVelocityIndex',burstTimeToUse='peak')
				
				ax.plot(delayT,periEventTrue,label='%.1f-%.1f Hz;\nnSacc=%d,nBursts=%d'%(freqLow,freqHigh,nSaccade,nBurst),c='C%d'%iFreq)
				
				ax2.plot(delayT,saccadeAutoCorr,c='black')							
				ax.legend()
			ax.axvline(0,ls='--',c='black',lw=0.5)			


			ax.set_ylabel("Burst Probability")
			ax2.set_ylabel("Saccade Probability")
			ax.yaxis.label.set_color('C0')
			ax.tick_params(axis='y', colors='C0')
			ax.spines["right"].set_edgecolor('C0')
			ax.set_xlabel("Time relative saccade (sec)")	

			ax.minorticks_on()
		#axs[0,0].set_xlim((-1.5,1.5))
		#plt.show()
		pdf.savefig(fig)
		plt.close()
		plt.clf()
	pdf.close()	




plotBurstProbability(sleepscoreSelect=5,outname='figures/periEventHist_saccades_REM.pdf',maxDelayInSec=60)



