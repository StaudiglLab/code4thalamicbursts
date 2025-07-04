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
from burst.plotHelpers import *
from psd.periodicPowerInBand import getAlignedPeriodicPower

import matplotlib.gridspec as gridspec

import matplotlib
sns.set_palette("deep")
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 


#plot ratio of burst rates in bands at the group level
def plotBurstRateRatioGroupLevel(band,ax):
	df_gamma=getSignificantBands(which=band) 
	
	pID=df_gamma['pID'].values
	pID[pID=='p14_followup']='p14'
	df_gamma['pID']=pID
	uniqPID=np.unique(pID)
	states=['wake','NREM','phasic_REM','tonic_REM']
	burstRates_subj=np.zeros((len(uniqPID),len(states)))
	nSubj=len(uniqPID)
	
	for iPID in range(0,len(uniqPID)):
		for iState in range(0,len(states)):
			burstRates_subj[iPID,iState]=np.mean(df_gamma['meanBurstRate_%s'%states[iState]][df_gamma['pID']==uniqPID[iPID]])
	#print(burstRates_subj)
	burstRates_subj=np.log10(burstRates_subj)
	colors=[sns.color_palette("deep")[0],sns.color_palette("deep")[2],sns.color_palette("deep")[1],sns.color_palette("deep")[3]]
	violin=sns.violinplot(burstRates_subj,palette=colors,ax=ax,cut=0,alpha=0.75,width=0.5)

	for i in range(0,len(uniqPID)):
		ax.plot(np.arange(len(states)),(burstRates_subj[i]),c='gray',lw=1,marker='o',ms=1,zorder=-999)
	start=1.9
	spacing=0.07
	
    #plotting significant effects
	#the significance was obtained from the code groupStats.py
	ax.text(0.50,0.05,r"N$_\mathrm{subjects}=%d$"%nSubj,transform=ax.transAxes)
	
	ax.plot([0,1],[start,start],c='black',marker='|')
	ax.plot([0,3],[start+spacing,start+spacing],c='black',marker='|')

	ax.plot([1,2],[start+2*spacing,start+2*spacing],c='black',marker='|')
	ax.plot([1,3],[start+3*spacing,start+3*spacing],c='black',marker='|')    
	ax.plot([2,3],[start+4*spacing,start+4*spacing],c='black',marker='|')
	
	ax.set_ylabel("Burst Rate (/min)")
	ax.set_yticks([0,1],["1","10"])
	ax.set_yticks([0,1],["1","10"])
	ax.set_xticks([0,1,2,3],['wake','NREM','phasic\nREM','tonic\nREM'])
	ax.set_ylim((-0.3,2.25))

if(__name__=='__main__'):	
	ax=plt.subplot(111)
	plotBurstRateRatioGroupLevel('gamma',ax)
	plt.savefig("figures/phasicTonic.pdf",bbox_inches='tight',dpi=300)