import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
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

from coreFunctions import getPeriEventHist
from numba import njit
import pandas as pd
from burst.coreFunctions import *	

#shuffle to get pvalue
def crossCorrShuffle(saccadeAutoCorr,	#saccade autocorrelation
			eventHist,	#burst event histogram
			nShuffle=int(2e5),	#number of shuffles
			pvalue=1-9e-4	#p-value at which to estimate range of r values in the shuffle distribution (used for plotting)
			):
	#interpolate central NaN in saccade autocorrelation
	notNanMask=np.logical_not(np.isnan(saccadeAutoCorr))
	xaxis=np.arange(len(saccadeAutoCorr))
	saccadeAutoCorr=np.interp(xaxis,xaxis[notNanMask],saccadeAutoCorr[notNanMask])
	
	#z-score both time series
	eventHist_zscored=(eventHist-np.mean(eventHist))/np.std(eventHist)
	saccadeAutoCorr_zscored=(saccadeAutoCorr-np.mean(saccadeAutoCorr))/np.std(saccadeAutoCorr)
	
	#true correlation coefficient
	trueCorr=np.mean(eventHist_zscored*saccadeAutoCorr_zscored)
	shuffledDist=np.zeros(nShuffle)
	
	#shuffle and get distribution
	for i in range(0,nShuffle):
		np.random.shuffle(eventHist_zscored)
		shuffledDist[i]=np.mean(eventHist_zscored*saccadeAutoCorr_zscored)
	
	pvalueGreater=np.mean(shuffledDist>trueCorr)
	pvalueLess=np.mean(shuffledDist<trueCorr)
	erRange=np.quantile(shuffledDist,q=[(1-pvalue)/2.,pvalue+(1-pvalue)/2.])
	return trueCorr,pvalueGreater,pvalueLess,erRange[0],erRange[1]

#function to test whether the perievent histograms and the saccade autocorrelations are correlated		
def periEventCorrelationStats():
	#load frequency bands
	df_selected=getSignificantBands(which='gamma')

	
	
	print("Number of bands:%d"%len(df_selected))
	#load perievent histograms
	periEventAll=np.load("outfiles/periEventAll_REM.npy")
	saccadeAutoCorrAll=np.load("outfiles/saccadeAutoCorrAll_REM.npy")
	print("Shape of perievent:",periEventAll.shape)	
	
	crossCorrCoeff=np.ones(len(df_selected))
	crossCorrCoeff_pvalue=np.ones(len(df_selected))
	coeff_low=np.ones(len(df_selected))
	coeff_high=np.ones(len(df_selected))	
	crossCorrCoeff_pvalue_neg=np.ones(len(df_selected))		
	pID_last=''
	
	#iterate over all bands
	for i in range(len(df_selected)):
		crossCorrCoeff[i],crossCorrCoeff_pvalue[i], crossCorrCoeff_pvalue_neg[i], coeff_low[i],coeff_high[i]= crossCorrShuffle(saccadeAutoCorrAll[i], periEventAll[i])
	
	#save dataframe with pvalues	
	df_selected['crossCorrCoeff']=crossCorrCoeff
	df_selected['crossCorrCoeffShuffle_low']=coeff_low
	df_selected['crossCorrCoeffShuffle_high']=coeff_high		
	df_selected['crossCorrCoeff_pvalue']=crossCorrCoeff_pvalue
	df_selected['crossCorrCoeff_pvalue_neg']=crossCorrCoeff_pvalue_neg
	df_selected.to_csv("./outfiles/REM_burstSaccadeCrossCorr.txt",sep=' ')
	
	
periEventCorrelationStats()	

