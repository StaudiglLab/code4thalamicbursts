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
import pandas as pd

from coreFunctions import *
from scipy.stats import wilcoxon


# block bootstrap detection counts
def getDetectionCounts( pIDs, 		#list of pIDs
			ch_names, 	#list of corresponding channel names
			detectionMasks, #tupple with arrays, each of same size indicating whether a band is detected or not 
			niter		#number of iterations for bootstrap
			):
	
			

	nMasks=len(detectionMasks)  #number of detection masks to work with
	pID_chname=pIDs+ch_names
	uniquePID=np.unique(pIDs)	
	npID=len(uniquePID)
	
	#get counts per electrode
	meanRate=np.zeros(nMasks)	
	for i in range(0,nMasks):
		meanRate[i]=len(np.unique(pID_chname[detectionMasks[i]]))
	meanRate/=len(np.unique(pID_chname))
	print("Total number of contacts: %d"%len(np.unique(pID_chname)))
	print("Mean detection rates",meanRate)
	#bootstrap subjects and get counts

	countRealizations=np.zeros((nMasks,niter))
	for iRealization in range(0,niter):
		#randomly select a set of patients
		pIDselect=uniquePID[np.random.randint(low=0,high=npID,size=npID)]
		
		norm=0 #total number of channels for the given patient selection is counted here
		for iPID in range(0,npID):
			for k in range(0,nMasks):
				countRealizations[k,iRealization]+=len(np.unique(pID_chname[np.logical_and(detectionMasks[k],pIDs==pIDselect[iPID])]))
			norm+=len(np.unique(pID_chname[pIDs==pIDselect[iPID]]))
		countRealizations[:,iRealization]/=norm
	return meanRate,countRealizations
	

#function to perform wilcoxon rank test at subject level
def compareSubjLevel(rate1,rate2,pID):
	uniqPID=np.unique(pID)
	rate1Subj=np.zeros(len(uniqPID))
	rate2Subj=np.zeros(len(uniqPID))
	print("Number of subjects for test:%d"%len(uniqPID))
	for i in range(0,len(uniqPID)):
		selmask=pID==uniqPID[i]
		rate1Subj[i]=np.mean(rate1[selmask])
		rate2Subj[i]=np.mean(rate2[selmask])
	return wilcoxon(x=rate1Subj, y=rate2Subj).pvalue
	

#function to count number of bands
def countBands():
	df=getSignificantBands('allUnique')
	print("Total number of unique bands: %d"%len(df))
	
	df=getSignificantBands('allUnique')
	print("Total number of unique bands at <=19 Hz: %d"%np.sum(df['freqPeak']<19))
	
	
	df=getSignificantBands('spindle')
	print("Total number of unique contacts with spindles: %d"%len(np.unique(df['pID']+df['ch_name'])))
	df=getSignificantBands()
	print("Number of REM specific bands: %d"%np.sum(np.logical_and(df['state']=='REM',df['freqPeak']>=19)))
	print("Number of Wake specific bands: %d"%np.sum(np.logical_and(df['state']=='wake',df['freqPeak']>=19)))
	print("Number of overlap bands: %d"%np.sum(np.logical_and.reduce((df['state']=='REM',df['freqPeak']>=19,df['hasOverlap_wake']))))
	
#function to get detection probability of bands per bipolar contact	
def getDetectionProbabilities():
	df=getSignificantBands()
	detmaskREM=np.logical_and.reduce((df['state']=='REM',df['freqPeak']>=19))
	detmaskwake=np.logical_and.reduce((df['state']=='wake',df['freqPeak']>=19))	
	detmaskNREM=np.logical_and.reduce((df['state']=='NREM',df['freqPeak']>=19))

	
	
	#changing stuff for reimplantion patient to make sure it is counted as one patient
	#but that the contacts from the reimplantion epochs are counted as different contacts
	pID=df['pID'].values
	ch_name=df['ch_name'].values
	ch_name[pID=='p14_followup']+='_fup'
	pID[pID=='p14_followup']='p14'
	print(ch_name[pID=='p14'])

	detmasks=np.array([detmaskREM,detmaskwake,detmaskNREM])
	meanCounts,countRealizations=getDetectionCounts(pIDs=pID,
				ch_names=ch_name,
				detectionMasks=detmasks,
				niter=100000)
				
	print("Detection probability in of REM specific oscillation (at >19 Hz):%.2f+/-%.2f"%(np.mean(countRealizations[0]),np.std(countRealizations[0])))
	print("Detection probability in of wake specific oscillation (at >19 Hz):%.2f+/-%.2f"%(np.mean(countRealizations[1]),np.std(countRealizations[1])))
	print("pvalue of detection probability in REM>NREM (for bands at >19 Hz)")
	print(np.mean(countRealizations[0]<=countRealizations[2]))
	print("pvalue of detection probability in Wake>NREM (for bands at >19 Hz)")	
	print(np.mean(countRealizations[1]<=countRealizations[2]))
	
#compare rates at the subject level between different bands	
def getRateStats():
	df=getSignificantBands(which='uniqueAll')	
	print("Number of unique bands:%d"%len(df))
	
	burstRateNREM=df['meanBurstRate_NREM'].values
	burstRateWake=df['meanBurstRate_wake'].values	
	burstRateREM=df['meanBurstRate_REM'].values
	
	#changing ID of reimplanted patient to ensure proper subject level comparisions
	pID=df['pID'].values
	pID[pID=='p14_followup']='p14'
	
	maskSpindle=np.logical_and(df['freqPeak']>=11,df['freqPeak']<=17)
	maskGamma=df['freqPeak']>=19

	
	pSpindleWake=compareSubjLevel(burstRateNREM[maskSpindle],burstRateWake[maskSpindle],pID[maskSpindle])
	pSpindleREM=compareSubjLevel(burstRateNREM[maskSpindle],burstRateREM[maskSpindle],pID[maskSpindle])
	print("p-values for rates in spindle bands: %.3e (wake), %.3e (REM)"%(pSpindleWake,pSpindleREM))
	pGammaWake=compareSubjLevel(burstRateNREM[maskGamma],burstRateWake[maskGamma],pID[maskGamma])
	pGammaREM=compareSubjLevel(burstRateNREM[maskGamma],burstRateREM[maskGamma],pID[maskGamma])
	print("p-values for rates in bands >=19 Hz: %.3e (wake), %.3e (REM)"%(pGammaWake,pGammaREM))


#compare rates at the subject level between different bands	
def getScalpDetectionStats():
	df=getSignificantBands(which='gamma')
	pID,freqLow,freqHigh=df['pID'].values,df['freqLow'].values,df['freqHigh'].values
	print("Number of unique bands:%d"%len(df))
	nScalpOverlap_Wake=np.zeros(len(df))
	nScalpOverlap_REM=np.zeros(len(df))
	nScalpOverlap_overlap=np.zeros(len(df))

	df_scalp=pd.read_csv("outfiles/bursts_detectedFrequencies_scalp_selected.txt",sep=' ')	
	for i in range(0,len(df)):
		subjsel=df_scalp['pID']==pID[i]
		bandsel=np.logical_and(freqLow[i]<df_scalp['freqPeak'],df_scalp['freqPeak']<freqHigh[i])
		wakeMask=np.logical_and.reduce((subjsel,bandsel,df_scalp['state']=='wake',df_scalp['meanBurstRate_wake']>df_scalp['meanBurstRate_NREM']))
		REMmask=np.logical_and.reduce((subjsel,bandsel,df_scalp['state']=='REM',df_scalp['meanBurstRate_REM']>df_scalp['meanBurstRate_NREM']))
		nScalpOverlap_Wake[i]=np.sum(wakeMask)
		nScalpOverlap_REM[i]=np.sum(REMmask)
		nScalpOverlap_overlap[i]=len(np.intersect1d(df_scalp['ch_name'][wakeMask],df_scalp['ch_name'][REMmask]))
		print(pID[i],np.intersect1d(df_scalp['ch_name'][wakeMask],df_scalp['ch_name'][REMmask]))
	print(pID,nScalpOverlap_overlap)
			
	detmasks=np.array([nScalpOverlap_overlap>0])
	meanCounts,countRealizations=getDetectionCounts(pIDs=pID,
				ch_names=pID,
				detectionMasks=detmasks,
				niter=10000)
	print(np.mean(countRealizations==0,axis=1))
	print("Detection probability in of REM/wake specific oscillation (at >19 Hz):%.2f+/-%.2f"%(np.mean(countRealizations[0]),np.std(countRealizations[0])))
getScalpDetectionStats()
#countBands()	
#getDetectionProbabilities()
#getRateStats()
