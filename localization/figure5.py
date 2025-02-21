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
import matplotlib.patches as mpatches

from core import *
from core.helpers import *
from burst.coreFunctions import *
from sklearn.linear_model import LogisticRegression
import matplotlib.gridspec as gridspec

from vizualizeCT import vizCT


def getDetectionsOnContact_REMs(pvalThresh=5e-4):

	#get contacts with burst that correlate significantly with REMs
	df_REM=pd.read_csv("../burstAndREMs/outfiles/REM_burstSaccadeCrossCorr.txt",sep=' ')
	selmask=df_REM['crossCorrCoeff_pvalue']<=pvalThresh
	df_REM=df_REM[selmask]
	
	#get all pIDs for study
	pIDs=np.array(cohortForPaper)
		
	#print(pIDs)
	#print(np.unique(df_REM['pID'].values))
	
	#create array with True for contacts where signal is detected
	hasSignal=np.zeros((len(pIDs),8),dtype=bool)
	
	for iSub in range(0,len(pIDs)):
		#select contacts for all channels
		selmask=df_REM['pID'].values==pIDs[iSub]
		dfsel=df_REM[selmask].reset_index(drop=True)
		uniqCh=np.unique(dfsel['ch_name'].values)
		#loop over channels with detection
		for ch_name in uniqCh:
			#set both contacts in bipolar montage to true
			if(ch_name[0]=='L'):
				hasSignal[iSub,int(ch_name[1])-1]=True
				hasSignal[iSub,int(ch_name[-1])-1]=True		
			elif(ch_name[0]=='R'):
				hasSignal[iSub,4+int(ch_name[1])-1]=True
				hasSignal[iSub,4+int(ch_name[-1])-1]=True
	return pIDs,hasSignal
	

#bootstrap and do logistic regression
def bootStrapLogisticRegression(hasSignal,distances,niter):

		
	nstruct=distances.shape[-1]	
	
	#distribution of regression coefficients
	coeff=np.zeros((nstruct,niter))
	
	#distribution of probability curves, for plotting
	distanceRange=np.expand_dims(np.arange(0,10,0.05),axis=1)
	probDistribution=np.zeros((nstruct,niter,len(distanceRange)))	
	nSubj=len(hasSignal)
	
	#bootstrap
	for i in range(niter):
		#select surrogate cohort
		subjSel=np.random.randint(low=0,high=nSubj,size=nSubj)
		
		hasSignalSel=hasSignal[subjSel]
		distancesSel=distances[subjSel]
		validMask=hasSignalSel>-1
		hasSignalSel=hasSignalSel[validMask].flatten()
		#perform logistic regression for reach structure
		for iStruct in range(0,nstruct):
			lg=LogisticRegression(solver='liblinear')
			lg.fit(np.expand_dims(distancesSel[:,:,iStruct][validMask].flatten(),axis=1),hasSignalSel)
			coeff[iStruct,i]=lg.coef_[0,0]
			probDistribution[iStruct,i]=lg.predict_proba(distanceRange)[:,1]

	return coeff,distanceRange[:,0],probDistribution

#perform logistic regression for multiple regions

def logisticReg(infile='contacts_freesurfer.txt',
		structures=['Central','AV'],
		axs=None			#matplotlib axes to plot regressions on
		):

	#get information on contacts with signal
	pIDs,hasSignal=getDetectionsOnContact_REMs()
	
	#load distances to structures
	df = pd.read_csv(infile,sep=' ',header=0)
	selmask=np.isin(df['pID'].values,pIDs)
	df=df[selmask]
	
	distances=np.zeros((hasSignal.shape[0],hasSignal.shape[1],len(structures)))
	for iStruct in range(len(structures)):
		distances[:,:,iStruct]=df['%s_distNearest'%structures[iStruct]].values.reshape(hasSignal.shape)
	#for distance < 1 mm, set to 0.5
	distances[distances<1.0]=0.5		
		
	#removing left electrode of p26; it was not connected.
	indx26=np.arange(len(pIDs))[pIDs=='p26']
	hasSignal=hasSignal.astype("int")
	hasSignal[indx26,:4]=-9999
	
		
	#bootstrap
	coeff,dist,probDistribution=bootStrapLogisticRegression(hasSignal,distances,niter=100000)
	
	for iStruct in range(len(structures)):
		print("---------------------")
		print(structures[iStruct])
		print("pvalues: %.2e"%np.mean(coeff[iStruct]>0))
		print("reg coeff",np.mean(coeff[iStruct]))
		print("reg coeff er",np.std(coeff[iStruct]))	
		print("---------------------")
		
	print("pvalues for coefficient for %s > %s"%(structures[0],structures[1]),np.mean(coeff[0]>coeff[1]))
	
	
	
	if(axs is None):
		return -1
	#plot regression coefficient
	

	c=['C3','C0' ,'C0']
	let=['(A)','(B)','(C)']
	
	#for plotting, with scatter
	distancesToPlot=distances.copy()
	distancesToPlot[distances<1.0]+=np.random.randn(np.sum(distances<1.0))*0.06
	for i in range(0,len(structures)):
		distThis=distancesToPlot[:,:,i]
		scaty=np.random.randn(distThis.shape[0],distThis.shape[1])*0.01		
		scaty[distThis>1]=0
		
		#violin plots for illustration
		violin_parts=axs[i].violinplot(distThis[hasSignal==0],positions=[-0.05],vert=False,widths=0.12*np.sum(distThis[hasSignal==0]<1)/50.0,showextrema=False)			
		for pc in violin_parts['bodies']:
			pc.set_facecolor('gray')
			pc.set_edgecolor('black')
		violin_parts=axs[i].violinplot(distThis[hasSignal==1],positions=[1.0],vert=False,widths=0.12*np.sum(distThis[hasSignal==1]<1)/50.0,showextrema=False)	

		for pc in violin_parts['bodies']:
			pc.set_facecolor('gray')
			pc.set_edgecolor('black')  
		#data points
		axs[i].scatter(distThis[hasSignal==0],scaty[hasSignal==0]-0.05,s=0.05,c='black')		
		axs[i].scatter(distThis[hasSignal==1],scaty[hasSignal==1]+1,s=0.05,c='black')				
		
		#plot mean and standard deviation
					
		mean=np.mean(probDistribution[i],axis=0)
		std=np.std(probDistribution[i],axis=0)
		
		axs[i].plot(dist,mean,c=c[i])
		axs[i].fill_between(dist,mean-std,mean+std,fc=c[i],alpha=0.5)
		axs[i].set_xlim((0,np.max(distThis)+0.5))
		axs[i].set_xlabel("Distance (mm)")
		axs[i].minorticks_on()
		axs[i].set_title("%s %s Thalamus"%(let[i+1],structures[i]),loc='left',fontdict={'fontweight':'bold','fontsize':10})


def plotFigure5():
	fig = plt.figure(figsize=(14, 5))
	plt.subplots_adjust(wspace=0.7)
	gs = fig.add_gridspec(2, 8)
	
	ax_CT=fig.add_subplot(gs[:, 2:5])
	ax_AV=fig.add_subplot(gs[:, 5:])
	logisticReg(structures=['Central','AV'],
			axs=[ax_CT,ax_AV])
	ax_CT.set_ylabel("Detection probability of Wake, REM Oscillations")
	ax_AV.set_ylabel("Detection probability of Wake, REM Oscillations")
	

	ax_slice1=fig.add_subplot(gs[0, :2])
	ax_slice2=fig.add_subplot(gs[1, :2])
	#ax_slice1.set_title(loc='left',fontdict={'fontweight':'bold','fontsize':10})
	vizCT([ax_slice1,ax_slice2])
	ax_slice1.set_title("(A) Thalamic Regions",loc='center',fontdict={'fontweight':'bold','fontsize':10})
	plt.savefig("figures/figure5.pdf",bbox_inches='tight',dpi=600)
	
plotFigure5()



