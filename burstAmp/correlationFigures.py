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
from pingouin import partial_corr
import matplotlib

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 



def getImpedenceValues(impedencefile='outfiles/impedance.csv'):
	dfImpedence=pd.read_csv(impedencefile)
	dfImpedence_bipolar=pd.DataFrame(columns=['pID','ch_name','impedance'])
	indx=0
	for i in range(1,len(dfImpedence)):
		for ch in ['L1-L2','L2-L3','L3-L4','R1-R2','R2-R3','R3-R4']:
			dfImpedence_bipolar.loc[indx,'pID']=dfImpedence.loc[i,'pID']
			dfImpedence_bipolar.loc[indx,'ch_name']=ch
			dfImpedence_bipolar.loc[indx,'impedance']=dfImpedence.loc[i,ch]/1000.0
			indx+=1
	return dfImpedence_bipolar

#get burst widths for all profiles
def getAllBurstAmplitudes():
	#load evoked during REM sleep



	taxis,evoked_REM,df_REM=getEvokedResponse(getSignificantBands('gammaREMCorrelated'),'REM')
	df_REM['amp']=getAmp(taxis,evoked_REM)
	
	#load evoked during wakefullness	
	taxis,evoked_wake,df_wake=getEvokedResponse(getSignificantBands('gammaREMCorrelated'),'wake')
	
	df_wake['amp']=getAmp(taxis,evoked_wake)	
	
	#combine REM and wake responses
	evoked_combined=(evoked_REM+evoked_wake)/2.
	df_combined=df_REM.copy()
	df_combined['amp']=getAmp(taxis,evoked_combined)	
	
	#load spindle evoked
	taxis,evoked_NREM,df_NREM=getEvokedResponse(getSignificantBands('spindleInGammaChannels'),'NREM')

	df_NREM['amp']=getAmp(taxis,evoked_NREM)

	#merge dataframes on pIDs and contacts
	df_merged=df_NREM.merge(df_combined,left_on=['pID','ch_name'],right_on=['pID','ch_name'],validate='m:m')
	df_merged_REM=df_NREM.merge(df_REM,left_on=['pID','ch_name'],right_on=['pID','ch_name'],validate='m:m')
	df_merged_wake=df_NREM.merge(df_wake,left_on=['pID','ch_name'],right_on=['pID','ch_name'],validate='m:m')

	return df_wake,df_REM,df_NREM,df_merged,df_merged_REM,df_merged_wake


#block bootstrap to get p-value
def calculatePValueSubj(pID,x,y,niter):

	uniqPID=np.unique(pID)
	rvals=np.zeros(int(niter))
	nSubj=np.zeros(len(uniqPID))
	nContacts=np.zeros(len(x))	
	for i in range(int(niter)):
		#draw surrogate cohort
		select=uniqPID[np.random.randint(low=0,high=len(uniqPID),size=len(uniqPID))]
		
		xthis=np.zeros(0)
		ythis=np.zeros(0)
		
		for j in range(0,len(select)):
			xthis=np.append(xthis,x[pID==select[j]])
			ythis=np.append(ythis,y[pID==select[j]])			
		
		res=np.mean((xthis-xthis.mean())*(ythis-ythis.mean()))/(xthis.std()*ythis.std())
		rvals[i]=res
		
	true_r=np.mean((x-x.mean())*(y-y.mean()))/(x.std()*y.std())
	print("True r: %.3f"%true_r)
	print("Mean r value: %.3f"%np.mean(rvals))
	print("Error on r value: %.3f"%np.std(rvals))
	print("Pvalue for positive correlation: %.2e"%np.mean(rvals<0)) 
	return np.mean(rvals<0)


#plot regression with pvalue
def plotCorrelationHelper(pID,x,y,ax,niter=int(1e5),shiftcolor=0):

	markers=['o',"v","^","<",">","8","s","p","P","h","X","D","H","d"]
	c=['C0','C3','C4','C5','C6','C7']
	uniqPID=np.unique(pID)
	for i in range(0,len(uniqPID)):
		pIDmask=pID==uniqPID[i]		
		ax.scatter(x[pIDmask],y[pIDmask],marker=markers[(i+shiftcolor)%5],c=c[(i+shiftcolor)%6],s=20)
	pID=pID.copy()
	pID[pID=='p14_followup']='p14'
	r=scipy.stats.linregress(x,y)
	pearson=scipy.stats.pearsonr(x,y,alternative='greater')
	pvalue=calculatePValueSubj(pID,x,y,niter=niter)
	if(pvalue<=1/niter):
		pvaltext=r'p-value<10$^{-%d}$'%np.log10(niter)
	elif(pvalue<1e-4):
		pvaltext='p-value<%d x$10^{-4}$'%(pvalue/1e-4)
	else:
		pvaltext='p-value=%.4f'%pvalue	
	xrnge=np.linspace(np.min(x),np.max(x),100)
	ax.plot(xrnge,r.slope*xrnge+r.intercept,ls='--',c='black',label='r = %.2f\n%s'%(pearson.statistic,pvaltext))
	ax.legend()	



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




def plotSupplementaryFigure():
	#fig = plt.figure(figsize=(14, 5))
	#plt.subplots_adjust(hspace=0.5,wspace=0.7)

	fig = plt.figure(figsize=(14*0.7, 3.5))

	plt.subplots_adjust(hspace=0.8,wspace=0.7)
	gs = gridspec.GridSpec(4,6)
	ax_spindle_example=[]
	ax_gamma_example=[]
	for i in range(0,4):	
		ax_spindle_example.append(fig.add_subplot(gs[i,0]))
		ax_gamma_example.append(fig.add_subplot(gs[i,1]))
		
	ax_spindle_example[0].set_title("(A) Spindle",loc='left',fontdict={'fontweight':'bold','fontsize':10})
	#ax_gamma_example[0].set_title("(B) Fast Oscillation",loc='left',fontdict={'fontweight':'bold','fontsize':10})
	ax_gamma_example[0].set_title("(B) Fast Osc.",loc='left',fontdict={'fontweight':'bold','fontsize':10})

	ax_corr_amp=fig.add_subplot(gs[:, 2:4])
	#ax_corr_width=fig.add_subplot(gs[:, 4:])
	
	#plot example bursts
	plotSingleAverage(['p03','pthal106','p21','p18']
			,['R3-R4','R1-R2','L2-L3','R1-R2'],ax_gamma_example,ax_spindle_example)
	
	for i in range(0,4):	
		ax_gamma_example[i].set_ylabel("")
	ax_corr_amp.set_title("(C) Amplitude Correlation",loc='left',fontdict={'fontweight':'bold','fontsize':10})
	

	
	#get amplitudes
	df_wake,df_REM,df_NREM,df_merged,df_merged_REM,df_merged_wake=getAllBurstAmplitudes()
	
	amp_1=df_merged['amp_x'].values
	amp_2=df_merged['amp_y'].values		

	print("----")	
	print("amp correlation")
	print("----")	
	plotCorrelationHelper(df_merged['pID'].values,amp_1,amp_2,ax_corr_amp)


	
	ax_corr_amp.set_xlabel(r"Spindle Amplitude ($\mu$V)")
	ax_corr_amp.set_ylabel(r"Fast Oscillation Amplitude ($\mu$V)")
		
			
	plt.savefig("figures/ampCorrelation.pdf",transparent=True,bbox_inches='tight',dpi=600.0)



def plotRateAmpCorrelation():

	fig,axs=plt.subplots(1,2,figsize=(8, 4))

	#get dataframes with widths
	df_wake,df_REM,df_NREM,df_merged,df_merged_REM,df_merged_wake=getAllBurstAmplitudes()

	rate_spindle,amp_spindle=df_merged['meanBurstRate_NREM_x'].values,df_merged['amp_x'].values
	rate_fosc,amp_fosc=(df_merged['meanBurstRate_REM_y'].values+df_merged['meanBurstRate_wake_y'].values)/2.,df_merged['amp_y'].values#
	pIDs=df_merged['pID'].values.copy()

	print("----")	
	print("spindle amp-rate correlation")
	print("----")	
	plotCorrelationHelper(pIDs,amp_spindle,rate_spindle,axs[0])
	axs[0].set_xlabel(r"Spindle Amplitude ($\mu$V)")
	axs[0].set_ylabel(r"Spindle Rate (per min)")

	print("----")	
	print("fosc amp-rate correlation")
	print("----")		
	plotCorrelationHelper(pIDs,amp_fosc,rate_fosc,axs[1])
	axs[1].set_xlabel(r"Fast Oscillation Amplitude ($\mu$V)")
	axs[1].set_ylabel(r"Fast Oscillation Rate (per min)")
	plt.savefig("figures/ampRate.png")
	

def plotImpedanceCorrelation():

	fig,axs=plt.subplots(1,3,figsize=(12, 4))
	#read impedance
	dfImpedance=getImpedenceValues()

	#get dataframes 
	df_wake,df_REM,df_NREM,df_merged,df_merged_REM,df_merged_wake=getAllBurstAmplitudes()

	#merge with impedance table
	df_merged=df_merged.merge(dfImpedance,left_on=['pID','ch_name'],right_on=['pID','ch_name'],validate='m:1')
	df_merged=df_merged[np.logical_not(np.isnan(np.array(df_merged['impedance'].values,dtype='float')))]



	imp,amp_spindle=df_merged['impedance'].values.astype("float"),df_merged['amp_x'].values#
	imp,amp_gamma=df_merged['impedance'].values.astype("float"),df_merged['amp_y'].values#
	
	

	
	pIDs=df_merged['pID'].values.copy()
	

	#correct for regression with impedance

	r1=scipy.stats.linregress(imp,amp_spindle)
	r2=scipy.stats.linregress(imp,amp_gamma)
	res_spindle=amp_spindle-(r1.slope*imp)
	res_gamma=amp_gamma-(r2.slope*imp)
	
	print("----")	
	print("spindle_amp-impedance correlation")
	print("----")	
	plotCorrelationHelper(pIDs,imp,amp_spindle,axs[1],shiftcolor=1)
	axs[1].set_xlabel(r"Impedance (k$\Omega$)")
	axs[1].set_ylabel(r"Spindle Amplitude ($\mu$V)")
	axs[1].set_title("(b) Spindles")


	print("----")	
	print("fastosc_amp-impedance correlation")
	print("----")
	plotCorrelationHelper(pIDs,imp,amp_gamma,axs[2],shiftcolor=1)
	axs[2].set_xlabel(r"Impedance (k$\Omega$)")
	axs[2].set_ylabel(r"Fast Oscillation Amplitude ($\mu$V)")
	axs[2].set_title("(c) Fast Oscilations")

	print("----")	
	print("fastosc_amp-spindle_amp correlation")
	print("----")	
	axs[0].set_title("(a) Amplitudes after regressing out impedence")
	plotCorrelationHelper(pIDs,res_spindle,res_gamma,axs[0],shiftcolor=1)
	axs[0].set_xlabel(r"Spindle Amplitude ($\mu$V)")
	axs[0].set_ylabel(r"Fast Oscillation Amplitude ($\mu$V)")


	plt.savefig("figures/impedance.pdf",bbox_inches='tight')




plotSupplementaryFigure()
plotImpedanceCorrelation()