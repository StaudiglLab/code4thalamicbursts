import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git
repo = git.Repo('.', search_parent_directories=True)
import sys
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import matplotlib

from core import *
from core.helpers import *
from scipy.interpolate import interp1d
from burst.coreFunctions import getSignificantBands

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 

#load connectivity file and average metric over specified scalp electrodes 
def readMeanConnectivity(pID,
			ch_name_thal,	#thalamic channel
			ch_names_scalp, #scalp channels
			state,		#brain state
			ref='Cpz'	#reference
			):
	#dummy load to get shape of array
	freqs,pli=np.loadtxt(rootdir+"/connectivity/pli_%s_%s-%s_%s_ref%s.txt"%(pID,ch_name_thal,ch_names_scalp[0],state,ref),usecols=[0,2],unpack=True)
	pli_all=np.zeros((len(ch_names_scalp),len(freqs)))
	#iterate over all channels
	for iChan in range(len(ch_names_scalp)):
		freqs,pli_all[iChan,:]=np.loadtxt(rootdir+"/connectivity/pli_%s_%s-%s_%s_ref%s.txt"%(pID,ch_name_thal,ch_names_scalp[iChan],state,ref),usecols=[0,2],unpack=True)
	return freqs,np.mean(pli_all,axis=0)	
	
#function to get frequency resolved connectivity metric
#after aligning to the peak frequency of each band
def getConnectivityInBand(band,
			reference,
			subjectlevel=True #get subject level average
			):
	#load dataframe with bands
	if(band=='spindle'):
		df_bands=getSignificantBands('spindleInGammaChannels')
	else:
		df_bands=getSignificantBands(band)
		
	#define states and electrode groups
	states=['wake','REM','NREM']
	electrodeGroups={'Frontal L':['Fp1','F7'],'Frontal R':['Fp2','F8'],'Central L':['T7','P7'],'Central R':['T8','P8'],'Occipital L':['O1'],'Occipital R':['O2']}
	electrodeGroupNames=list(electrodeGroups)
	
	#common frequency axis to align and interpolate onto
	freqAxisCommon=np.arange(-6,6.1,1)
	
	
	#initialize new variables into dataframe
	connectivityBands=np.zeros((len(df_bands),len(list(electrodeGroups)),len(states),len(freqAxisCommon)))
	
	#iterate over all bands
	for i in range(len(df_bands.index)):
		indx=df_bands.index[i]
		pID=df_bands.loc[indx,'pID']
		if(pID=='p20' or pID=='p26'):
			electrodeGroups['Frontal L']=['F7']
			electrodeGroups['Frontal R']=['F8']
		else:
			electrodeGroups['Frontal L']=['Fp1','F7']
			electrodeGroups['Frontal R']=['Fp2','F8']

		freqLow=df_bands.loc[indx,'freqLow']
		freqHigh=df_bands.loc[indx,'freqHigh']
		for iState in range(len(states)):
			for iGroup in range(len(electrodeGroupNames)):
				#load connectivity
				freqs,conn=readMeanConnectivity(pID=pID,
						ch_name_thal=df_bands.loc[indx,'ch_name'],
						ch_names_scalp=electrodeGroups[electrodeGroupNames[iGroup]],
						state=states[iState],
						ref=reference)
				#interpolate to common frequency axes
				connectivityBands[i,iGroup,iState]=np.interp(freqAxisCommon,freqs-df_bands.loc[indx,'freqPeak'],np.abs(conn))
				
	#perform subject level average, if asked for
	if(subjectlevel):
		pIDs=df_bands['pID'].values
		pIDs[pIDs=='p14_followup']='p14'	
		uniqPID=np.unique(pIDs)
		print(uniqPID)
		connectivityBandsSubj=np.zeros((len(uniqPID),len(list(electrodeGroups)),len(states),len(freqAxisCommon)))
		for iPID in range(len(uniqPID)):
			connectivityBandsSubj[iPID]=np.mean(connectivityBands[pIDs==uniqPID[iPID]],axis=0)
		connectivityBands=connectivityBandsSubj
	return df_bands,freqAxisCommon,electrodeGroupNames,states,connectivityBands

#function to get mean connectivity metric in each band
#after aligning to the peak frequency of each band	
def getConnectivityInBandMean(band,reference):
	#load dataframe with bands
	if(band=='spindle'):
		df_bands=getSignificantBands('spindleInGammaREMChannels')
	else:
		df_bands=getSignificantBands(band)
	#define states and electrode groups
	states=['wake','NREM','REM']
	electrodeGroups={'Frontal_L':['Fp1','F7'],'Frontal_R':['Fp2','F8'],'Central_L':['T7','P7'],'Central_R':['T8','P8'],'Occipital_L':['O1'],'Occipital_R':['O2']}

	#add variables to dataframe where average metrics would be saved
	for state in states:
		for group in list(electrodeGroups):
			df_bands['pli_%s_%s'%(group,state)]=np.zeros(len(df_bands))
	df_bands.reset_index()
	
	#loop over all bands
	for indx in df_bands.index:
		pID=df_bands.loc[indx,'pID']
		if(pID=='p20' or pID=='p26'):
			electrodeGroups['Frontal_L']=['F7']
			electrodeGroups['Frontal_R']=['F8']
		else:
			electrodeGroups['Frontal_L']=['Fp1','F7']
			electrodeGroups['Frontal_R']=['Fp2','F8']
		
		freqLow=df_bands.loc[indx,'freqLow']
		freqHigh=df_bands.loc[indx,'freqHigh']
		#loop over all states and electrode groups
		for state in states:
			for group in list(electrodeGroups):
				freqs,conn=readMeanConnectivity(pID=pID,
						ch_name_thal=df_bands.loc[indx,'ch_name'],
						ch_names_scalp=electrodeGroups[group],
						state=state,
						ref=reference)
				#take average over frequency band
				meanConn=np.mean(np.abs(conn)[np.logical_and(freqs>=freqLow,freqs<=freqHigh)])				
				df_bands.loc[indx,'pli_%s_%s'%(group,state)]=meanConn							
	return df_bands





	
	
#plot group level distribution
def plotDistributionAndDoStats(df_conn_group, #data frame containing average connectivity metrics
			ax,	#matplotlib axes to draw on
			colors=['C0','C2'],
			alpha=[1,1]):
	electrodeGroups=['Frontal','Central','Occipital']
	binWake_REM=np.zeros((len(df_conn_group),len(df_conn_group)))
	binNREM=np.zeros((len(df_conn_group),len(df_conn_group)))
	
	#seperate data into two states
	for iGroup in range(len(electrodeGroups)):
		group=electrodeGroups[iGroup]				
		for i in range(len(df_conn_group)):
			indx=df_conn_group.index[i]
			binWake_REM[iGroup,i]=(df_conn_group.loc[indx,'pli_%s_%s_%s'%(group,'ipsi','wake/REM')]+df_conn_group.loc[indx,'pli_%s_%s_%s'%(group,'contra','wake/REM')])/2.0
			binNREM[iGroup,i]=(df_conn_group.loc[indx,'pli_%s_%s_%s'%(group,'ipsi','NREM')]+df_conn_group.loc[indx,'pli_%s_%s_%s'%(group,'contra','NREM')])/2.0
		
		
		dataStacked=(np.column_stack((binWake_REM[iGroup],binNREM[iGroup])))
		stats=scipy.stats.wilcoxon(binWake_REM[iGroup],binNREM[iGroup])
		print(group,stats)
		#plot line to indicate significance
		if(stats.pvalue<0.05/(2*len(electrodeGroups))): #correcting for multiple comparisions
			ax.text(iGroup*2,0.31,"*")
			ax.plot(iGroup*2+np.array([-0.5,0.5]),[0.3,0.3],c='black')
		#make boxplots
		ax.plot(iGroup*2+np.array([-0.5,0.5]),dataStacked.T,c='gray',marker='o',alpha=0.5,ms=1)
		ax.boxplot(binWake_REM[iGroup],positions=[iGroup*2-0.5],widths=0.75,patch_artist=True,medianprops = dict(color='black'),boxprops  = dict(color='black',facecolor=colors[0],alpha=alpha[0]),showfliers=False)		
		ax.boxplot(binNREM[iGroup],positions=[iGroup*2+0.5],widths=0.75,patch_artist=True,medianprops = dict(color='black'),boxprops  = dict(color='black',facecolor=colors[1],alpha=alpha[1]),showfliers=False)
		ax.set_xticks(np.arange(0,3)*2,electrodeGroups)
		ax.set_ylim((10**-2.5,0.55))
	
	ax.set_ylabel("wPLI")
	ax.set_yscale("log")

	#plot legends
	labels=['wakefullness and REM sleep','NREM sleep']	
	patch1 = mpatches.Patch(color=colors[0], label=labels[0])
	patch2 = mpatches.Patch(color=colors[1], label=labels[1])	
	ax.legend(handles=[patch1,patch2])
	return binWake_REM,binNREM			
def connectivityGroupAnalysis(axs,band,reference):
	df_conn=getConnectivityInBandMean(band=band,reference=reference)
	electrodeGroups=['Frontal','Central','Occipital']
	hemis=['ipsi','contra']
	
	#group connectivity by hemisphere and combine wake and REM into one.
	for indx in df_conn.index:
		ipsi=df_conn.loc[indx,'ch_name'][0]
		hemiMap={'ipsi':ipsi}
		if(ipsi=='L'):
			hemiMap['contra']='R'
		else:
			hemiMap['contra']='L'			
		for group in electrodeGroups:
			for hemi in hemis:
				df_conn.loc[indx,'pli_%s_%s_wake/REM'%(group,hemi)]=(df_conn.loc[indx,'pli_%s_%s_%s'%(group,hemiMap[hemi],'wake')]
										+df_conn.loc[indx,'pli_%s_%s_%s'%(group,hemiMap[hemi],'REM')])/2.
				df_conn.loc[indx,'pli_%s_%s_NREM'%(group,hemi)]=df_conn.loc[indx,'pli_%s_%s_%s'%(group,hemiMap[hemi],'NREM')]	
	

	#get group level connecitivity metrics
	df_conn_group=pd.DataFrame()	
	pIDs=df_conn['pID'].values
	pIDs[pIDs=='p14_followup']='p14'
	df_conn['pID']=pIDs
	uniqPID=np.unique(pIDs)
	df_conn_group['pID']=uniqPID
	for i in range(0,len(uniqPID)):
		for group in electrodeGroups:
			for hemi in hemis:
				selmask=df_conn['pID']==uniqPID[i]
				df_conn_group.loc[i,'pli_%s_%s_wake/REM'%(group,hemi)]=np.mean(df_conn.loc[selmask]['pli_%s_%s_wake/REM'%(group,hemi)])
				df_conn_group.loc[i,'pli_%s_%s_NREM'%(group,hemi)]=np.mean(df_conn.loc[selmask]['pli_%s_%s_NREM'%(group,hemi)])

	
	#plot Distribution
	binWake_REM,binNREM=plotDistributionAndDoStats(df_conn_group,ax=axs,colors=[sns.color_palette()[0],sns.color_palette()[2]],alpha=[1,1])
	
	
	#check for significant differences amongst groups
	if(band=='gamma'):
		data=binWake_REM
	elif(band=='spindle'):
		data=binNREM
	
	iSig=0
	for iGroup in range(len(electrodeGroups)):
		for jGroup in range(iGroup+1,len(electrodeGroups)):	
			stats=scipy.stats.wilcoxon(data[iGroup],data[jGroup])
			print(iGroup,jGroup,stats)
			#plot if significant
			if(stats.pvalue<0.05/(2*len(electrodeGroups))): #correcting for multiple comparisions
				axs.text(iGroup+jGroup-0.5,0.41,"*")
				axs.plot([iGroup*2-0.5,jGroup*2-0.5],iSig*0.1+np.array([0.4,0.4]),c='black')
				iSig+=1
	
	print("Number of significant differences amongst electrode groups: %d"%iSig)

#plot average time resolved connectivity
def plotConnectivityRelFreq(axs,	#matplotlib axes to draw on
			band,		#frequency band (gamma or spindle)
			reference	#reference channel
			):
			
	df_bands,freqAxisCommon,electrodeGroupNames,states,connectivityBands=getConnectivityInBand(band=band,reference=reference)
	#plot group level mean and S.E.M for each band
	contactMean=np.mean(connectivityBands,axis=0)
	contactStd=np.std(connectivityBands,axis=0)/np.sqrt(connectivityBands.shape[0])
	for iGroup in range(0,len(electrodeGroupNames)):
		for iState in range(len(states)):
			axs[iGroup].text(0.02,0.9,electrodeGroupNames[iGroup],transform=axs[iGroup].transAxes,fontsize=10)
			axs[iGroup].plot(freqAxisCommon,contactMean[iGroup,iState],c='C%d'%iState)
			axs[iGroup].fill_between(freqAxisCommon,contactMean[iGroup,iState]-contactStd[iGroup,iState],contactMean[iGroup,iState]+contactStd[iGroup,iState],fc='C%d'%iState,alpha=0.5)

def EDF3():
	fig = plt.figure(figsize=(14, 12))
	gs = fig.add_gridspec(4,9,height_ratios=[2,1,1,1])
	fig.subplots_adjust(hspace=0.3)
	
	
	ax_mean_gamma=fig.add_subplot(gs[0,:4])
	connectivityGroupAnalysis(ax_mean_gamma,'gamma','Cpz')
	ax_mean_spindle=fig.add_subplot(gs[0,5:])
	connectivityGroupAnalysis(ax_mean_spindle,'spindle','Cpz')
	ax_freq_gamma=[]
	ax_freq_spindle=[]
	
	gs_sub=gs[1:, :4].subgridspec(3, 2, hspace=0.1)
	ax_freq_gamma = gs_sub.subplots(sharex=True,sharey=True)
	
	gs_sub=gs[1:, 5:].subgridspec(3, 2, hspace=0.1)
	ax_freq_spindle = gs_sub.subplots(sharex=True,sharey=True)
	
	
	ax_freq_gamma[0,0].set_ylabel("wPLI")
	ax_freq_gamma[1,0].set_ylabel("wPLI")
	ax_freq_gamma[2,0].set_ylabel("wPLI")	
	ax_freq_gamma[2,0].set_xlabel("Frequency (Hz)")	
	ax_freq_gamma[2,1].set_xlabel("Frequency (Hz)")	
	ax_freq_gamma[0,0].set_xlim((-6,6))
		
	ax_freq_spindle[0,0].set_ylabel("wPLI")
	ax_freq_spindle[1,0].set_ylabel("wPLI")
	ax_freq_spindle[2,0].set_ylabel("wPLI")	
	ax_freq_spindle[2,0].set_xlabel("Frequency (Hz)")	
	ax_freq_spindle[2,1].set_xlabel("Frequency (Hz)")	
	ax_freq_spindle[0,0].set_xlim((-6,6))
	
	ax_freq_gamma[0,0].set_xticks([-5,0,5],[r"f$_\mathrm{0}$-5 Hz", r"f$_\mathrm{0}$",r"f$_\mathrm{0}$+5 Hz"])
	ax_freq_gamma[1,0].set_xticks([-5,0,5],[r"f$_\mathrm{0}$-5 Hz", r"f$_\mathrm{0}$",r"f$_\mathrm{0}$+5 Hz"])
	ax_freq_spindle[0,0].set_xticks([-5,0,5],[r"f$_\mathrm{0}$-5 Hz", r"f$_\mathrm{0}$",r"f$_\mathrm{0}$+5 Hz"])
	ax_freq_spindle[1,0].set_xticks([-5,0,5],[r"f$_\mathrm{0}$-5 Hz", r"f$_\mathrm{0}$",r"f$_\mathrm{0}$+5 Hz"])	
	
	
	ax_mean_gamma.set_title("(A) Group Statistics for Fast Oscillations",loc='left',fontdict={'fontweight':'bold','fontsize':10})	
	ax_mean_spindle.set_title("(B) Group Statistics for Spindles",loc='left',fontdict={'fontweight':'bold','fontsize':10})			
		
	
	ax_freq_gamma[0,0].set_title("(C) Mean wPLI for Fast Oscillations",loc='left',fontdict={'fontweight':'bold','fontsize':10})	
	ax_freq_spindle[0,0].set_title("(D) Mean wPLI for Spindles",loc='left',fontdict={'fontweight':'bold','fontsize':10})		
	
	
	plotConnectivityRelFreq(ax_freq_gamma.flatten(),'gamma','Cpz')	
	plotConnectivityRelFreq(ax_freq_spindle.flatten(),'spindle','Cpz')		
	plt.savefig("figures/edf3.pdf",bbox_inches='tight',dpi=300.0)
EDF3()
