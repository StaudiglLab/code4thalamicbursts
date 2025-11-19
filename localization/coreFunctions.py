import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import numpy as np
import mne
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import Patch


import pandas as pd
import nilearn 
from nilearn.plotting import plot_markers,plot_roi,plot_anat
from nilearn.plotting.displays import OrthoSlicer,MosaicSlicer,XSlicer

from core import *
from core.helpers import *
from matplotlib.colors import  ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from coreFunctions import *

cmapToUse=ListedColormap(plt.cm.Dark2.colors[:7])
groupCodes={'AV':1,'MDm':2,'VP':3,'VA':4,'VL':5,'Central':6,'Posterior':7,'Reticular':8,'CM':9,'NA':0}

#function to read in coordinates of electrodes
def getCoordinates(pID,whichFrame='native'):
	f = h5py.File(rootdir+'/localization/leadDBSMat/%s.mat'%pID,'r')
	coord_R = f[f.get('reco/%s/coords_mm'%whichFrame)[0][0]]
	coord_L=f[f.get('reco/%s/coords_mm'%whichFrame)[1][0]]
	return {'L':np.array(coord_L).T,'R':np.array(coord_R).T}

	
#crop image to smaller size (useful to run faster)
	
def cropImage(im,coord,size,interpolation='nearest'):
	affine=np.copy(im.affine)
	voxelSize=abs(np.diag(affine)[0])
	imSize=np.array([int(size/voxelSize)]*3)
	if(affine.shape[1]==4):
		affine[:-1,3]=coord-np.matmul(affine[:3,:3],imSize/2.)
	else:
		affine_new=np.zeros((4,4))
		affine_new[:3,:3]=affine
		affine_new[3,3]=1
		affine_new[:-1,3]=coord-size/2.	
		affine=affine_new
	im2 = nilearn.image.resample_img(
	    im, target_affine=affine,interpolation=interpolation, target_shape=imSize
	)
	return im2

#group nuclei into broader catagories
def getThalamicParcellation(imTh,hemi):
	#read mapping of nuclei into groups from the .txt file
	code=np.loadtxt("freesurferLabels.txt",usecols=[0]).astype("int")
	groups=np.loadtxt("freesurferLabels.txt",usecols=[6],dtype='str')		

	imdata=nilearn.image.get_data(imTh)
	imdataNew=np.zeros_like(imdata,dtype='int32')

	for i in range(0,len(code)):
		imdataNew[imdata==code[i]]=groupCodes[groups[i]]
		
	#select group in only one hemisphere	
	if(hemi=='L'):
		imdataNew[imdata>8200]=0
	elif(hemi=='R'):
		imdataNew[imdata<8200]=0
		
	return nilearn.image.new_img_like(imTh,imdataNew)

def getDetectionsOnContact_REMs(pvalThresh=0.05/54.,isMidPointAnalysis=False):

	#get contacts with burst that correlate significantly with REMs
	df_REM=pd.read_csv("../burstAndREMs/outfiles/REM_burstSaccadeCrossCorr.txt",sep=' ')
	selmask=df_REM['crossCorrCoeff_pvalue']<=pvalThresh
	df_REM=df_REM[selmask]
	
	#get all pIDs for study
	pIDs=np.array(cohortForPaper)
		
	print(pIDs)
	print(np.unique(df_REM['pID'].values))
	print(len(df_REM))
	#create array with True for contacts where signal is detected
	if(isMidPointAnalysis):
		hasSignal=np.zeros((len(pIDs),6),dtype=bool)
	else:
		hasSignal=np.zeros((len(pIDs),8),dtype=bool)
	
	for iSub in range(0,len(pIDs)):
		#select contacts for all channels
		selmask=df_REM['pID'].values==pIDs[iSub]
		dfsel=df_REM[selmask].reset_index(drop=True)
		uniqCh=np.unique(dfsel['ch_name'].values)
		#loop over channels with detection
		if(isMidPointAnalysis):
			for ch_name in uniqCh:
				if(ch_name[0]=='L'):
					hasSignal[iSub,int(ch_name[1])-1]=True
					
				elif(ch_name[0]=='R'):
					hasSignal[iSub,3+int(ch_name[1])-1]=True
		else:
			for ch_name in uniqCh:
				#set both contacts in bipolar montage to true
				if(ch_name[0]=='L'):
					hasSignal[iSub,int(ch_name[1])-1]=True
					hasSignal[iSub,int(ch_name[-1])-1]=True		
				elif(ch_name[0]=='R'):
					hasSignal[iSub,4+int(ch_name[1])-1]=True
					hasSignal[iSub,4+int(ch_name[-1])-1]=True

	indx26=np.arange(len(pIDs))[pIDs=='p26']
	hasSignal=hasSignal.astype("int")
	#removing left electrode of p26
	if(isMidPointAnalysis):
		hasSignal[indx26,:3]=-9999
	else:
		hasSignal[indx26,:4]=-9999
	return pIDs,hasSignal