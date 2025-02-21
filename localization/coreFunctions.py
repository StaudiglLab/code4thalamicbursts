import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import numpy as np
import mne
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import Patch



import nilearn 
from nilearn.plotting import plot_markers,plot_roi,plot_anat
from nilearn.plotting.displays import OrthoSlicer,MosaicSlicer,XSlicer

from core import *
from core.helpers import *
from matplotlib.colors import  ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from coreFunctions import *

cmapToUse=ListedColormap(plt.cm.Dark2.colors[:7])
groupCodes={'AV':1,'MDm':2,'VP':3,'VA':4,'VL':5,'Central':6,'Posterior':7,'Reticular':8,'NA':0}

#function to read in coordinates of electrodes
def getCoordinates(pID):
	f = h5py.File(rootdir+'/localization/leadDBSMat/%s.mat'%pID,'r')
	coord_R = f[f.get('reco/native/coords_mm')[0][0]]
	coord_L=f[f.get('reco/native/coords_mm')[1][0]]
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

