import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import numpy as np
import mne
import matplotlib.pyplot as plt

import pandas as pd
import nilearn 

from coreFunctions import *
from core import *


from matplotlib.colors import  ListedColormap


import nibabel as nib


freesurferdir='/media/data/chowdhury/freesurfer/subjects'

	
def getDistanceFromNucleus(coordContacts,		 #coordinates
				segmentationMap,	#segmentation map from freesurfer
				segmentationCodes	#codes for nuclei
				):
	#load segmentation map
	imdata=nilearn.image.get_data(segmentationMap)
	
	#get affine and inverse affine transform
	affine=np.copy(segmentationMap.affine)
	invaffine=np.linalg.inv(affine)
	
	#get all indices as meshgrid
	x=np.arange(imdata.shape[0])
	y=np.arange(imdata.shape[1])
	z=np.arange(imdata.shape[2])
	x,y,z=np.meshgrid(x,y,z,indexing='ij')
	
	#get native space co ordinates for all indices	
	x,y,z=x.flatten(),y.flatten(),z.flatten()	
	coord=np.row_stack((x,y,z,np.ones(len(x))))
	coord_native=np.matmul(affine,coord)[:3]
	coord_native=coord_native.reshape((3,imdata.shape[0],imdata.shape[1],imdata.shape[2]))
	x_native,y_native,z_native=coord_native[0],coord_native[1],coord_native[2]
	
	#voxel of DBS contact
	

	nearestDistance=np.zeros((len(coordContacts),len(segmentationCodes)))
	hasContact=np.zeros((len(coordContacts),len(segmentationCodes)),dtype=bool)
	mask=np.zeros_like(imdata,dtype=bool)
	#iterate over contacts
	for iContact in range(0,len(coordContacts)):
		#iterate over all nuclei
		for iSegment in range(0,len(segmentationCodes)):
			coordContact=coordContacts[iContact]	
			#check if centre in nuclei
			contact_voxel=np.matmul(invaffine,np.append(coordContact,[1]))[:3].astype("int")
			hasContact[iContact,iSegment]=mask[contact_voxel[0],contact_voxel[1],contact_voxel[2]]
			#get mask containing points with nuclei	
			mask[imdata!=segmentationCodes[iSegment]]=False
			mask[imdata==segmentationCodes[iSegment]]=True
			
			#get all distances to the nuclei voxels			
			distanceToContact=np.linalg.norm(coord_native[:,mask]-np.expand_dims(coordContact,axis=1),axis=0)
			
			#get minimum distance
			nearestDistance[iContact,iSegment]=np.min(distanceToContact)			
			

	return hasContact,nearestDistance
	
def hasContactInNucleus(pID,hemi='L',segmentationCodesForDistance=[1,2,4,6]):
	if(pID=='p14_followup'):
		pID_freesurf='p14'
	else:
		pID_freesurf=pID	

	#load coordinates of electrode contacts
	coord=getCoordinates(pID)[hemi]

	#load freesurfer parcellation image
	imgThalParcellation=nilearn.image.load_img(freesurferdir+'/%s/mri/ThalamicNuclei.v13.T1.mgz'%pID_freesurf)
	
	#group nuclei and select hemisphere
	if(hemi=='L'):
		imgThalParcellation=getThalamicParcellation(imgThalParcellation,hemi='L')
	elif(hemi=='R'):
		imgThalParcellation=getThalamicParcellation(imgThalParcellation,hemi='R')		
	
	#get nearest distance to nuclei
	
	hasContact,nearestDistance=getDistanceFromNucleus(coord,segmentationMap=imgThalParcellation,segmentationCodes=segmentationCodesForDistance)

	return hasContact,nearestDistance


groupNames=list(groupCodes.keys())[:-1]
segmentationCodesForDistance=[1,2,3,4,5,6]
subjects=np.repeat(cohortForPaper,8)
contacts=np.tile(['tL1','tL2','tL3','tL4','tR1','tR2','tR3','tR4'],len(cohortForPaper))


#get nearest distances to each nuclei
hasContact=np.zeros((len(cohortForPaper)*8,len(segmentationCodesForDistance)),dtype=bool)
nearestDistance=np.zeros((len(cohortForPaper)*8,len(segmentationCodesForDistance)),dtype=float)
for iSub in range(len(cohortForPaper)):
	sl=np.s_[iSub*8:iSub*8+4]
	hasContact[sl],nearestDistance[sl]= hasContactInNucleus(cohortForPaper[iSub],segmentationCodesForDistance=segmentationCodesForDistance,hemi='L')
	sl=np.s_[iSub*8+4:iSub*8+8]
	hasContact[sl],nearestDistance[sl]= hasContactInNucleus(cohortForPaper[iSub],segmentationCodesForDistance=segmentationCodesForDistance,hemi='R')

#save output to dataframe
df = pd.DataFrame()
df['pID']=subjects
df['contacts']=contacts
for i in range(0,len(segmentationCodesForDistance)):
	groupN=groupNames[segmentationCodesForDistance[i]-1]
	df["%s"%groupN]=hasContact[:,i]	
	df["%s_%s"%(groupN,'distNearest')]=nearestDistance[:,i]	
df.to_csv("contacts_freesurfer.txt",sep=' ')	


