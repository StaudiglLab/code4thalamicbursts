import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import numpy as np
import mne
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

import nilearn 
from nilearn.plotting import plot_roi,plot_prob_atlas
import pandas as pd
from matplotlib.colors import  ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from coreFunctions import *
cmapToUse=ListedColormap(plt.cm.Dark2.colors[:7])


#helper function to group nuclei for the MNI atlas
def getThalamicParcellationForViz(imTh,labelTypes='Thal',whichGroups=np.array(['AV','Central']),hemi='LR'):
	if(labelTypes=='Group'):
		groups=np.loadtxt("MNIAtlas/ThalamusProbs.MNIsymSpace.names.txt",usecols=[2],delimiter=',',dtype='str')

	
		imdata=nilearn.image.get_data(imTh)
		imdataNew=np.zeros((imdata.shape[0],imdata.shape[1],imdata.shape[2],len(whichGroups)),dtype=imdata.dtype)

		for i in range(0,len(whichGroups)):
			imdataNew[:,:,:,i]=np.sum(imdata[:,:,:,groups==whichGroups[i]],axis=3)>0.5

	elif(labelTypes=='Thal'):	
		imdata=nilearn.image.get_data(imTh)
		imdataNew=np.zeros((imdata.shape[0],imdata.shape[1],imdata.shape[2]),dtype='int32')
		imdataNew[:,:,:]=np.sum(imdata[:,:,:,1:],axis=3)>0.05
	if(hemi=='L'):
		imdataNew[imdata>8200]=0
	elif(hemi=='R'):
		imdataNew[imdata<8200]=0
	return nilearn.image.new_img_like(imTh,imdataNew)

#helper function to read coordinates from the txt file
def readAllCoordinates(whichGroups):
	df=pd.read_csv("contacts_freesurfer.txt",sep=' ')
	coords=df[['MNI_X','MNI_Y','MNI_Z']].to_numpy()
	columns=[]
	for i in range(0,len(whichGroups)):
		columns.append('%s_distNearest'%whichGroups[i])
	distances=df[columns].values
	return df['pID'].values,coords,distances<1.0

def iterateThroughSlices(axs, #matplotlib axes to draw slices on
						whichContacts='hasSignal',
						whichSlice=0,
						nIter=5,
						whichGroups=np.array(['AV','Central','VA','MDm','VL'])
						):
	colors=np.array(['C0','C3','C2','C1','C4'])
	cmapToUse=ListedColormap(colors)
	
	#read contact information
	pIDcoord,allCoords,hasContact=readAllCoordinates(whichGroups=whichGroups)

	#flip all x-coordinates to plot everything on L-hemi
	allCoords[allCoords[:,0]>0,0]=-1*allCoords[allCoords[:,0]>0,0]

	#get contacts with singla
	pID,hasSignal=getDetectionsOnContact_REMs(isMidPointAnalysis=False)

	
	hasSignal=hasSignal.flatten()

	#these lines ensure that L electrode of p26 is excluded
	allCoords=allCoords[hasSignal>=0]
	#allNotDetMask=allNotDetMask[hasSignal>=0]
	hasContact=hasContact[hasSignal>=0]
	hasSignal=hasSignal[hasSignal>=0]


	#range of slices for the iteration
	sliceRange=np.linspace(np.min(allCoords[:,whichSlice]),np.max(allCoords[:,whichSlice]),nIter+1)
	
	
	sliceWidth=sliceRange[1]-sliceRange[0]
	sliceRange=(sliceRange[1:]+sliceRange[:-1])/2.

	


	#load probabilistic atlas of Iglesias et al. atlas download from https://freesurfer.net/fswiki/SubfieldAtlasesICBMspace

	imgThalParcellation=nilearn.image.load_img('MNIAtlas/ThalamusProbs.MNIsymSpace.nii.gz')
	
	#get boundary of thalamus
	imgThalParcellationThal=getThalamicParcellationForViz(imgThalParcellation,labelTypes='Thal')

	#get CT and AV parcellation
	imgThalParcellation=getThalamicParcellationForViz(imgThalParcellation,labelTypes='Group',whichGroups=whichGroups)
	
	#plot slices
	for i in range(0,len(sliceRange)):
		disp=plot_prob_atlas(imgThalParcellation,axes=axs[i],cut_coords=[sliceRange[i]],
					   black_bg=False, dim='auto',draw_cross=False,
					   cmap=cmapToUse,linewidths=1.5,display_mode=['x','y','z'][whichSlice],annotate=False,
					   view_type='contours',threshold=0.01) #,view_type='contours'
		disp.add_contours(imgThalParcellationThal,colors='black',linewidths=0.5)	

		#plot inset that shows the slice being plotted
		if(whichSlice==0):
			axin1 = axs[i].inset_axes([0.73, 0.99-0.3, 0.3, 0.3])
			insetCut=0
			display_mode='z'
		else:
			axin1 = axs[i].inset_axes([0.3/2., 1-0.3, 0.3, 0.3])
			insetCut=5
			display_mode='x'

		dispInset=plot_roi(imgThalParcellationThal,axes=axin1,cut_coords=[insetCut],black_bg=False, dim='auto',
					 draw_cross=False,cmap=ListedColormap(['black']),
					 linewidths=1.5,display_mode=display_mode,annotate=False,view_type='contours') #,view_type='contours'

		axMain=disp.axes[sliceRange[i]].ax		
		axInset=dispInset.axes[insetCut].ax

		if(whichSlice==1 or whichSlice==2):
			if(whichSlice==2):
				axInset.axhline(sliceRange[i],ls='--',c='black',lw=0.5)	
			else:
				axInset.axvline(sliceRange[i],ls='--',c='black',lw=0.5)	
			axInset.set_xlim((-35,10))
			axInset.set_ylim((-10,30))			
		elif(whichSlice==0):
			axInset.axvline(sliceRange[i],ls='--',c='black',lw=0.5)	
			axInset.set_xlim((-35,5))
			axInset.set_ylim((-40,5))
		axInset.set_axis_on()
		axInset.set_xticks([])
		axInset.set_yticks([])

		#select DBS contacts that lie in the given slice

		selmaskCoord=np.logical_and(allCoords[:,whichSlice]>=sliceRange[i]-sliceWidth/2.,allCoords[:,whichSlice]<sliceRange[i]+sliceWidth/2.)
		
		if(whichSlice==0):
			indx1,indx2=0,1			
		elif(whichSlice==1):
			indx1,indx2=0,2			
		elif(whichSlice==2):
			indx1,indx2=1,2

		if(whichContacts=='AVCTviz'):
			selmask=np.logical_and(selmaskCoord,np.logical_and(hasContact[:,0],np.logical_not(hasContact[:,1])))
			axMain.plot(allCoords[selmask,indx1],allCoords[selmask,indx2],markeredgecolor='black',markerfacecolor=colors[0],marker='o',ms=3,zorder=999,lw=0,markeredgewidth=0.3)
			selmask=np.logical_and(selmaskCoord,np.logical_and(hasContact[:,1],np.logical_not(hasContact[:,0])))		
			axMain.plot(allCoords[selmask,indx1],allCoords[selmask,indx2],markeredgecolor='black',markerfacecolor=colors[1],marker='o',ms=3,zorder=999,lw=0,markeredgewidth=0.3)
			selmask=np.logical_and(selmaskCoord,np.logical_and(hasContact[:,1],hasContact[:,0]))		
			axMain.plot(allCoords[selmask,indx1],allCoords[selmask,indx2],markeredgecolor='black',fillstyle='bottom',markerfacecolor=colors[1],markerfacecoloralt=colors[0],marker='o',ms=3,zorder=999,lw=0,markeredgewidth=0.3)
		elif(whichContacts=='hasSignal'):
			selmask=np.logical_and(selmaskCoord,hasSignal)
			axMain.plot(allCoords[selmask,indx1],allCoords[selmask,indx2],markeredgecolor='black',markerfacecolor='C6',marker='o',ms=3,zorder=999,lw=0,markeredgewidth=0.3)
			selmask=np.logical_and(selmaskCoord,np.logical_not(hasSignal))		
			axMain.plot(allCoords[selmask,indx1],allCoords[selmask,indx2],markeredgecolor='black',markerfacecolor='black',marker='s',ms=3,zorder=998,lw=0,markeredgewidth=0.3)
			
		
		
		if(whichSlice==0):
			axMain.set_xlim(-40, 20)
			axMain.set_ylim(-20, 30)
		elif(whichSlice==1):
			axMain.set_xlim(-35,5)
			axMain.set_ylim(-10,35)
		elif(whichSlice==2):
			axMain.set_xlim(-35,5)
			axMain.set_ylim(-40,5)
	
	
	
		axs[i].axis("off")
	
	legend_elements = [ Patch(facecolor=cmapToUse.colors[i], edgecolor='None', label=whichGroups[i]) for i in range(len(whichGroups))]
	axs[-3].legend(handles=legend_elements,loc=(-0.5,-0.2),ncols=5)
	

def plotMNIViz():
	fig,axs = plt.subplots(2,5,figsize=(14, 5))
	plt.subplots_adjust(wspace=0.05)
	
	iterateThroughSlices(axs.flatten(),whichSlice=1,nIter=10,whichContacts='AVCTviz')
	plt.savefig("figures/overview_y_AV_MD.pdf",bbox_inches='tight',dpi=300)
	plt.clf()

	fig,axs = plt.subplots(2,5,figsize=(14, 5))
	plt.subplots_adjust(wspace=0.05)
	
	iterateThroughSlices(axs.flatten(),whichSlice=1,nIter=10,whichContacts='hasSignal')
	plt.savefig("figures/overview_y.pdf",bbox_inches='tight',dpi=300)
	
plotMNIViz()


