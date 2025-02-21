import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import numpy as np
import mne
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

import nilearn 
from nilearn.plotting import plot_roi

from matplotlib.colors import  ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from coreFunctions import *

cmapToUse=ListedColormap(plt.cm.Dark2.colors[:7])

def setAxesLim(disp,size):
	
	disp.axes["x"].ax.set_xlim(10-size, 10+size)
	disp.axes["x"].ax.set_ylim(25-size, 25+size)
		
	disp.axes["z"].ax.set_xlim(-size, size)
	disp.axes["z"].ax.set_ylim(-size,size)
		
	return disp
	
def getThalamicParcellationForViz(imTh,labelTypes='Thal',whichGroups=np.array(['AV','Central']),hemi='LR'):
	if(labelTypes=='Group'):
		code=np.loadtxt("freesurferLabels.txt",usecols=[0]).astype("int")
		groups=np.loadtxt("freesurferLabels.txt",usecols=[6],dtype='str')
		

		imdata=nilearn.image.get_data(imTh)
		imdataNew=np.zeros_like(imdata,dtype='int32')

		for i in range(0,len(code)):
			if(np.sum(groups[i]==whichGroups)):
				imdataNew[imdata==code[i]]=groupCodes[groups[i]]
	elif(labelTypes=='Thal'):	
		imdata=nilearn.image.get_data(imTh)
		imdata[imdata<8100]=0
		imdataNew=np.zeros_like(imdata,dtype='int32')
		imdataNew[imdata>0]=1.0
	if(hemi=='L'):
		imdataNew[imdata>8200]=0
	elif(hemi=='R'):
		imdataNew[imdata<8200]=0
	return nilearn.image.new_img_like(imTh,imdataNew)
def vizCT(axs #matplotlib axes to draw slices on
	):
	cmapToUse=ListedColormap(['C0','C3'])	
	size=70
	coord=[1,12,11.5]
	
	#load MRI and thalamic parcellation of freesurfer example "bert")
	imgMRI = nilearn.image.load_img('bert/T1.mgz')		
	imgMRI=cropImage(imgMRI,0,size=size,interpolation='continuous')	
	imgThalParcellation=nilearn.image.load_img('bert/ThalamicNuclei.v13.T1.mgz')
	#get boundary of thalamus
	imgThalParcellationThal=getThalamicParcellationForViz(imgThalParcellation,labelTypes='Thal')
	#get CT and AV parcellation
	imgThalParcellation=getThalamicParcellationForViz(imgThalParcellation,labelTypes='Group')
	
	#crop images for quicker processing
	imgThalParcellation=cropImage(imgThalParcellation,0,size=size)
	imgThalParcellationThal=cropImage(imgThalParcellationThal,0,size=size)	

	#plot slices
	dispx=plot_roi(imgThalParcellation,axes=axs[0],bg_img=imgMRI,cut_coords=[coord[0]],black_bg=False, dim='auto',draw_cross=False,cmap=cmapToUse,linewidths=1.5,display_mode='x',annotate=False) #,view_type='contours'
	dispx.add_contours(imgThalParcellationThal,colors='black',linewidths=0.5)	
	
	dispz=plot_roi(imgThalParcellation,axes=axs[1],bg_img=imgMRI,cut_coords=[coord[2]],black_bg=False, dim='auto',draw_cross=False,cmap=cmapToUse,linewidths=1.5,display_mode='z',annotate=False) #,view_type='contours'
	dispz.add_contours(imgThalParcellationThal,colors='black',linewidths=0.5)

	axx=dispx.axes[coord[0]].ax
	axz=dispz.axes[coord[-1]].ax
	axz.axvline(coord[0],ls='--',c='black',lw=0.5)	
	axx.axhline(coord[-1],ls='--',c='black',lw=0.5)
	
	cropSize=20
	
	axx.set_xlim(5-cropSize, 5+cropSize)
	axx.set_ylim(10-cropSize, 10+cropSize)
	
	cropSize=25		
	axz.set_xlim(5-cropSize, 5+cropSize)
	axz.set_ylim(-cropSize,cropSize)

	
	axs[0].axis("off")
	axs[1].axis("off")
	#groupNames=['Central Thalamus','AntroVentral Thalamus (AV)']
	
	#legend_elements = [ Patch(facecolor=cmapToUse.colors[-1::-1][i], edgecolor='None', label=groupNames[i]) for i in range(len(groupNames))]
	#axs[1].legend(handles=legend_elements,loc='center',ncols=4)
	#plt.show()
	#plt.savefig("CT_viz_fill.png",bbox_inches='tight',dpi=300.0)


		
