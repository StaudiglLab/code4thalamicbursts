import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
from detectFromMorlet import getMorletBurstsFromPeaks
from selectBursts import selectBurstsFromEvents
from detectBandsInStates import detectOscillation
from getSelectedBands import getSelection
import matplotlib.pyplot as plt

from core.params import *
#initial events detection

#loop over all subjects for iEEG
for pID in ['demo']:
	ch_names_allsel=['R1-R2']	
	for ch_name in ch_names_allsel:
		getMorletBurstsFromPeaks(pID=pID,ch_name=ch_name)

#select bursts

#loop over all subjects for iEEG

for pID in ['demo']:
	ch_names_allsel=['R1-R2']	
	for ch_name in ch_names_allsel:
		selectBurstsFromEvents(pID=pID,ch_name=ch_name)



def getDetectedFrequencies(niter=int(1e6),outfigpdf="figures/bursts_demo.pdf",outfile='outfiles/bursts_detectedFrequencies_demo.txt'):
	pdf=PdfPages(outfigpdf)
	df = pd.DataFrame(columns=['pID','ch_name','state','freqPeak','freqLow','freqHigh','pvaluesLeft','pvaluesRight','pvaluesBoth','pvaluesMean'])
	rowIndx=0
	for pID in ['demo']:		
		for ch_name in ['R1-R2']:
			print(pID,ch_name)
			fig,freqlist= detectOscillation(pID,ch_name,niter=niter)
			#print(freqlist)
			#plt.show()			
			pdf.savefig(fig)
			plt.clf()	
			plt.close()

			for state in ['wake','REM','NREM']:
				nRows=len(freqlist[state])
				for i in range(0,nRows):
					df.loc[rowIndx]=[pID,ch_name,state]+freqlist[state][i].tolist()
					rowIndx+=1
	df.to_csv(outfile,sep=' ',index=False)	
	print("-----")
	print("writing detected bands to %s"%outfile)
	print("summary figure in %s"%outfigpdf)	
	print("-----")	
	pdf.close()



getDetectedFrequencies()
getSelection(infile='outfiles/bursts_detectedFrequencies_demo.txt',outfile='outfiles/bursts_detectedFrequencies_demo_selected.txt')

