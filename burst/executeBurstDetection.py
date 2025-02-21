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
import matplotlib.pyplot as plt

from core.params import *
#initial events detection

#loop over all subjects for iEEG
for pID in cohortForPaper:
	ch_names_allsel=['L1-L2','L2-L3','L3-L4','R1-R2','R2-R3','R3-R4']	
	for ch_name in ch_names_allsel:
		getMorletBurstsFromPeaks(pID=pID,ch_name=ch_name)

#loop over all subjects for scalp
for pID in ['p13']:
	if(pID=='p26' or pID=='p20'):
		ch_names_allsel=['F7','F8','T7','P7','T8','P8','O1','O2']
	else:
		ch_names_allsel=['Fp1','Fp2','F7','F8','T7','P7','T8','P8','O1','O2']
	for ch_name in ch_names_allsel:
		getMorletBurstsFromPeaks(pID=pID,ch_name=ch_name)


#select bursts


#loop over all subjects for iEEG

for pID in cohortForPaper:
	ch_names_allsel=['L1-L2','L2-L3','L3-L4','R1-R2','R2-R3','R3-R4']	
	for ch_name in ch_names_allsel:
		selectBurstsFromEvents(pID=pID,ch_name=ch_name)

#loop over all subjects for scalp
for pID in cohortForPaper:
	if(pID=='p26' or pID=='p20'):
		ch_names_allsel=['F7','F8','T7','P7','T8','P8','O1','O2']
	else:
		ch_names_allsel=['Fp1','Fp2','F7','F8','T7','P7','T8','P8','O1','O2']
	for ch_name in ch_names_allsel:
		selectBurstsFromEvents(pID=pID,ch_name=ch_name)




def getDetectedFrequencies(niter=int(1e6),outfigpdf="figures/bursts.pdf",outfile='outfiles/bursts_detectedFrequencies.txt'):
	pdf=PdfPages(outfigpdf)
	df = pd.DataFrame(columns=['pID','ch_name','state','freqPeak','freqLow','freqHigh','pvaluesLeft','pvaluesRight','pvaluesBoth','pvaluesMean'])
	rowIndx=0
	for pID in cohortForPaper:		
		for ch_name in ['L1-L2','L2-L3','L3-L4','R1-R2','R2-R3','R3-R4']:
			print(pID,ch_name)
			fig,freqlist= detectOscillation(pID,ch_name,niter=niter)
			print(freqlist)
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

	pdf.close()


def getDetectedFrequenciesOnScalp(outfile='outfiles/bursts_detectedFrequencies_scalp.txt'):
	df = pd.DataFrame(columns=['pID','ch_name','state','freqPeak','freqLow','freqHigh','pvaluesLeft','pvaluesRight','pvaluesBoth','pvaluesMean'])
	rowIndx=0

	
	ch_names_allsel=['Fp1','Fp2','F7','F8','T7','P7','T8','P8','O1','O2']
	for ch_name in ch_names_allsel:	
		pdf=PdfPages("figures/bursts_scalp_%s.pdf"%ch_name)
		for pID in cohortForPaper:
			if((ch_name=='Fp1' or ch_name=='Fp2') and (pID=='p26' or pID=='p20')):
				continue			
			fig,freqlist= detectOscillation(pID,ch_name,niter=int(1e6))
			#plt.show()			
			pdf.savefig(fig)
			plt.clf()	
			plt.close()

			for state in ['wake','REM','NREM']:
				nRows=len(freqlist[state])
				for i in range(0,nRows):
					df.loc[rowIndx]=[pID,ch_name,state]+freqlist[state][i].tolist()
					rowIndx+=1
		pdf.close()
	df.to_csv(outfile,sep=' ',index=False)
			
getDetectedFrequencies(niter=int(1e6))
getDetectedFrequenciesOnScalp()


