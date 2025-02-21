import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

from core import *
from eyeMovDetectionCore import getEyeMovParamsFromScalp

for subID in cohortForPaper:
   if(subID=='p11'):
    	continue
   rawfname='C:/Aditya/thalamus-census'+'/data/rereferenced/%s/raw_%s_electrode_F7-F8_eeg.fif'%(subID,subID)
   figname="figures/%s_saccadeRate.png"%subID
   outfname=rootdir+"/eyeMovParams/%s_eyeMovEvents_alldetections.csv"%subID
   sleepscorefile=rootdir+"/sleepscore/final/%s_sleepscore.txt"%subID
   getEyeMovParamsFromScalp(rawfname=rawfname,figname=figname,sleepscorefile=sleepscorefile,outfname=outfname)



