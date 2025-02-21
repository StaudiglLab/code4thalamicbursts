import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)
from core import *
from core.helpers import *
import numpy as np
import pandas as pd
def saccadeSummary():
	sub=cohortForPaper
	print(len(sub))
	totalREM=np.zeros(len(sub))
	totalEM=np.zeros(len(sub))
        
	for i in range(0,len(sub)):
		saccadeParams=pd.read_csv(rootdir+"/eyeMovParams/%s_eyeMovEvents_alldetections.csv"%sub[i])
		sleepScoreAtSaccade=saccadeParams['sleepScore'].values.astype("int")
		totalEM[i]=np.sum(sleepScoreAtSaccade==5)
		taxis,sleepscore=readSleepScoreFinal(sub[i])
		totalREM[i]=np.sum(sleepscore==5)*30/60.

	print(np.min(totalREM),np.max(totalREM),np.mean(totalREM))
	print(np.min(totalEM),np.max(totalEM),np.mean(totalEM))
	print(np.min(totalEM/totalREM),np.max(totalEM/totalREM),np.mean(totalEM/totalREM),np.std(totalEM/totalREM))

saccadeSummary()
