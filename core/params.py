import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)

import os

'''
if (os.name=='posix'):
	if(os.uname().nodename=='klara'):
		rootdir='/media/data/chowdhury/thalamus-census/'
	else:
		rootdir='/media/10A/Aditya/thalamus-census/'	
else:
	rootdir='/Aditya/thalamus-census/'
'''

rootdir=repo.working_tree_dir+'/dataFilesPublic/'
		
cohortForPaper=['p03','p05','p09','p13','p14','p14_followup',
	'p16','p18','p20','p21','p22','p26','p30',
	'pthal101','pthal102','pthal103','pthal104','pthal106']
	

sleepLabels=['Wake','Light','N2','SWS','REM']

sleepScoringList=[0,1,2,3,5]


		
