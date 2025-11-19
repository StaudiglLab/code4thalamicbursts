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
		
pIDLabelsForPaper={'p03':'p1','p05':'p2','p09':'p3','p13':'p4','p14':'p5','p14_followup':'p5 (follow up)',
	'p16':'p6','p18':'p7','p20':'p8','p21':'p9','p22':'p10','p26':'p11','p30':'p12',
	'pthal101':'p13','pthal102':'p14','pthal103':'p15','pthal104':'p16','pthal106':'p17'}

cohortForPaper=['p03','p05','p09','p13','p14','p14_followup',
	'p16','p18','p20','p21','p22','p26','p30',
	'pthal101','pthal102','pthal103','pthal104','pthal106']
	

sleepLabels=['Wake','Light','N2','SWS','REM']

sleepScoringList=[0,1,2,3,5]


		
