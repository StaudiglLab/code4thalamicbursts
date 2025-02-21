import numpy as np
import pandas as pd


def getContactStats(infile='contacts_freesurfer.txt',
			structures=np.array(['Central','MDm','AV','VA','VP','VL']),
			maxDist=1.0):	
	#read csv and exclude Left contact of p26 which was not recorded	
	df = pd.read_csv(infile,sep=' ',header=0)
	selmask=np.logical_or(df['pID']!='p26',np.logical_and.reduce((df['contacts']!='tL1',df['contacts']!='tL2',df['contacts']!='tL3',df['contacts']!='tL4')))
		
	df=df[selmask]		
	print(len(df))
	for i in range(0,len(structures)):
		print("%s : %d contacts"%(structures[i],np.sum(df['%s_distNearest'%structures[i]].values<maxDist)))
		
getContactStats()
