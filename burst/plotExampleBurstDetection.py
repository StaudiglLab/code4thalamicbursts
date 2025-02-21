import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import numpy as np
import matplotlib.pyplot as plt



from plotHelpers import *

'''
NOTE: This section requires access to full raw data
extractMorletSegment('p21','R1-R2',tstart=2*3600+30,duration=25,fmin=7)
extractMorletSegment('p21','R1-R2',tstart=14.8*3600+260,duration=25,fmin=7)
extractMorletSegment('p21','R1-R2',tstart=16.0*3600+230,duration=25,fmin=7)
'''


fig,axs=plt.subplots(3,1,figsize=(9,6),sharex=True)
plt.subplots_adjust(hspace=0.2)
plotMorletSegment(axs[0],'p21','R1-R2',2*3600+30,25,fmin=7,tmax=20,titletext='')
plotMorletSegment(axs[1],'p21','R1-R2',14.8*3600+260,25,fmin=7,tmax=20,titletext='')
plotMorletSegment(axs[2],'p21','R1-R2',16.0*3600+230,25,fmin=7,tmax=20,titletext='')
plt.savefig("figures/edf4.pdf",bbox_inches='tight',dpi=600.0)

