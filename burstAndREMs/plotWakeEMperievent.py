import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git,sys
repo = git.Repo('.', search_parent_directories=True)
sys.path.append(repo.working_tree_dir)

import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec


from core import *
from core.helpers import *
from burst.coreFunctions import *
from coreFunctions import *

import matplotlib.gridspec as gridspec

import matplotlib

from figure3 import plotAveragePeriEvent
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 


def plotWakeCorrelations():	
		
	#fig = plt.figure(figsize=(14, 12))
	#fig.subplots_adjust(wspace=6.0,hspace=0.35)	

	fig,axs = plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=False)	
	fig.subplots_adjust(wspace=0.4,hspace=0.35)	
	
	
	#plot subject level curves
	
	axs[0].set_title("(a) wake eye movements")
	plotAveragePeriEvent(axs[0],state='wake')	
	axs[1].set_title("(b) REM eye movements")
	plotAveragePeriEvent(axs[1],state='REM')
	plotAveragePeriEvent(axs[1],state='wake',overlay=True)	
	plt.savefig("figures/wakeCorrelation.pdf",bbox_inches='tight',dpi=300)

plotWakeCorrelations()