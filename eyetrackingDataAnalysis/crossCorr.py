import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
import matplotlib
import seaborn as  sns
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=sns.color_palette("deep")) 

#calculate corss correlation as a function of lag
def calcCrossCorr(x,y,maxLag=30):
    lags=np.arange(-maxLag,maxLag+0.5).astype("int")
    crossCorr=np.zeros_like(lags,dtype=float)
    #iterate over all lag values (in samples)
    for i in range(0,len(lags)):
        lag=lags[i]
       
        if(lag>0):
            xThis=x[lag:]
            yThis=y[:-lag]
        elif(lag<0):
            xThis=x[:lag]
            yThis=y[-lag:]
        else:
            xThis=x
            yThis=y        
        xThis=(xThis-np.nanmean(xThis))/np.nanstd(xThis)
        yThis=(yThis-np.nanmean(yThis))/np.nanstd(yThis)
        crossCorr[i]=np.nanmean(xThis*yThis)
    return lags,crossCorr

#this functions ensures that the cross correlation coefficients are calculated seperately for each of the two segments
#this is because the two segments are not continuous
def calcCrossCorrSegments(dilation,
                          burstRate,
                          taxisSep,
                               maxLag=30):
    for i in range(0,len(taxisSep)-1):
        lags,crossCorr_=calcCrossCorr(dilation[taxisSep[i]:taxisSep[i+1]],burstRate[taxisSep[i]:taxisSep[i+1]],maxLag=maxLag)
        if(i==0):
            crossCorr=crossCorr_
        else:
            crossCorr+=crossCorr_
    return lags,crossCorr/(len(taxisSep)-1)

#get shuffled distributions
#also making sure that the two segments are treated seperately
def getShuffled(x,taxisSep):
    xShuffle=np.zeros_like(x)
    for i in range(0,len(taxisSep)-1):
        xThis=x[taxisSep[i]:taxisSep[i+1]]
        xThis=np.roll(xThis,np.random.randint(-len(xThis)//2,len(xThis)//2))
        xShuffle[taxisSep[i]:taxisSep[i+1]]=xThis
    return xShuffle



def doCrossCorrelationAnalysis(dilation,        #time series of dilations
                               burstRate,       #time series of burst rates
                               taxisSep,        #array containing discontineuties
                               maxLag=60,       #maximum lag in samples
                               nShuffle=1000    #number of permutations
                               ):
    
    lags,crossCorrTrue=calcCrossCorrSegments(dilation,burstRate,taxisSep=taxisSep,maxLag=maxLag)

    #get all shuffles
    crossCorrShuffle=np.zeros((nShuffle,len(lags)))
    for i in range(nShuffle):
        dilationShuffled=getShuffled(dilation,taxisSep)
        lags,crossCorrShuffle[i]=calcCrossCorrSegments(dilationShuffled,burstRate,taxisSep=taxisSep,maxLag=maxLag)
    return lags,crossCorrTrue,crossCorrShuffle

def plotOverlays(pID,axs,
                 whichMetric='pupil'):
    
    #read in the appropriate time series files
    burstRateAll=np.zeros(0)
    eyeMetricAll=np.zeros(0)
    taxisAll=np.zeros(0)
    taxisSep=[0]
    for session in [1,2]:
        taxis,eyeMetric=np.loadtxt("outfiles/%s_sessions%d_%s.csv"%(pID,session,whichMetric),delimiter=',',unpack=True) 
        taxis,burstRate=np.loadtxt("outfiles/%s_sessions%d_burstRate.csv"%(pID,session),delimiter=',',unpack=True)  
        taxis=taxis-taxis[0]
        burstRateAll=np.append(burstRateAll,burstRate)
        eyeMetricAll=np.append(eyeMetricAll,eyeMetric)        
        if(len(taxisAll)==0):
            taxisAll=taxis
        else:                
            taxisAll=np.append(taxisAll,taxis+taxisAll[-1]+taxis[1]-taxis[0])
        taxisSep.append(len(taxisAll)-1)

    
   
   
    #plot both the time series
    ax=axs[0]
    ax.plot((taxisAll-taxisAll[0])/60.,eyeMetricAll,label='Pupil size',c='C0',lw=1)
    ax2=ax.twinx()
    
    for sep in taxisSep[1:-1]:    
        ax.axvline(taxisAll[sep]/60.,c='gray',lw=4,zorder=1000)
        ax2.axvline(taxisAll[sep]/60.,c='gray',lw=4,zorder=1000)    
    ax2.plot((taxisAll-taxisAll[0])/60.,burstRateAll,label='Burst rate',c='C1',lw=1)
    
    ax.set_xlabel("Time (min)")
    if(whichMetric=='pupil'):
        ax.set_ylabel("Pupil diameter (mm)")
    else:
        ax.set_ylabel("saccade rate (/sec)")
        
    ax2.set_ylabel("Thalamic Burst rate (/sec)")
    ax2.yaxis.label.set_color('C1')
    ax.yaxis.label.set_color('C0')

    ax.set_xlim((0,(taxisAll[-1]-taxisAll[0])/60.))
    ax2.set_xlim((0,(taxisAll[-1]-taxisAll[0])/60.))   
    

    #remove any time period where the eyetracking data was not valid

    burstRateAll=burstRateAll[np.logical_not(np.isnan(eyeMetricAll))]
    eyeMetricAll=eyeMetricAll[np.logical_not(np.isnan(eyeMetricAll))]   


    #cross correlation analysis and shuffling 
    lags,crossCorrelation,crossCorrShuffle=doCrossCorrelationAnalysis(eyeMetricAll,burstRateAll,taxisSep,nShuffle=10000)   

    #compute and print p-value
    pvalue=np.mean(crossCorrShuffle[:,lags==0]>crossCorrelation[lags==0][0])
    print(crossCorrShuffle.shape)
    print("lag-zero correlation:%.3f"%crossCorrelation[lags==0][0])
    print("pvalues:%.3f"%pvalue)

    #plot the cross correlation vs. lag curve with s.e.m.
    lags=lags*(taxisAll[1]-taxisAll[0])
    axs[1].plot(lags,crossCorrelation,lw=3,c='C3')
    upperRange=np.sort(crossCorrShuffle,axis=0)[int(len(crossCorrShuffle)*(0.5+0.95/2.))]
    lowerRange=np.sort(crossCorrShuffle,axis=0)[int(len(crossCorrShuffle)*(0.5-0.95/2.))]

    axs[1].fill_between(lags,
                          upperRange,
                          lowerRange,
                          fc='gray',alpha=0.3)
    axs[1].legend(fontsize=8,framealpha=0.9,loc='lower right')
    axs[1].set_ylabel("Cross-correlation coefficient")
    axs[1].axvline(0,ls='--',c='black')
    axs[1].axhline(0,ls='--',c='black')
    axs[1].set_ylim([np.nanmin(lowerRange)*1.2,np.nanmax(crossCorrelation)*1.1])
    axs[1].set_xlabel("Lag (seconds)")
    axs[1].set_xlim([np.min(lags),np.max(lags)])

   
#function for getting the plot in paper
def plotForPaper(whichMetric):
    fig = plt.figure(layout="constrained",figsize=(14*0.7,2*3.5*0.7))
    #fig.suptitle("%s"%(pID))
    
    axs = fig.subplot_mosaic(mosaic='AAAABB\nCCCCDD')
    axs['A'].set_title("P15")
    axs['B'].set_title("P15")

    axs['C'].set_title("P17")
    axs['D'].set_title("P17")

    axs['A'].text(0.02,0.9,'(a)',transform=axs['A'].transAxes, fontweight='bold')
    axs['B'].text(0.02,0.9,'(b)',transform=axs['B'].transAxes, fontweight='bold')
    axs['C'].text(0.02,0.9,'(c)',transform=axs['C'].transAxes, fontweight='bold')
    axs['D'].text(0.02,0.9,'(d)',transform=axs['D'].transAxes, fontweight='bold')

    fig.subplots_adjust(wspace=0.1)
    plotOverlays('pthal103',
                 whichMetric=whichMetric,axs=[axs['A'],axs['B']])
    plotOverlays('pthal106b',
                 whichMetric=whichMetric,axs=[axs['C'],axs['D']])

    plt.savefig("figures/Chowdhury_EDF3.jpg",transparent=False,bbox_inches='tight',dpi=300.0)
    #plt.savefig("figures/%s.png"%whichMetric,transparent=False,bbox_inches='tight',dpi=300.0)

plotForPaper('pupil')

#uncomment these to get plots with saccade rates
#plotForPaper('saccadeRateTobii')
#plotForPaper('saccadeRateEEG')
