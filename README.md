## Installation and Dependencies


Please clone the current repository (using git clone on linux or gitHub Desktop on windows). Simply downloading the .zip compressed file would not work due to path issues.

Dependencies:

* python (3.12.2) and the following python packages:
* MNE-Python (1.8.0)
* numpy (1.26.4)
* numba (0.60.0)
* matplotlib (3.9.2)
* scipy (1.14.1)
* pandas (2.2.3)
* seaborn (0.13.2)
* gitpython (3.1.43)
* joblib (1.4.2)
* fooof (1.26.4)
* scikit-learn (1.5.2)
* nilearn (0.10.4)
* h5py (3.12.1)

NOTE: The versions for the packages indicate those on which the codes were tested to run. However, the code is also expected to run with other versions of the packages.

The install time of the packages would vary with system but each installation is typically few tens of seconds.


## Code to generate figures of the manuscript


Following are the scripts for the figures in manuscript

### Main Text Figures
* figure 1 : psd/figure1.py  [NOTE: Running this requires full resolution raw data file to be present.]
* figure 2 : burst/figure2.py
* figure 3 : burstAndREMs/figure3.py
* figure 4 : localization/plotRegressionFigures.py

### Extended Data Figures
* edf 1 : psd/plotEEGAndiEEGPSD.py
* edf 2 : connectivity/connectivityPlotForPaper.py
* edf 3 : eyetrackingDataAnalysis/crossCorr.py
* edf 4 : burst/plotExampleBurstDetection.py
* edf 5 : eyeMovementDetection/topoAtEyeEvent.py
* edf 6 : eyetrackingDataAnalysis/eegSaccadeCheckPlot.py
* edf 7 : localization/plotRegressionFigures.py

### Supplementary Information Figures
* sup. fig. 1 : connectivity/connectivityPlotForPaper.py
* sup. fig. 2 : burstAmp/correlationFigures.py
* sup. fig. 3 : burstAmp/correlationFigures.py
* sup. fig. 4 : supplementaryAnalysis/seizureRateCorrelation.py 
* sup. fig. 5 : burst/plotN2N3rates.py 
* sup. fig. 6 : artefacts/plotIEDRates.py 
* sup. fig. 7 : supplementaryAnalysis/plotInfraslowPSD.py
* sup. fig. 8 : burstAndREMs/plotWakeEMperievent.py
* sup. fig. 9 : localization/MNIOverviewPlots.py
* sup. fig. 10 : localization/MNIOverviewPlots.py
* sup. fig. 10-46 : the generation of these figures require individual patient MRIs which cannot be uploaded.

## Demo of burst detection algorithm

For a demo of the burst detection algorithm, please run burst/demoDetection.py (this should not take more than a few minutes on a typical computer).
This would run the entire algorithm on 15m of raw data extracted from one of the patients in our cohort. The 15m data contains 5m each of wakefullness, REM, and NREM sleep, extracted from the raw thalamic iEEG recording of the patient.
The script would produce an output summary figure, "burst/figures/bursts_demo.pdf" showing the burst rate as a function of time and frequency (in colorscale). The timestamps and frequencies corresponding to individual bursts can be found at "dataFilesPublic/burstFromMorlet/demo_R1-R2_selected.txt" while the list of bands with significantly elevated burst probabilities can be found at "burst/outfiles/bursts_detectedFrequencies_demo_selected.txt".
The files already provided as part of the repository would serve as expected outputs for the script.
