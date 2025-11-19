### File Descriptions

* calcPSD_timeresolved.py computes the time resolved Welch power spectrum and saves to disk. The output from here is used to check for oscillatory bumps. 

* calcMorlet.py computes the morlet power spectrum, averages over 1sec blocks, and saves to disk. Note that the output from here is used for visualization purposes only (e.g. Figure 1) and is _not_ used to detect bursts. The burst detection requires the Morlet at full temporal resolution and is computed on the fly (too large to dump to disk).

* getStateAveragedPowerSpectra.py computes the average Welch power spectrum for each patient, channel, state. This output is required to generate Panel d of Figure 2 and Extended Data Figure 1

#### For figures and results in the paper:

* figure1.py generates Figure 1 for the paper. Note this requires the raw data.

* plotEEGAndiEEGPSD.py generates extended data figure 1.
