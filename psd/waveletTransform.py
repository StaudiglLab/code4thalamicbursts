import numpy as np
from numba import njit
import numba


#The following is a custom code to compute wavelet transform. The default routines from mne was not used because it is much slower. 
#The code is heavily based on the MNE code with some sections copy pasted (including related comments)
@njit('complex128[:,:](float64[:],float64,float64[:],int32)',fastmath=True,parallel=True,nogil=True,cache=False)
def compute_wavelet_transform(sig, fs, freqs, n_cycles):
	scaling=0.5
	mwt = np.zeros((len(freqs), len(sig)), dtype=np.complex128)
	signalLength=len(sig)
	for ind in numba.prange(len(freqs)):
		sigma_t = n_cycles / (2.0 * np.pi * freqs[ind])
		# time vector. We go 5 standard deviations out to make sure we're
		# *very* close to zero at the ends. We also make sure that there's a
		# sample at exactly t=0
		t = np.arange(0.0, 5.0 * sigma_t, 1.0 / fs)
		wavelet_len_left=len(t)-1
		t = np.append(-t[-1::-1], t[1:])
		oscillation = np.exp(2.0 * 1j * np.pi * freqs[ind] * t)
		# this offset is equivalent to the κ_σ term in Wikipedia's
		# equations, and satisfies the "admissibility criterion" for CWTs
		real_offset = np.exp(-2 * (np.pi * freqs[ind] * sigma_t) ** 2)
		oscillation -= real_offset
		gaussian_envelope = 1/(sigma_t*np.sqrt(2*np.pi))*np.exp(-(t**2) / (2.0 * sigma_t**2))
		morlet_f = oscillation * gaussian_envelope
		#no additional frequency scaling
		#morlet_f =morlet_f*np.sqrt(2)* morlet_f /  (np.sqrt(np.sum(np.abs(morlet_f)**2)))
		
		wavelet_len=len(t)

		wavelet_len_right=wavelet_len-wavelet_len_left
		for j in range(wavelet_len_left,signalLength-wavelet_len_right): 
			startp=j-wavelet_len_left 	
			for k in range(0,wavelet_len):
				mwt[ind, j] = mwt[ind, j]+sig[startp+k]*morlet_f[k]    
		mwt[ind, :wavelet_len_left]=np.nan
		mwt[ind, -wavelet_len_right:]=np.nan			 		
	return mwt


