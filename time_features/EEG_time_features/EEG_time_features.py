# EEG time features extractor
#
# Author: ROBALDO Axel for PI-Psy Institute
# Date: july 2020
#
# Description: 
# This programm extract several time features from preprocessed data tensor obtained via 
# EEG_preprocessing script. The resulting features are stored in another features tensor 
# which is finally serialized to .npy file.
#
# EntroPy was created and is maintained by Raphael Vallat.


import numpy as np
import scipy as sp
import itertools
from entropy import katz_fd



dt = 0.002              # sample rate of the original datas
c = 3e8

data_tensor = np.load("data_tensor_out.npy")[1:31,:]
feat_tensor = np.empty((data_tensor.shape[0], data_tensor.shape[1], data_tensor.shape[2], 12), dtype = np.float32)  #preallocate tensor of the good shape


for epoch in np.ndindex(data_tensor.shape[0]):                  # for each epoch
    for wave in np.ndindex(data_tensor.shape[1]):               # for each wave
        for channel in np.ndindex(data_tensor.shape[2]):        # for each channel
            
            values = data_tensor[epoch[0], wave[0], channel[0], :]          # take all the values (250)
            diff_values = np.diff(values)                                                               # derivate the values to find leading coefficient between each values
            enveloppe = np.abs(sp.signal.hilbert(values))                                               # find the hilbert enveloppe
            

            feat_tensor[epoch[0], wave[0], channel[0], 0]  = np.mean(values)                             #mean
            feat_tensor[epoch[0], wave[0], channel[0], 1]  = sum(abs(values)**2.0)*dt                    #energy
            feat_tensor[epoch[0], wave[0], channel[0], 2]  = sp.stats.kurtosis(values)                   #kurtosis
            feat_tensor[epoch[0], wave[0], channel[0], 3]  = sp.stats.skew(values)                       #skewness
            feat_tensor[epoch[0], wave[0], channel[0], 4]  = katz_fd(values)                             #katz fractal
                        
            
            feat_tensor[epoch[0], wave[0], channel[0], 5]  = np.mean(enveloppe)                          #enveloppe mean
            feat_tensor[epoch[0], wave[0], channel[0], 6]  = np.std(enveloppe)                           #enveloppe standard deviation
            feat_tensor[epoch[0], wave[0], channel[0], 7]  = sp.stats.kurtosis(enveloppe)                #enveloppe kurtosis
            feat_tensor[epoch[0], wave[0], channel[0], 8]  = sp.stats.skew(enveloppe)                    #enveloppe skewness
            

            feat_tensor[epoch[0], wave[0], channel[0], 9]  = ((values[:-1] * values[1:]) < 0).sum()      #zero crossing rate: we summing 1 each time a value cross the horizontal axis
            feat_tensor[epoch[0], wave[0], channel[0], 10] = len(list(itertools.groupby(diff_values, lambda diff_values: diff_values > 0)))    #slope changes: we find each sign change in the 1st degree derivate of the values
             
            
            spectrum = sp.fft.fft(values)                                   # we process a 1D fft
            freq = sp.fft.fftfreq(len(spectrum), dt)                        # then we set the frequency axis
            peak_freq = abs(freq[abs(spectrum) == max(abs(spectrum))][0])   # we find which frequency belong to the highest fft value
            
            feat_tensor[epoch[0], wave[0], channel[0], 11] = (c/peak_freq)  # finally we obtain the wavelength by dividing c by this frequency


np.save("time_features_tensor", feat_tensor)             # serialize the tensor


print(feat_tensor)
print(feat_tensor.shape)
