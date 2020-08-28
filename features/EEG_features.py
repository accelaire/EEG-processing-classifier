# EEG features extractor
#
# Author: ROBALDO Axel for PI-Psy Institute
# Date: august 2020
#
# Description: 
# This programm extract several time and frequential features from preprocessed data tensor obtained via 
# EEG_preprocessing script. 
# Also, this programm extract the frequency sprectrum and Morlet continuous wavelets representation from 
# preprocessed data tensor obtained via EEG_preprocessing script. The extracted features are statistical
# and mathematical descritors of the distribution of the signal and it's spectrum. We then proceed to a 
# shape modelisaton of the wavelet heat map in order to find "hotspots". Descriptors are geometrical 
# properties of the shape.
#
# EntroPy was created and is maintained by Raphael Vallat.
# Bradski, G. (2000). The OpenCV Library. Dr. Dobb&#39;s Journal of Software Tools.


import numpy as np
import scipy as sp
import cv2
import io
import time
import itertools
from entropy import katz_fd
from tqdm import tqdm
import matplotlib.pyplot as plt



# Useful constants ----------------------------------------------------------------------------------
dt = 0.002
fs = int(1/dt)
epoch_duration = 0.5
n_per_epochs = int(epoch_duration*fs)+1             # number of samples per epochs
freq = np.linspace(1, fs/8, 100)           # we set a frequency axis of n samples (that fix the length of the width parameter and the shapes of returned wavelet matrix)               
t = np.linspace(0,0.5,n_per_epochs)
c = 3e8

w = 5                                                # wavelets omega parameter
width = w*fs /(2*freq*np.pi)


data_tensor = np.load("data_20190731101326_LEG01_V3_EEG.npy")
feat_tensor = np.empty((data_tensor.shape[0], data_tensor.shape[1], data_tensor.shape[2], 36), dtype = np.float32)  #preallocate tensor of the good shape



print("Extracting features on " + str(data_tensor.shape[0]) + " epochs...")
#start = time.time()
for epoch in tqdm(np.ndindex(data_tensor.shape[0])):                  # for each epoch
    for wave in np.ndindex(data_tensor.shape[1]):               # for each wave
        for channel in np.ndindex(data_tensor.shape[2]):        # for each channel
            
            values = data_tensor[epoch[0], wave[0], channel[0], :]          # take all the values (250)
            diff_values = np.diff(values)                                   # derivate the values to find leading coefficient between each values
            enveloppe = np.abs(sp.signal.hilbert(values))                   # find the hilbert enveloppe
            

            # Time and stats
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
             
            

            # Fourier
            spectrum = sp.fft.fft(values)                                       # we process a 1D fft
            freq_axe = sp.fft.fftfreq(len(spectrum), dt)                        # then we set the frequency axis
            peak_freq = abs(freq_axe[abs(spectrum) == max(abs(spectrum))][0])   # we find which frequency belong to the highest fft value
            
            feat_tensor[epoch[0], wave[0], channel[0], 11] = (c/peak_freq)      # finally we obtain the wavelength by dividing c by this frequency

            feat_tensor[epoch[0], wave[0], channel[0], 12]  = np.mean(abs(spectrum))             #mean
            feat_tensor[epoch[0], wave[0], channel[0], 13]  = sum(abs(spectrum)**2.0)*dt         #energy
            feat_tensor[epoch[0], wave[0], channel[0], 14]  = sp.stats.kurtosis(abs(spectrum))   #kurtosis
            feat_tensor[epoch[0], wave[0], channel[0], 15]  = sp.stats.skew(abs(spectrum))       #skewness
            feat_tensor[epoch[0], wave[0], channel[0], 16]  = katz_fd(abs(spectrum))             #fractal dimension
            


            # Morlet wavelet   
            cwtm = sp.signal.cwt(values, sp.signal.morlet2, width, w = w)
            
            ax = plt.axes([0,0,1,1], frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.autoscale(tight=True)
            plt.pcolormesh(t, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')

            
            buf = io.BytesIO() 
            plt.savefig(buf, format='png')
            plt.cla()
            buf.seek(0)
            img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 0)
            buf.close()

            
            ret,th = cv2.threshold(img, 127, 255, 0)                        # First obtain the threshold using the greyscale image
            contours, hierarchy = cv2.findContours(th, 2, 1)                # Find all the contours in the binary image
            
            big_contour = []
            max_area = 0
            for i in contours:
                area = cv2.contourArea(i)                   # Find the contour having biggest area
                if(area > max_area):
                    max_area = area
                    big_contour = i 


            X,Y,W,H = cv2.boundingRect(big_contour)         # Extract properties from it
            (xx,yy),(MA,ma),angle = cv2.fitEllipse(big_contour)
            rect_area = W*H
            area = cv2.contourArea(big_contour)
            hull = cv2.convexHull(big_contour)
            hull_area = cv2.contourArea(hull)
            
            mask = np.zeros(img.shape, np.uint8)
            cv2.drawContours(mask,[big_contour],0,255,-1)

            feat_tensor[epoch[0], wave[0], channel[0], 17] = area
            feat_tensor[epoch[0], wave[0], channel[0], 18] = hull_area
            feat_tensor[epoch[0], wave[0], channel[0], 19] = float(W)/H                             #aspect ratio
            feat_tensor[epoch[0], wave[0], channel[0], 20] = rect_area                              #rect area
            feat_tensor[epoch[0], wave[0], channel[0], 21] = float(area)/rect_area                  #extent
            feat_tensor[epoch[0], wave[0], channel[0], 22] = float(area)/hull_area                 #solidity
            feat_tensor[epoch[0], wave[0], channel[0], 23] = np.sqrt(4*area/np.pi)                 #equi diameter
            feat_tensor[epoch[0], wave[0], channel[0], 24] = MA                                    #major axe
            feat_tensor[epoch[0], wave[0], channel[0], 25] = ma                                    #minor axe
            feat_tensor[epoch[0], wave[0], channel[0], 26] = angle                                 #orientation
            feat_tensor[epoch[0], wave[0], channel[0], 27] = cv2.mean(img, mask = mask)[0]         #average intensity inside the shape
            feat_tensor[epoch[0], wave[0], channel[0], 28] = tuple(big_contour[big_contour[:,:,0].argmin()][0])[0]    # X leftmost point
            feat_tensor[epoch[0], wave[0], channel[0], 29] = tuple(big_contour[big_contour[:,:,0].argmin()][0])[1]    # Y leftmost point
            feat_tensor[epoch[0], wave[0], channel[0], 30] = tuple(big_contour[big_contour[:,:,0].argmax()][0])[0]    # X rightmost point
            feat_tensor[epoch[0], wave[0], channel[0], 31] = tuple(big_contour[big_contour[:,:,0].argmax()][0])[1]    # Y rightmost point
            feat_tensor[epoch[0], wave[0], channel[0], 32] = tuple(big_contour[big_contour[:,:,1].argmin()][0])[0]    # X topmost point
            feat_tensor[epoch[0], wave[0], channel[0], 33] = tuple(big_contour[big_contour[:,:,1].argmin()][0])[1]    # Y topmost point
            feat_tensor[epoch[0], wave[0], channel[0], 34] = tuple(big_contour[big_contour[:,:,1].argmax()][0])[0]    # X bottommost point
            feat_tensor[epoch[0], wave[0], channel[0], 35] = tuple(big_contour[big_contour[:,:,1].argmax()][0])[1]    # Y bottommost point
            
            
            #final = cv2.drawContours(img, big_contour, -1, (0,255,0), 3)
            #cv2.imshow('final', final)
            #cv2.waitKey(0) 
            #cv2.destroyAllWindows()
    
    #pc = ((epoch[0]+1)/data_tensor.shape[0])*100
    #print("epoch " + str(epoch[0]+1) + " on " + str(data_tensor.shape[0]) + " OK") 
    #print("  => " + str(pc) + "%")


np.save("features_tensor", feat_tensor)             # serialize the tensor

print(feat_tensor)
#print("--- %s total extraction ---" % (time.time() - start))

