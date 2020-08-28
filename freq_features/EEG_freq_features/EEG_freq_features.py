# EEG frequency features extractor
#
# Author: ROBALDO Axel for PI-Psy Institute
# Date: july 2020
#
# Description: 
# This programm extract the frequency sprectrum and Morlet continuous wavelets representation from 
# preprocessed data tensor obtained via EEG_preprocessing script. The extracted features are statistical
# and mathematical descritors of the distribution of the spectrum. We then proceed to a shape modelisaton
# of the wavelet heat map in order to find "hotspots". Descriptors are geometrical properties of the shape.
# Tensor is finally serialized to .npy file.
#
# EntroPy was created and is maintained by Raphael Vallat.



import numpy as np
import scipy as sp
import cv2
import io
import time
from scipy import signal, stats
from entropy import katz_fd
import matplotlib.pyplot as plt
#import vispy.mpl_plot as plt            # (version 0.4.0) matplotlib on cuda, offers no significant acceleration




# useful constants----------------------------
dt = 0.002
fs = int(1/dt)
epoch_duration = 0.5
n_per_epochs = int(epoch_duration*fs)+1             # number of samples per epochs
freq = np.linspace(1, fs/8, 100)           # we set a frequency axis of n samples (that fix the length of the width parameter and the shapes of returned wavelet matrix)               
t = np.linspace(0,0.5,n_per_epochs)

w = 5                                                # wavelets omega parameter
width = w*fs /(2*freq*np.pi)


data_tensor = np.load("data_tensor_out.npy")[1:11,:]        # load input data tensor

freq_feat_tensor = np.empty((data_tensor.shape[0], data_tensor.shape[1], data_tensor.shape[2], 24), dtype = np.float32)     #preallocate the features container (24 descriptors)

n_epochs = data_tensor.shape[0]
start = time.time()

for epoch in np.ndindex(data_tensor.shape[0]):                  # for each epoch
    for wave in np.ndindex(data_tensor.shape[1]):               # for each band
        for channel in np.ndindex(data_tensor.shape[2]):        # for each channel
            
            values = data_tensor[epoch[0], wave[0], channel[0], :]                              # take all the values (251) from the epoch

            # Fourier
            spectrum = abs(sp.fft.fft(values))

            freq_feat_tensor[epoch[0], wave[0], channel[0], 0]  = np.mean(spectrum)             #mean
            freq_feat_tensor[epoch[0], wave[0], channel[0], 1]  = sum(spectrum**2.0)*dt         #energy
            freq_feat_tensor[epoch[0], wave[0], channel[0], 2]  = sp.stats.kurtosis(spectrum)   #kurtosis
            freq_feat_tensor[epoch[0], wave[0], channel[0], 3]  = sp.stats.skew(spectrum)       #skewness
            freq_feat_tensor[epoch[0], wave[0], channel[0], 4]  = katz_fd(spectrum)             #fractal dimension
            


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
            max = 0
            for i in contours:
                area = cv2.contourArea(i)                   # Find the contour having biggest area
                if(area > max):
                    max = area
                    big_contour = i 


            X,Y,W,H = cv2.boundingRect(big_contour)         # Extract properties from it
            (xx,yy),(MA,ma),angle = cv2.fitEllipse(big_contour)
            rect_area = W*H
            area = cv2.contourArea(big_contour)
            hull = cv2.convexHull(big_contour)
            hull_area = cv2.contourArea(hull)
            
            mask = np.zeros(img.shape, np.uint8)
            cv2.drawContours(mask,[big_contour],0,255,-1)

            freq_feat_tensor[epoch[0], wave[0], channel[0], 5] = area
            freq_feat_tensor[epoch[0], wave[0], channel[0], 6] = hull_area
            freq_feat_tensor[epoch[0], wave[0], channel[0], 7] = float(W)/H                             #aspect ratio
            freq_feat_tensor[epoch[0], wave[0], channel[0], 8] = rect_area                              #rect area
            freq_feat_tensor[epoch[0], wave[0], channel[0], 9] = float(area)/rect_area                  #extent
            freq_feat_tensor[epoch[0], wave[0], channel[0], 10] = float(area)/hull_area                 #solidity
            freq_feat_tensor[epoch[0], wave[0], channel[0], 11] = np.sqrt(4*area/np.pi)                 #equi diameter
            freq_feat_tensor[epoch[0], wave[0], channel[0], 12] = MA                                    #major axe
            freq_feat_tensor[epoch[0], wave[0], channel[0], 13] = ma                                    #minor axe
            freq_feat_tensor[epoch[0], wave[0], channel[0], 14] = angle                                 #orientation
            freq_feat_tensor[epoch[0], wave[0], channel[0], 15] = cv2.mean(img, mask = mask)[0]         #average intensity inside the shape
            freq_feat_tensor[epoch[0], wave[0], channel[0], 16] = tuple(big_contour[big_contour[:,:,0].argmin()][0])[0]    # X leftmost point
            freq_feat_tensor[epoch[0], wave[0], channel[0], 17] = tuple(big_contour[big_contour[:,:,0].argmin()][0])[1]    # Y leftmost point
            freq_feat_tensor[epoch[0], wave[0], channel[0], 18] = tuple(big_contour[big_contour[:,:,0].argmax()][0])[0]    # X rightmost point
            freq_feat_tensor[epoch[0], wave[0], channel[0], 19] = tuple(big_contour[big_contour[:,:,0].argmax()][0])[1]    # Y rightmost point
            freq_feat_tensor[epoch[0], wave[0], channel[0], 20] = tuple(big_contour[big_contour[:,:,1].argmin()][0])[0]    # X topmost point
            freq_feat_tensor[epoch[0], wave[0], channel[0], 21] = tuple(big_contour[big_contour[:,:,1].argmin()][0])[1]    # Y topmost point
            freq_feat_tensor[epoch[0], wave[0], channel[0], 22] = tuple(big_contour[big_contour[:,:,1].argmax()][0])[0]    # X bottommost point
            freq_feat_tensor[epoch[0], wave[0], channel[0], 23] = tuple(big_contour[big_contour[:,:,1].argmax()][0])[1]    # Y bottommost point
            
            
            final = cv2.drawContours(img, big_contour, -1, (0,255,0), 3)
            cv2.imshow('final', final)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
    
    pc = ((epoch[0]+1)/n_epochs)*100
    print("epoch " + str(epoch[0]+1) + " on " + str(n_epochs) + " OK") 
    print("  => " + str(pc) + "%")


np.save("freq_features_tensor", freq_feat_tensor)             # serialize the tensor

print(freq_feat_tensor)
print("--- %s total extraction ---" % (time.time() - start))
#print(feat_tensor.shape)
