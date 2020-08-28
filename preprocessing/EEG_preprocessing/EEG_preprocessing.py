# EEG data preprocessing
#
# Author: ROBALDO Axel for PI-Psy Institute
# Date: july 2020
#
# Description: 
# This algorithm preprocess EEG signals by proceeding to artifacts repair/rejection, epoching in desired fixed time
# duration and then filter by Cuda GPU compatible in 5 differents frequency bands. Metadata from acquisition are also 
# readen and used to get sample rate and channel names.
# Finally epochs are converted to classics Numpy 4D arrays.
#
#       MNE-Python package developped by:
# A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, 
# T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data analysis with MNE-Python, 
# Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X, [DOI]
#
#       Autoreject algorithm written by:
# Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort,
# "Automated rejection and repair of bad trials in MEG/EEG." In 6th International Workshop on
# Pattern Recognition in Neuroimaging (PRNI). 2016.
# Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017.
# Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.
############################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import tqdm
from autoreject import AutoReject, get_rejection_threshold


# Load the data and infos ----------------------------------------------------------------------------------------------------------------------------------
array_file = np.loadtxt(fname = "..\..\..\EEG_Pi_Psy/20190801135320_LEG02_V3_EEG recording.easy")     #load in numpy array
array_file = np.transpose(array_file)                                           #transpose to match shape (n_channel, n_time)

ch_names = []
unit = []
sfreq = []

with open ("..\..\..\EEG_Pi_Psy/20190801135320_LEG02_V3_EEG recording.info", 'rt') as myfile:         #parsing the info file to find channels name and smapling rate
    
    for myline in myfile:                               # For each line, read it to a string 
        if (myline.find("G sampling rate") > 0):              # For each string seek for...
            sfreq.append(myline[-19:-16])       
            print("sampling rate "+ str(sfreq))

        if  (myline.find("G units") > 0):
            unit.append(myline[-3:-1])
            print("unit "+ str(unit))

        if myline.find("Channel") > 0:
            ch_name = myline[-4:-1]
            ch_name = ch_name.replace(" ", "")                    #remove space from name string
            ch_names.append(ch_name)

sfreq = int(sfreq[0])
tEpoch = 0.5
nSampleEpoch = int(sfreq*tEpoch)+1

if unit[0] == 'µV':                                         #
    array_file = array_file/1e6
elif unit[0] == 'nV':
    array_file = array_file/1e9

info = mne.create_info(ch_names, sfreq, ch_types='eeg')     #creat info structure

raw = mne.io.RawArray(array_file[0:8,:], info)              #create Raw object from Numpy array


for ch in ch_names:                                         #loop to suppress empty channels
    data, times = raw[ch, :] 
    if data[0,0] == data[0,len(raw)-1]:                     #comparing first and last values
        raw.drop_channels(ch)
        print("channel " + ch + " dropped: no data")


montage_dir = os.path.join(os.path.dirname(mne.__file__), 'channels', 'data', 'montages')   # find the eloectrode montage folder
raw = raw.set_montage('standard_1020')                                                      # apply the montage from the standard_1020 library
print("corresponding montage found in 1020!")




# Data info and plot--------------------------------------------------------------------------------------------------------------------------------------------
print(raw)
print(raw.info)


#raw.plot(title='Original')            #observe original EEG data




# First epoching for ICA---------------------------------------------------------------------------------------------------------------------------------------------------------- 
raw.load_data()                                                         #we load the data in RAM in ordre to process it

filt_raw = raw.copy().filter(l_freq=1., h_freq=None)                    #high pass to suppress slow drifts

events = mne.make_fixed_length_events(filt_raw, start=0, duration=tEpoch)           # we fix here the period of 1 epoch
epochs = mne.Epochs(filt_raw, events, tmin=0.0, baseline=(None, None), tmax=tEpoch, preload=True, verbose=True)     # and we divide the signals in epoch

#epochs.plot(title='Epochs original')



# Repair artifacts w/ ICA-----------------------------------------------------------------------------------------------------------------------------------------
ica = mne.preprocessing.ICA(random_state=97, max_iter=1600)             # setup the ICA

reject = get_rejection_threshold(epochs)                                # get the threshold of the values (Autoreject fonction)

ica.fit(epochs, reject=reject, tstep=tEpoch)                            # fit ICA model with our datas
clean_raw = ica.apply(filt_raw)                                         # apply and get the cleaned raw data



# Main epoching-----------------------------------------------------------------------------------------------------------------------------------------------------
events = mne.make_fixed_length_events(clean_raw, start=0, duration=tEpoch)
epochs_clean = mne.Epochs(clean_raw, events, tmin=0.0, baseline=(None, None), tmax=tEpoch, preload=True, verbose=True)



# Bad epochs Autoreject---------------------------------------------------------------------------------------------------------------------------------------------
ar = AutoReject(random_state=42, verbose='tqdm')

epochs_clean = ar.fit_transform(epochs_clean)  

#epochs_clean.plot(title='Epochs clean')



# EEG filtering (5 ranges)----------------------------------------------------------------------------------------------------------------------------------------

delta = epochs_clean.copy().filter(l_freq=3, h_freq=4, n_jobs='cuda', method='fir')      #filtering executed by GPU
theta = epochs_clean.copy().filter(l_freq=4, h_freq=8, n_jobs='cuda', method='fir')
alpha = epochs_clean.copy().filter(l_freq=8, h_freq=12, n_jobs='cuda', method='fir')
beta = epochs_clean.copy().filter(l_freq=12, h_freq=25, n_jobs='cuda', method='fir')
gamma = epochs_clean.copy().filter(l_freq=25, h_freq=35, n_jobs='cuda', method='fir')

waves = [delta, theta, alpha, beta, gamma]
wavesTab = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

#for i in range(len(waves)):
    #waves[i].plot(title=wavesTab[i])



# OUTPUT TENSORS-------------------------------------------------------------------------------------------------------------------------------------------------

# First, convert filtred epochs values to a 3D numpy array shape(nsamples, nchannels, 5waves)
for i in range(len(waves)):                                     # for each waves object

    df = waves[i].to_data_frame()                               # we first convert the wave datas to a panda dataframe
    df = df.drop(columns=['condition'], axis=1)                 # we drop 3 not important columns
    df = df.drop(columns=['time'], axis=1)
    df = df.drop(columns=['epoch'], axis=1)
    df_array = np.asarray(df).astype(np.float32)                # we then convert the dataframe to a numpy array

    if i == 0:                                                  # debug conditions for the 2 firsts loop
        df_array1 = df_array
    elif i==1:
        tensor = np.dstack((df_array1, df_array))
    else:
        tensor = np.dstack((tensor, df_array))                  # we stacks the array along the 3rd dimension of the new array

                                                                # tensor.shape = [nsamples_total,nchannels,5waves]


# Then convert it to a more conveniant 4D array
tensor = np.reshape(np.transpose(tensor), (1, 5, len(ch_names), len(epochs_clean)*nSampleEpoch))      #1serie*5waves*n_sensors*n_samples_total


for i in range(len(epochs_clean)):                              # epoch differenciation
    
    epoch = tensor[:,:,:,(nSampleEpoch*i):(nSampleEpoch*(i+1))] # we select slices of 251 values (nSamples/Epochs)

    if i==0:
        epoch1 = epoch
    elif i==1:
        data_tensor = np.concatenate((epoch1, epoch))
    else:
        data_tensor = np.concatenate((data_tensor, epoch))      # concateante each epoch under the previous ones

                                                                # data_tensor.shape = [nepochs*5waves*4sensors*nsamples_per_epochs]

np.save("data_tensor_out", data_tensor)                         # finally, serialize the tensor to a .npy file

#plt.show()