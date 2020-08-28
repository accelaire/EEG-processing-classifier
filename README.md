# EEG-processing-classifier
Development of statistical learning algorithms for electroencephalographic signal processing.
Some script are provided with GUI.


1. Preprocessing:
    Process EEG signal automatic repair on Neuroelectric Enobio 8 channels EEG amplifier (see EEG_exemples) thank to MNE and Autoreject tools.
    Epoch the signal to a fixed window time, then filters the signal in interesting observations frequency bands (delta, theta, alpha, beta, gamma).
    Finally put the data into a convenient ndarray.
    
2. Features extraction:   (reunite both time_features and freq_features exctractors)
    This programm extract several time and frequential features from preprocessed data tensor obtained via EEG_preprocessing script. 
    Also, this programm extract the frequency sprectrum and Morlet continuous wavelets representation from preprocessed data tensor obtained via EEG_preprocessing script. The extracted features are statistical and mathematical descritors of the distribution of the signal and it's spectrum. We then proceed to a shape modelisaton of the wavelet heat map in order to find "hotspots". Descriptors are geometrical properties of the shape.
    
3. Cross terms calculation and orthogonal forward regression:
    This script increases the features by calculating the cross terms.
    It then performs an orthogonalization of Gram Schmidt in order to find the most representative features of the dataset, and performs a classification. The table o_ranking_selected contains an index of the representative features.
  Finally we end this dimensional reduction by selecting only the features of interest in the original 
  cross term dataset. This dataset will be used to feed the classifier.
  Since the Gram-Schmidt process takes (2*N_epoch*36)^2 FLOPS to complete, it is very advisable to slice the initial features by taking just hundreds of epochs.
    
4. Classification
  Thanks to sklearn library, we perform LDA, QDA and SVM classification test performance.
