# EEG cross term calculation and features ranking
#
# This script increases the features by calculating the cross terms.
# It then performs an orthogonalization of Gram Schmidt in order to find the most representative 
# features of the dataset, and performs a classification. The table o_ranking_selected contains an 
# index of the representative features.
# Finally we end this dimensional reduction by selecting only the features of interest in the original 
# cross term dataset. This dataset will be used to feed the classifier.
#
# Since the Gram-Schmidt process takes (2*N_epoch*36)^2 FLOPS to complete, it is very advisable to slice 
# the initial features by taking just hundreds of epochs.
#
#
# Initial MATLAB creation Date : 28/04/2017
#   Updates and contributors :
#       16/03/2017 G. Dreyfus drafted the pseudocode for this function
#       28/04/2017 F.B. Vialatte created the first version of this code
#       19/06/2017 T. MEDANI correction on the implementation (get the right ranking) 
#
# Python porting carried out by: ROBALDO Axel for PI-Psy Institute
# Date: august 2020
#


import numpy as np
from sklearn import preprocessing
from tqdm import tqdm



def ofr_randperm(feats, labels, threshold):     # This function  ranks the features for machine learning, uses Gram-Schmidt Orthogonalization and probe variables
    ## Init
    n_features =  feats.shape[1]
    n_probe = n_features
    n_probe_ranked = 0

    indexing = list(range(1,n_features+n_probe))
    ranking_selected = np.zeros((1,2))

    #normalized_feature0 = feats - np.matlib.repmat(np.mean(feats), n_epochs, 1)
    normalized_feature0 = preprocessing.normalize(feats)                    # sklearn normalisation
    #normalized_feature = normalized_feature0 / np.tile(np.sqrt(np.sum(normalized_feature0**2)), (n_epochs, 0));

    output_vector = labels - np.mean(labels);
    #output_vector = labels

    ## OFR
    matrix_probe = normalized_feature0

    for l_feature in tqdm(range(n_features)):
        label_noise = np.random.permutation(n_epochs)
        matrix_probe[:,l_feature] = matrix_probe[label_noise, l_feature]

    feature_probe = np.concatenate((normalized_feature0, matrix_probe), 1)

    selecting = 1           #to come-in the OFR
    step = 1                #balayer l'ensemble des features probe
    n_pass = 0

    
    while selecting == 1:

        square_cosine = []
        
        for feature in tqdm(range(step, n_features+n_probe)):
            vector_value = feature_probe[:,feature].conj().T        # conjugate transpose
            numerator = np.dot(vector_value, output_vector)**2      # squared matrix multiplication
            denominator = np.dot(np.dot(vector_value,vector_value.conj().T),np.dot(output_vector.conj().T, output_vector))
            square_cosine.append(numerator/denominator)
        
        value = max(square_cosine)
        index = square_cosine.index(value)

        max_cos = value**(1/2)


        if (index <= n_features):
            feature_probe[:, [step]] = feature_probe[:, [index]] 

            if (max_cos > 0):
                indexing[step] = indexing[index]


            # Gram-Schmidt orthogonalization
            ranked = feature_probe[:,step] 
            print("\n Gram-Schmidt orthogonalization...")
            for feature in tqdm(range(step+1, n_features+n_probe)):
                vector = feature_probe[:,feature].conj().T
                feature_probe[:,feature] = vector - np.dot(np.dot(ranked.conj().T,vector.conj().T)/np.dot(ranked.conj().T,ranked),ranked.conj().T)

            output_vector = output_vector - np.dot(np.dot(ranked.conj().T,output_vector)/np.dot(ranked.conj().T,ranked),ranked)

            #step = step + 1;

        else:
            feature_probe = feature_probe[:, np.concatenate((np.asarray(list(range(1,index-1))),np.asarray(list(range(index+1,end)))),1)]
            indexing = indexing[:, np.concatenate((np.asarray(list(range(1,index-1))),np.asarray(list(range(index+1,end)))),1)]
            number_probe -= 1

        if (index > n_features):
            n_probe_ranked += 1
        else:
            item = np.array([[float(indexing[step]), float(max_cos)]])
            ranking_selected = np.concatenate((ranking_selected, item))
            step += 1

        if (n_pass > n_epochs) or (n_pass > n_features) or ((n_probe_ranked/n_features) > threshold):
            selecting = 0
    
        n_pass += 1
        print("n pass: " + str(n_pass))

    return ranking_selected





# Initialisation ------------------------------------------------------------------------------
feats_V2 = np.load("features_20190729144249_LEG04_V2_EEG.npy")[0:200]      # => label 0 (we only take the 500 first epochs)
feats_V3 = np.load("features_20190731101326_LEG01_V3_EEG.npy")[0:200]      # => label 1
#feats = np.load("features_tensor.npy")

feats = np.concatenate((feats_V2, feats_V3))                    # get one tensor from the two below
print("original features shape: " + str(feats.shape))

labels = np.concatenate((np.zeros(int(feats_V2.shape[0])).T,np.ones(int(feats_V3.shape[0])).T))     # create label vector

threshold = 0.1


# Preprocessing --------------------------------------------------------------------------------
feats_flatten = feats.reshape(feats.shape[0],-1)

print("flatten features shape: " + str(feats_flatten.shape))
print(feats_flatten)
feats_flatten[feats_flatten == np.inf] = 0



# Generating cross-terms -----------------------------------------------------------------------
n_epochs = feats_flatten.shape[0]
n_features = feats_flatten.shape[1]

n_cross = int(n_features + (((n_features-1)*n_features)/2)) +1
print(n_cross)

feats_cross = np.zeros((n_epochs,n_cross))
feats_cross[:,0:n_features] = feats_flatten[:,:]

index = -1 * np.ones((2,n_features**2))
index[0,0:n_features] = range(n_features)


cpt = n_features + 1

print("Calculating cross-terms...")
for i_feat1 in tqdm(range(n_features)):
    for i_feat2 in range(i_feat1+1, n_features):
        
        feat_x = feats_flatten[:,i_feat1]
        feat_y = feats_flatten[:,i_feat2]

        feats_cross[:,cpt] = np.multiply(feat_x,feat_y)

        index[0,cpt] = i_feat1
        index[1,cpt] = i_feat2

        cpt+=1


print("Cross terms generated...  cross terms shape: " + str(feats_cross.shape))
print("Entering OFR ranking features algorithm  until " + str(threshold) + " risk threshold is met")



# OFR ranking features algo --------------------------------------------------------------------
## Init base variable
NUMBER_SEED = 2
maximum_ranked = 0
average_ranked = 0

## Executing the Gram-Schimdt function NUMBER_SEED times
for i_seed in tqdm(range(NUMBER_SEED)):
    ranking_selected = ofr_randperm(feats_cross, labels, threshold)
    number_ranked = ranking_selected.shape[0]

    average_ranked = average_ranked + (number_ranked / NUMBER_SEED)

    if number_ranked > maximum_ranked:
        ranking = ranking_selected
        maximum_ranked = number_ranked

o_ranking_selected = ranking[1:round(average_ranked)]


print(o_ranking_selected)    
print(o_ranking_selected.shape)


# Extract most important features and serialize the new tensor ------------------------------------
output = np.empty((n_epochs,o_ranking_selected.shape[0]))
for i in range(o_ranking_selected.shape[0]):
    output[:,i] = feats_cross[:,int(o_ranking_selected[i,0])]

print(output)
print(output.shape)
np.save("ranking_selected400", o_ranking_selected)
np.save("ranked_cross_features400", output)


