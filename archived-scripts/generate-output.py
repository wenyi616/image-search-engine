#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:41:45 2019

@author: wenyi
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_float
    


##########################################################
#     load pred result
##########################################################
    
castle = pd.read_csv('./data/features_test/testing-image-feature-2.csv', sep=",", header=None)
haha = pd.read_csv('./output/ridge_pred.csv', sep=",", header=None)

samples = np.array(castle)
pred = np.array(haha)

print(samples.shape)
#print(samples)

print(pred.shape)
#print(pred)



##########################################################
#      KNN-regressor model to predict 20 neareast images
##########################################################
knn = NearestNeighbors(n_neighbors=20)
knn.fit(samples)
result = knn.kneighbors(pred)
print(np.array(result)[1])


# result[0] - distances
# result[1] - index
np.savetxt("./output/ridge_output.csv", np.array(result)[1], delimiter=",")


##########################################################
#      display description and images
##########################################################
text_df = pd.read_csv('./test_des.csv', header=None)
temp = np.array(result)[1]


#for i in text_df.iterrows(): 
for i in range(10):
    print("\n" + str(i))
    print(" ".join(text_df.iloc[i,:1]))    
    
    fig, axes = plt.subplots(5,4,figsize=(10,10)) 
    for j in range(20):
        index = int(temp[i][j]) #0-first 1-second....
        path = "./data/images_test/"+ str(index) + ".jpg"
    
        img = img_as_float(mpimg.imread(path))
        ax = axes[j//4, j%4]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
        
    plt.show()

    
