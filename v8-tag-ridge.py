#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 21:21:04 2019

@author: wenyi
"""


import numpy as np
import pandas as pd

import re, string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_float




def load_tag(isTrain):
    all_tag = []
    tag_for_one_image = []
    if isTrain:
        folder_path = "data/tags_train/"
        num = 10000
    else:
        folder_path = "data/tags_test/"
        num = 2000
        
    for n in range(num):
        path = folder_path + str(n) + ".txt"
        txtfile = open(path, "r")
        lines = txtfile.read().split('\n')
        # format: vehicle:airplane   
        temp = ''
        cat = ''
        for line in lines:
            words = line.split(':')
            if words[0] == '':
                break
            tags = words[1]
            cat = cat + ' '
            tags = tags.replace(" ", "")
            temp = temp + ' ' + tags
            all_tag.append(tags)
        tag_for_one_image.append(temp)
        
    all_tag = list(set(all_tag))
    
    return tag_for_one_image, all_tag

#training word to vector
tag2image_train, tag_train = load_tag(True)  #tag_train length:80
cv_train = CountVectorizer(vocabulary = tag_train)
tags_train_1 = cv_train.fit_transform(tag2image_train).toarray() #10000*80

#testing word to vector
tag2image_test, tag_test = load_tag(False)   
cv_test = CountVectorizer(vocabulary = tag_train) 
tags_test_1 = cv_test.fit_transform(tag2image_test).toarray() #2000*80
print(tags_train_1.shape)
print(tags_test_1.shape)
print("****** Done loading training/testing tag data ******")


##########################################################
#     Descrptions features (data)
##########################################################
#training_data = load_description_data("train", 10000)
#testing_data = load_description_data("test", 2000)

training_data_df = pd.read_csv('./train_des.csv',  header=None)
testing_data_df = pd.read_csv('./test_des.csv', header=None)

training_data = []
testing_data = []

for row in training_data_df.iterrows():
    training_data.append(" ".join(row[1]))
for row in testing_data_df.iterrows():
    testing_data.append(" ".join(row[1]))

training_data = np.array(training_data)
testing_data = np.array(testing_data)

print(training_data.shape)
print(testing_data.shape)
print("****** Done loading training/testing description data ******")



# Bag of Words Model 
word_dict = {}
word_voc = []

# iterate thru all reviews in the training set
for i in range(len(training_data)):
    review = training_data[i]
    
    for w in word_tokenize(review):
        if w not in word_dict:
            word_voc.append(w)
            word_dict[w] = 0

print(len(word_dict)) #6719
print(len(word_voc))
print("****** Done building word dictionary ******")


# iterate all reviews in both sets to create review feature vectors
#training_vectors = build_description_feature("train")
#testing_vectors = build_description_feature("test")

#description training word to vector
training_vectors = CountVectorizer(vocabulary = word_voc).fit_transform(
        training_data).toarray() #10000*7322
testing_vectors = CountVectorizer(vocabulary = word_voc).fit_transform(
        testing_data).toarray() #10000*7322


print(training_vectors.shape) #10000*6719
# print(training_vectors)

print(testing_vectors.shape) # 2000*6719
# print(testing_vectors)
print("****** Done building bow features normalizations ******")



##########################################################
#     Model 
##########################################################
ridge = Ridge(alpha=1.0)
ridge.fit(training_vectors, tags_train_1)
ridge_pred = ridge.predict(testing_vectors)

print(ridge_pred.shape)
print(ridge_pred)
print("\n**** Done Ridge Model ****")   

np.savetxt("./output/ridge_cv_tag_pred.csv",ridge_pred, delimiter=",")
print("\n**** Done saving output ****")   




##########################################################
#      KNN-regressor model to predict 20 neareast images
##########################################################
samples = tags_test_1
pred = ridge_pred
print(samples.shape)
print(pred.shape)

knn = NearestNeighbors(n_neighbors=20)
knn.fit(samples)
result = knn.kneighbors(pred)


##########################################################
#      display description and images
##########################################################
# result[0] - distances
# result[1] - index
text_df = pd.read_csv('./test_des.csv', header=None)
temp = np.array(result)[1]

submission = []
for i in range(2000):
    txt = str(i)+".txt"
    img_ids = ""
    for j in range(20):
        img_ids = img_ids + str(int(temp[i][j]))+".jpg "
    
    submission.append([txt, img_ids.strip()])
    
submission_df = pd.DataFrame(submission, columns = ['Descritpion_ID', 'Top_20_Image_IDs']) 

submission_df.to_csv ("./output/ridge_cv_tag_output.csv", index = None, header=True)


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


