#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:15:39 2019

@author: wenyi
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_float

from nltk.tokenize import word_tokenize 

from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


##########################################################
#     Image features  (label)
##########################################################
training_labels = pd.read_csv(
        './data/features_train/training-image-feature-2.csv', 
        sep=",", header=None)

print(training_labels.shape)

testing_labels = pd.read_csv(
        './data/features_test/testing-image-feature-2.csv', 
        sep=",", header=None)

print(testing_labels.shape)
print("****** Done loading images features labels ******")



##########################################################
#     tag  (label)
##########################################################
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

# training word to vector
tag2image_train, tag_train = load_tag(True)  #tag_train length:80
cv_train = CountVectorizer(vocabulary = tag_train)
training_tag = cv_train.fit_transform(tag2image_train).toarray() #10000*80

# testing word to vector
tag2image_test, tag_test = load_tag(False)   
cv_test = CountVectorizer(vocabulary = tag_train) 
testing_tag = cv_test.fit_transform(tag2image_test).toarray() #2000*80
print(training_tag.shape)
print(testing_tag.shape)
print("****** Done processing training/testing tag labels ******")




##########################################################
#     Descrptions features (data)
##########################################################
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

# description training word to vector
training_vectors = CountVectorizer(vocabulary = word_voc).fit_transform(
        training_data).toarray() #10000*7322
testing_vectors = CountVectorizer(vocabulary = word_voc).fit_transform(
        testing_data).toarray() #10000*7322
print(training_vectors.shape) #10000*6719
print(testing_vectors.shape) #10000*6719

print("****** Done processing training/testing description data ******")




##########################################################
#     Model 
##########################################################
ridge_tag = Ridge(alpha=1.0)
ridge_tag.fit(training_vectors, training_tag)
ridge_m1_pred = ridge_tag.predict(testing_vectors)

print(ridge_m1_pred.shape)
print("\n**** Done Ridge Model 1 ****")   

np.savetxt("./output/ridge_m1_pred.csv", ridge_m1_pred, delimiter=",")
print("\n**** Done saving output ****")   


ridge_img = Ridge(alpha=1.0)
ridge_img.fit(training_vectors, training_labels)
ridge_m2_pred = ridge_img.predict(testing_vectors)

print(ridge_m2_pred.shape)
print("\n**** Done Ridge Model 2 ****")   

np.savetxt("./output/ridge_m2_pred.csv", ridge_m2_pred, delimiter=",")
print("\n**** Done saving output ****")   



##########################################################
#      cos similarity
##########################################################
samples = preprocessing.normalize(testing_tag, norm='l2') 
pred = preprocessing.normalize(ridge_m1_pred, norm='l2') 

print(samples.shape)
print(pred.shape)

cos_1 = cosine_similarity(pred, samples)
print(cos_1.shape)
print(cos_1)

##########################################################
samples = preprocessing.normalize(testing_labels, norm='l2') 
pred = preprocessing.normalize(ridge_m2_pred, norm='l2') 

print(samples.shape)
print(pred.shape)

cos_2 = cosine_similarity(pred, samples)
print(cos_2.shape)
print(cos_2)

##########################################################
# cos similarity combined
cos = cos_1 + cos_2
print(cos.shape)
print(cos)

text_df = pd.read_csv('./test_des.csv', header=None)
submission = []


##########################################################
#      display description and images
##########################################################
for i in range(cos.shape[0]):
#for i in range(10):
    txt = str(i)+".txt"
    img_ids = ""

    img_list = cos[i].argsort()[-20:][::-1]
    # print(img_list)
    
    # print description
    # print("\n" + str(i))
    # print(" ".join(text_df.iloc[i,:1]))  
    
    # plot preview images
    # fig, axes = plt.subplots(5,4,figsize=(10,10)) 
    
    for j in range(20):
        img_ids = img_ids + str(img_list[j]) +".jpg "
        # path = "./data/images_test/"+ str(img_list[j]) + ".jpg"

        # img = img_as_float(mpimg.imread(path))
        # ax = axes[j//4, j%4]
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.imshow(img)
        
    # plt.show()
    # print(txt)
    # print(img_ids)
    submission.append([txt, img_ids.strip()])
    submission_df = pd.DataFrame(submission, 
                                 columns = ['Descritpion_ID', 'Top_20_Image_IDs']) 
    
    submission_df.to_csv ("./output/final_submission.csv", index = None, header=True)

print("\n**** Done saving submission ****")   

