#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:55:26 2019

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
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model



# helper function for lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def pre_process_data(data):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    new_data = []
    for i in range(len(data)):
        review = data[i]
        # convert to lowercase
        review = review.lower()

        # add whiltespace after punctuation
        review = re.sub( r'([a-zA-Z])([,.!])', r'\1\2 ', review)

        # remove punctuation
        review = review.translate(string.maketrans("",""), string.punctuation)

        filtered_review = "" 
        for w in word_tokenize(review):
#            print(w)
#            print(get_wordnet_pos(w))
            if get_wordnet_pos(w) == "v":
#                print("discard verb")
                continue

            # Lemmatize with POS Tag
            try:
                w = lemmatizer.lemmatize(w, get_wordnet_pos(w))
            except:
                # fiancé, café, crêpe, puréed
                # w = unidecode.unidecode(unicode(w, "utf-8"))
                continue
            
            # remove stop word
            if w not in stop_words:
                filtered_review =filtered_review + w + " "

        review = filtered_review
        new_data.append(review)  

    return np.array(new_data)


# "train" "test"
def load_description_data(data, size):
    """      
    load description data
    -----------
    
    Parameters
    -----------
    data: "train" / "test"
    size: int
    
    
    Return
    -----------
    5-sentences data after removing stop words and lemmatizing 
    """
    
    path = "./data/descriptions_" + data + "/"
    temp = []
    
    for i in range(size):
        file_name = str(i) + '.txt'
        file_path = path + file_name
        des = ""
        with open(file_path) as f:
            for line in f.readlines():
                des = des + (line.strip('\n')) + ' '
        temp.append(des)
    
    result = np.array(pre_process_data(temp))
    savepath = data + "_des.csv"
    np.savetxt(savepath, result, fmt='%s')
    return result


def build_description_feature(data):
    """      
    build feature vectors
    -----------
    
    Parameters
    -----------
    data: "train" / "test"    
    
    Return
    -----------
    normalized feature vectors 
    """
    
    temp_dict = {}
    temp_dict["train"] = training_data
    temp_dict["test"] = testing_data
    
    temp = []
    for i in range(len(temp_dict[data])):
        # initialize a vector of size (1*6719) 
        review_vector = np.zeros(len(word_dict))
        review = temp_dict[data][i]
    
        for w in word_tokenize(review):
            if w in word_dict:
                index = word_dict.keys().index(w)
                review_vector[index] += 1   
        
        temp.append(review_vector)
    
    temp = np.array(temp)
    temp = preprocessing.normalize(temp, norm='l2') 
    return temp


##########################################################
#     Descrptions features (data)
##########################################################

# training_data = load_description_data("train", 10000)
# testing_data = load_description_data("test", 2000)

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

# iterate thru all reviews in the training set
for i in range(len(training_data)):
    review = training_data[i]
    
    for w in word_tokenize(review):
        if w not in word_dict:
            word_dict[w] = 0

print(len(word_dict)) #7321
# print(word_dict)
print("****** Done building word dictionary ******")


# iterate all reviews in both sets to create review feature vectors
training_vectors = build_description_feature("train")
testing_vectors = build_description_feature("test")

print(training_vectors.shape) #10000*7321
# print(training_vectors)

print(testing_vectors.shape) # 2000*7321
# print(testing_vectors)
print("****** Done building bow features normalizations ******")





##########################################################
#     Image features  (label)
##########################################################

training_label = pd.read_csv(
        './data/features_train/training-image-feature-2.csv', 
        sep=",", header=None)

print(training_label.shape)

testing_label = pd.read_csv(
        './data/features_test/testing-image-feature-2.csv', 
        sep=",", header=None)

print(testing_label.shape)
print("****** Done loading images features ******")




##########################################################
#     Model 
##########################################################
clf = Ridge(alpha=0.5)
clf.fit(training_vectors, training_label)
clf_pred = clf.predict(testing_vectors)

print(clf_pred.shape)
print(clf_pred)
print("\n**** Done Ridge ****")   

np.savetxt("./output/ridge_pred.csv",clf_pred, delimiter=",")
print("\n**** Saving output ****")   

