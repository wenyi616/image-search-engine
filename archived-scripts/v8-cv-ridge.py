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
#            if get_wordnet_pos(w) == "v":
##                print("discard verb")
#                continue

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
print("****** Done loading images features ******")


##########################################################
#     Model 
##########################################################
ridge = Ridge(alpha=1.0)
ridge.fit(training_vectors, training_labels)
ridge_pred = ridge.predict(testing_vectors)

print(ridge_pred.shape)
print(ridge_pred)
print("\n**** Done Ridge Model ****")   

np.savetxt("./output/ridge_pred.csv",ridge_pred, delimiter=",")
print("\n**** Done saving output ****")   




##########################################################
#      KNN-regressor model to predict 20 neareast images
##########################################################
samples = testing_labels
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

submission_df.to_csv ("./output/ridge_output.csv", index = None, header=True)


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


