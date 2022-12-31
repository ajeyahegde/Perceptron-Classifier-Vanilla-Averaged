import numpy as np

import sys
import math
import re
import random


input_filename = sys.argv[1]
#input_filename = "train-labeled.txt"
input_file = open(input_filename, "r", encoding='utf8')

data = list()

#Function to clean data
def clean_review(text):
    regex_alpha = '[^a-zA-Z]'
    regex_html = '<.*?>'
    #Remove URL from reviews
    text = re.sub(r"http\S+", "", text)
    #Remove HTML from reviews
    text = re.sub(regex_html, ' ', text)
    #Convert to lower letters
    text = text.lower()
    #Remove non alphabetical characters
    text = re.sub(regex_alpha, ' ', text)
    text = " ".join(text.split())
    return text

wordset = set()

#Loop to read input data
for line in input_file:
    review_id = line[0:7]
    label1 = line[8:12]
    label2 = line[13:16]
    review_text = line[17:-1]
    review_text = clean_review(review_text)
    words = review_text.split(" ")
    for word in words:
        if(len(word) != 1):
            wordset.add(word)
    data.append([review_text, label1, label2])

feature_size = len(wordset)
print(feature_size)
word_index = dict()
index = 0
for word in wordset:
    word_index[word] = index
    index += 1
dataset = list()
for row in data:
    text = row[0]
    words = text.split(" ")
    feature = np.zeros(feature_size)
    for word in words:
        if word in word_index:
            feature[word_index[word]] += 1
    row[0] = feature
    label1 = 1
    label2 = 1
    if row[1] == "Fake":
        label1 = -1
    if row[2] == "Neg":
        label2 = -1
    dataset.append([feature, label1, label2])

weights1 = np.zeros(feature_size)
bias1 = 0
epoch = 5


for iter in range(epoch):
    random.shuffle(dataset)
    for row in dataset:
        x = np.array(row[0])
        wx = np.dot(weights1, x)
        a = wx+bias1
        if (row[1] * a) <= 0:
            for i in range(feature_size):
                weights1[i] = weights1[i]+row[1]*x[i]
            bias1 = bias1+row[1]

weights2 = np.zeros(feature_size)
bias2 = 0

#Training Vanilla Perceptron Model
for iter in range(epoch):
    for row in dataset:
        x = np.array(row[0])
        wx = np.dot(weights2, x)
        a = wx+bias2
        if (row[2] * a) <= 0:
            for i in range(feature_size):
                weights2[i] = weights2[i]+row[2]*x[i]
            bias2 = bias2+row[2]

print(len(weights1))
print(len(weights2))

#Writing Vanilla Model to file
vanilla_file = open("vanillamodel.txt", "w", encoding='utf8')

vanilla_file.write("Model 1\n")
vanilla_file.write(str(feature_size)+"\n")

for i in range(feature_size):
    vanilla_file.write(str(weights1[i])+"\n")

vanilla_file.write(str(bias1)+"\n")

vanilla_file.write("Model 2\n")

for i in range(feature_size):
    vanilla_file.write(str(weights2[i]) + "\n")

vanilla_file.write(str(bias2) + "\n")

vanilla_file.write("Bag of words\n")
for word in word_index.keys():
    vanilla_file.write(word+"\n")


#Averaged Model Training
averaged_weights1 = np.zeros(feature_size)
average_bias1 = 0
cached_weights1 = np.zeros(feature_size)
cached_bias1 = 0
c = 1
epoch = 5
for iter in range(epoch):
    for row in dataset:
        x = np.array(row[0])
        wx = np.dot(averaged_weights1, x)
        a = wx+average_bias1
        if (row[1] * a) <= 0:
            for i in range(feature_size):
                averaged_weights1[i] = averaged_weights1[i]+row[1]*x[i]
                cached_weights1[i] = cached_weights1[i]+row[1]*x[i]*c
            average_bias1 = average_bias1+row[1]
            cached_bias1 = cached_bias1+row[1]*c
        c += 1

for i in range(feature_size):
    averaged_weights1[i] = averaged_weights1[i] - ((cached_weights1[i]) / c)

average_bias1 = average_bias1-(cached_bias1 / c)


averaged_weights2 = np.zeros(feature_size)
average_bias2 = 0
cached_weights2 = np.zeros(feature_size)
cached_bias2 = 0

c = 1
for iter in range(epoch):
    for row in dataset:
        x = np.array(row[0])
        wx = np.dot(averaged_weights2, x)
        a = wx+average_bias2
        if (row[2] * a) <= 0:
            for i in range(feature_size):
                averaged_weights2[i] = averaged_weights2[i]+row[2]*x[i]
                cached_weights2[i] = cached_weights2[i]+row[2]*x[i]*c
            average_bias2 = average_bias2+row[2]
            cached_bias2 = cached_bias2+row[2]*c
        c += 1

for i in range(feature_size):
    averaged_weights2[i] = averaged_weights2[i] - ((cached_weights2[i]) / c)

average_bias2 = average_bias2-(cached_bias2 / c)

#Writing Averaged Model to file
averaged_perceptron_file = open("averagedmodel.txt", "w", encoding='utf8')

averaged_perceptron_file.write("Model 1\n")
averaged_perceptron_file.write(str(feature_size)+"\n")

for i in range(feature_size):
    averaged_perceptron_file.write(str(averaged_weights1[i])+"\n")

averaged_perceptron_file.write(str(average_bias1)+"\n")

averaged_perceptron_file.write("Model 2\n")

for i in range(feature_size):
    averaged_perceptron_file.write(str(averaged_weights2[i]) + "\n")

averaged_perceptron_file.write(str(average_bias2) + "\n")

averaged_perceptron_file.write("Bag of words\n")
for word in word_index.keys():
    averaged_perceptron_file.write(word+"\n")



