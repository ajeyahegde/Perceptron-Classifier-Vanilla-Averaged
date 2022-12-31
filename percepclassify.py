import sys

import numpy as np
import re

model_filename = sys.argv[1]
input_filename = sys.argv[2]
#model_filename = "vanillamodel.txt"

model_file = open(model_filename, "r", encoding='utf8')

dummy = model_file.readline()

feature_size = int(model_file.readline().strip())

weights1 = np.zeros(feature_size)
bias1 = 0

weights2 = np.zeros(feature_size)
bias2 = 0

for i in range(feature_size):
    weights1[i] = float(model_file.readline().strip())

bias1 = float(model_file.readline().strip())


dummy = model_file.readline()

for i in range(feature_size):
    weights2[i] = float(model_file.readline().strip())

bias2 = float(model_file.readline().strip())

dummy = model_file.readline()

wordset = dict()

for i in range(feature_size):
    wordset[model_file.readline().strip()] = i

#input_filename = "dev-text.txt"
input_file = open(input_filename,"r", encoding='utf8')

review_id_list = list()

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

dataset = list()

for line in input_file:
    review_id = line[0:7]
    review_id_list.append(review_id)
    review_text = line[8:-1]
    review_text = clean_review(review_text)
    words = review_text.split(" ")
    feature = np.zeros(feature_size)
    for word in words:
        if word in wordset:
            feature[wordset[word]] += 1
    dataset.append(feature)

labels1 = list()

for feature in dataset:
    wx = np.dot(weights1, feature)
    a = wx + bias1
    if a < 0:
        label = "Fake"
    else:
        label = "True"
    labels1.append(label)

labels2 = list()
for feature in dataset:
    wx = np.dot(weights2, feature)
    a = wx + bias2
    if a < 0:
        label = "Neg"
    else:
        label = "Pos"
    labels2.append(label)

#print(labels1)
#print(labels2)

output_file = open("percepoutput.txt","w", encoding='utf8')
for i in range(len(review_id_list)):
    output_file.write(review_id_list[i]+" "+labels1[i]+" "+labels2[i]+"\n")

