import pandas as pd
import scipy         
#SciPy is a collection of mathematical algorithms and convenience functions built on the Python. high-level commands and classes for manipulating and visualizing data.
import pylab
import operator
import nltk
from nltk.probability import *
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import json


all = pd.read_json("comments.csv")
stop_eng = stopwords.words('english')
customstopwords =[]

tokens = []
sentences = []
tokenizedSentences =[]
for txt in all.text:
    sentences.append(txt.lower())
    tokenized = [t.lower().encode('utf-8').strip(":,.!?") for t in txt.split()]
    tokens.extend(tokenized)
    tokenizedSentences.append(tokenized)

hashtags = [w for w in tokens if w.startswith('#')]
ghashtags = [w for w in tokens if w.startswith('+')]
mentions = [w for w in tokens if w.startswith('@')]
links = [w for w in tokens if w.startswith('http') or w.startswith('www')]
filtered_tokens = [w for w in tokens if not w in stop_eng and not w in customstopwords and w.isalpha() and not len(w)<3 and not w in hashtags and not w in ghashtags and not w in links and not w in mentions]
english_words = set(nltk.corpus.words.words())
words=[ word for word in filtered_tokens if word in english_words ]
fd = nltk.FreqDist(words)
fd.plot(30,cumulative=False)
# comments plotted

def word_feats(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

trainfeats = negfeats + posfeats
classifier = NaiveBayesClassifier.train(trainfeats)
neg = 0
pos = 0
# sentence = all.text
# sentence = sentence.lower()

for word in words:
    print word
    classResult = classifier.classify( word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1

result_dict={};
array_value=[];
result_dict["pos"]=str(float(pos)/len(words));
result_dict["neg"]=str(float(neg)/len(words));
array_value.append(result_dict);

with open('test.json', 'w') as outfile:
    json.dump(array_value[0], outfile)
