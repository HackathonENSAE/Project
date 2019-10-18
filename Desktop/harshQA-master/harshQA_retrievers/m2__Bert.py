#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
import re
import sys
import uuid
import prettytable
import time
import cProfile
import re
import pandas as pd
import numpy as np
import tinyarray
import nltk
import torch
from tqdm import tqdm
from tika import parser
from nltk import tokenize as tkn
from string import digits
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_extraction.text as skf
from sklearn.base import BaseEstimator
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import enchant
from utils.models import InferSent
from sentence_transformers import SentenceTransformer
from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import spherecluster
from sklearn import decomposition
from sklearn import datasets


# In[ ]:


class m2_Bert(BaseEstimator):
    
    def __init__(self,top_n=5):
        self.top_n = top_n
        self.bert=SentenceTransformer('bert-large-nli-stsb-mean-tokens')
        
    def fit(self,X,y=None):
        """ X: any iterable which contains words to finetune vocabulary """
        return self
    
    def transform(self,X,y=None):
        """ X: any iterable which contains sentences to embed """
        self.embeddings = self.bert.encode(list([s for s in X ]))
        self.reduced_embeddings = np.apply_along_axis(lambda v: self.normalize(v), 1,self.embeddings)
        return self
    
    def normalize(self,array):
        return array/np.linalg.norm(array)
    
    def predict(self,X,metadata):
        
        question=self.bert.encode([X])
        encoded_question=self.normalize(question)
        self.reduced_question=self.normalize(question)
        
        reduced_embeddings=self.reduced_embeddings[metadata[0]:metadata[1],:]
        
        data=reduced_embeddings.dot(self.reduced_question.T)
        self.scores_inf=pd.DataFrame(data,index=range(len(data)))
        closest_docs_indices = self.scores_inf.sort_values(by=0, ascending=False).index[:self.top_n].values
        return closest_docs_indices,self.scores_inf

        

