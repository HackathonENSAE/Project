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
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_extraction.text as skf
from sklearn.base import BaseEstimator
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import tokenize as tkn
from tqdm import tqdm
from tika import parser
from string import digits


# In[ ]:


class m3_Tfidf(BaseEstimator):
    """
    A scikit-learn wrapper for TfidfRetriever. Trains a tf-idf matrix from a corpus
    of documents then finds the most N similar documents of a given input document by
    taking the dot product of the vectorized input document and the trained tf-idf matrix.
    
    Parameters
    ----------
    ngram_range : bool, optional
        [shape of ngram used to build vocab] (the default is False)
    max_df : bool, optional
        [while building vocab delete words that have a frequency>max_df] (the default is False)
    stop_words : str, optional
        ['english is the only value accepted'] (the default is False)
    paragraphs : iterable
        an iterable which yields either str, unicode or file objects
    top_n : int
        maximum number of top articles to retrieve
        header should be of format: title, paragraphs.
    verbose : bool, optional
        If true, all of the warnings related to data processing will be printed.
    Attributes
    ----------
    vectorizer : TfidfVectorizer
        See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_matrix : sparse matrix, [n_samples, n_features]
        Tf-idf-weighted document-term matrix.
        
    Examples
    --------
    >>> retriever = TfidfRetriever(ngram_range=(1, 2), max_df=0.85, stop_words='english')
    >>> retriever.fit(X=df['content'])
    
    >>> doc_index=int(input('Which document do you want to use for your question?'))
    >>> retriever.transform(X=df.loc[doc_index,'content'])
    
    >>> Q=str(input('Enter your question'))
    >>> Q=retriever.vectorizer.transform([Q])
    >>> closest_docs,scores = self.retriever.predict(newQst,df.loc[doc_index,'content'])
    """

    def __init__(self,
                 ngram_range=(1, 2),
                 max_df=0.85,
                 stop_words='english',
                 paragraphs=None,
                 verbose=False, top_n=5,
                lemmatize=False,
                transform_text=True):

        self.ngram_range = (1,1)
        self.max_df = max_df
        self.stop_words = stop_words
        self.paragraphs = paragraphs
        self.top_n = top_n
        self.verbose = verbose
        self.transform_text=transform_text
        self.lemmatize=lemmatize and self.transform_text
        self.stem=not lemmatize and self.transform_text
        if self.stem: self.stemmer=PorterStemmer()
        else: self.lemmatizer=WordNetLemmatizer() 
        self.stop_words_list=[self.tokenize(word)[0] for word in list(skf._check_stop_list('english'))]

    def stem_tokens(self,tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed
    
    def lemmatize_tokens(self,tokens,lemmatizer):
        lemmas=[]
        for item in tokens:
            for word, tag in pos_tag(word_tokenize(item)):
                wntag = tag[0].lower()
                wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
                if not wntag:
                    lemma = word
                else:
                    lemma = lemmatizer.lemmatize(word, wntag)
            
            lemmas.append(lemma)
        return lemmas

        """
        for item in tokens:
            lemmas.append(lemmatizer.lemmatize(item))
        return lemmas
        """

    def tokenize(self,text):
        tokens = nltk.word_tokenize(text)
        if self.lemmatize:
        #stems = self.stem_tokens(tokens, self.stemmer)
            lemmas=self.lemmatize_tokens(tokens,self.lemmatizer)
            return lemmas
        elif self.stem:
            stems = self.stem_tokens(tokens, self.stemmer)
            return stems
        else:
            return tokens
        
        
    def fit(self, X, y=None): #generate features and return tfidf scores matrix 

        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range,
                                          max_df=self.max_df,
                                   stop_words=self.stop_words_list,tokenizer=self.tokenize)
        self.vectorizer.fit(X)
        return self
    
    def transform(self,X,y=None):
        self.tfidf_matrix=self.vectorizer.transform(X)
        return self
    
    def predict(self, X, metadata):
        tfidf_matrix=self.tfidf_matrix[metadata[0]:metadata[1],:]
        #cherche les querries les plus proches de chaque sentence
        t0 = time.time()
        question_vector = self.vectorizer.transform([X])
        scores = pd.DataFrame(tfidf_matrix.dot(question_vector.T).toarray())
        closest_docs_indices = scores.sort_values(by=0, ascending=False).index[:self.top_n].values

        # inspired from https://github.com/facebookresearch/DrQA/blob/50d0e49bb77fe0c6e881efb4b6fe2e61d3f92509/scripts/reader/interactive.py#L63
        if self.verbose:
            rank = 1
            table = prettytable.PrettyTable(['rank', 'index', 'title'])
            for i in range(len(closest_docs_indices)):
                index = closest_docs_indices[i]
                if self.paragraphs:
                    article_index = self.paragraphs[int(index)]['index']
                    title = metadata.iloc[int(article_index)]['title']
                else:
                    title = metadata.iloc[int(index)]['title']
                table.add_row([rank, index, title])
                rank+=1
            print(table)
            print('Time: {} seconds'.format(round(time.time() - t0, 5)))

        return closest_docs_indices,scores

