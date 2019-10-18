#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
import nltk

""" If you never installed punkt and wordnet:
nltk.download('punkt') 
nltk.download('wordnet')
!python -m textblob.download_corpora
"""
from tqdm import tqdm
from tika import parser
from nltk import tokenize as tkn
from string import digits

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from textblob import TextBlob
import enchant
import numpy as np

pd.set_option('display.max_colwidth', 200)


# In[1]:


class pdfconverter():
    def __init__(self,directory_path=None,retrieved_company=None):
        
        self.english_voc=enchant.Dict("en_US")
        self.text_processor_pdf=np.vectorize(self.text_preprocessing_pdf,otypes=[str])
        self.df = pd.DataFrame(columns=['pdf','directory','directory_index','raw paragraphs','paragraphs'])
        self.parser=[]
        self.parser_raw=[]
        self.directory_path=directory_path

        self.list_folder=[]
        self.paths={}
        directories=[directory_path+retrieved_company, directory_path+'Domain_vocab']
        for dirs in directories:
            for r,d,f in os.walk(dirs):
               
                if d==[] and 'pdf' in '.'.join(f):
                    self.list_folder.append(r)

            for folder in self.list_folder:
                for i,pdf in enumerate(os.listdir(folder)):
                    if pdf!= '.DS_Store':
                        self.paths[folder]=self.paths.get(folder,[])+[(i,pdf)]
        
    def transform(self):
        """Pdf-files reader with Apache Tika"""
        count=1
        assert len(self.list_folder)>=1 ,"FILES NOT FOUND"
        for i,folder in enumerate(self.list_folder):
            path=folder
            for j,pdf in enumerate(os.listdir(path)):
                if pdf!= '.DS_Store':
                    self.df.loc[count] = [pdf,folder.split('/')[-2], i+1,None,None]
                    
                    """ 0- Read Pdf file """
                    raw = parser.from_file(os.path.join(path,pdf))
                    s = raw['content']
                    
                    """ 1- Handle linebreaks to optimize TextBlob.sentences results"""
                    s=self.treat_new_line(s)
                    
                    """ 2- Divide text by sentences using TextBlob"""
                    blob=TextBlob(s)
                    paragraphs = np.array([str(s) for s in blob.sentences],dtype=str)
                    self.parser = []
                    self.parser_raw=[]
                    p=self.text_processor_pdf(paragraphs)
                    
                    """
                    3- Get rid of bad text data:
                    Discard sentences with too long word (16 is the 99% quantile in english)
                    Discard sentences with too much upper words (CREDENTIALS, Link, TITLE ..)
                    """
                    index_=[i for i,c in enumerate(self.parser) if (True in [len(w)>=16 for w in c.split()] )]
                    index_raw=[i for i,c in enumerate(self.parser_raw) if np.sum([w==w.upper() for w in c.split()])>=4]
                    index=list(set(index_ + index_raw))
                    self.df.loc[count,'paragraphs']=np.delete(np.array(self.parser),index)
                    self.df.loc[count,'raw paragraphs']=np.delete(np.array(self.parser_raw),index)
                    count+=1
                            
            print("files from {} succesfully converted ".format(folder))
                
        return self.df
    
    def remove_non_alpha(self,text):
        
        """ Remove non alpha-decimal caracters that are not dot or linebreaker """
        
        removelist="-\.\/\?\@"
        re_alpha_numeric1=r"[^0-9a-zA-Z"+removelist+" ]"
        clean_text=re.sub(re_alpha_numeric1,'',text)
        clean_text=clean_text.replace('/',' ')
        clean_text=re.sub(' +', ' ', clean_text)
        return clean_text
    
    def treat_new_line(self,text):
        """ 
        This function is aimed to deal with all types of linebreaks we met during our tests 
        There is linebreaks dure to cut-sentences, cut-words, bullet-list, title, new paragraphs, or sentences breaks
        """
        text=text.replace('.\n','. ')
        text=re.sub(r'(\n\s*)+\n+', '\n\n',text )
        
        lw=text.split('\n\n')
        lw=[c for c in lw if c.replace(' ','')!='']
            
        for i in range(1,len(lw)):
            try:

                el=lw[i]
                if len(el)>=1:
                    try:
                        first_w=el.split()[0]
                    except:
                        first_w=el
                    first_l=first_w[0]
                    if first_l.isupper() :
                        if len(lw[i-1])>0 and lw[i-1].replace(' ','') !='':
                            if lw[i-1].replace(' ','')[-1] not in [":",'.',"-",'/',"'",";"]:
                                prec=lw[i-1].split(".")[-1]
                                merge=(prec+' '+lw[i]).split()
                                dic=dict(nltk.tag.pos_tag(merge))
                                proper_noun=dic[first_w]=='NNP'
                                if not proper_noun:
                                    if not "." in lw[i-1]:
                                        lw[i-1]=lw[i-1]+".\n\n "
                                    else:
                                        lw[i-1]=lw[i-1][:-1]+".\n\n "
                                else:
                                    lw[i-1]+=' '


                    elif first_l.islower():
                        if len(lw[i-1])>0 and lw[i-1][-1].replace(' ','')!='':

                            if lw[i-1][-1].replace(' ','')[-1]!='-':
                                lw[i-1]+=""
                            else:

                                ltemp_prev=lw[i-1].split(' ')
                                ltemp_next=lw[i].split(' ')
                                motprev=ltemp_prev[-1][:-1]
                                motnext=lw[i].split(' ')[0]
                                if len((motprev+' '+motnext).split())==2:

                                    if self.english_voc.check(motprev) and self.english_voc.check(motnext) and not self.english_voc.check("".join([motprev,motnext])) :
                                        newmot=" ".join([motprev,motnext])
                                    else:
                                        newmot="".join([motprev,motnext])
                                    ltemp_prev[-1]=newmot
                                    ltemp_next[0]=""
                                    lw[i-1]=" ".join(ltemp_prev)
                                    lw[i]=" ".join(ltemp_next)
                    else:
                        lw[i-1]+="\n\n"
            
            except:
                print('Error occurs, the reader may not be suitable for your pdf files')
            
            
        text="".join(lw)
        
        lw=text.split('\n')
        lw=[c for c in lw if c.replace(' ','')!='']
        for i in range(1,len(lw)):
            try:
                el=lw[i]
                if len(el)>=1:
                    try:
                        first_w=el.split()[0]
                    except:
                        first_w=el
                    first_l=first_w[0]
                    if first_l.isupper() :
                        if len(lw[i-1])>0 and lw[i-1].replace(' ','')!='':
                            if lw[i-1].replace(' ','')[-1] not in [":",'.',"-",'/',"'",";"]:
                                prec=lw[i-1].split(".")[-1]
                                merge=(prec+' '+lw[i]).split()
                                dic=dict(nltk.tag.pos_tag(merge))
                                proper_noun=dic[first_w]=='NNP'
                                if not proper_noun:
                                    if not "." in lw[i-1]:
                                        lw[i-1]=lw[i-1]+".\n\n "
                                    else:
                                        lw[i-1]=lw[i-1][:-1]+".\n\n "
                                else:
                                    lw[i-1]+=' '
                    elif first_l.islower():
                        if len(lw[i-1])>0 and lw[i-1].replace(' ','')!='':
                            if lw[i-1].replace(' ','')[-1]=="-":
                                ltemp_prev=lw[i-1].split(' ')
                                ltemp_next=lw[i].split(' ')
                                motprev=ltemp_prev[-1][:-1]
                                motnext=lw[i].split(' ')[0]
                                if len((motprev+' '+motnext).split())==2:
                                    if self.english_voc.check(motprev) and self.english_voc.check(motnext) and not self.english_voc.check("".join([motprev,motnext])) :
                                        newmot=" ".join([motprev,motnext])
                                    else:
                                        newmot="".join([motprev,motnext])
                                    ltemp_prev[-1]=newmot
                                    ltemp_next[0]=""
                                    lw[i-1]=" ".join(ltemp_prev)
                                    lw[i]=" ".join(ltemp_next)



                            else:
                                lw[i-1]+=" "
                    else:
                        lw[i-1]+=" "
        
            except:
                print('Error occurs, the reader may not be suitable for your pdf files')
        
        text="".join(lw)
        return text
    
    """
    def remove_end_paragraphs(self,p):
        if '-\n' in p:
            paraph=[]
            ltemp=p.split(' ')
            for mot in ltemp:
                if '-\n' in mot:
                    if len(mot.replace('-\n',' ').split())==2:
                        mot1,mot2=mot.replace('-\n',' ').split()
                        if self.english_voc.check(mot1) and self.english_voc.check(mot2) and not self.english_voc.check("".join([mot1,mot2])) :
                            newmot=" ".join([mot1,mot2])
                        else:
                            newmot="".join([mot1,mot2])
                        paraph.append(newmot)
                else:
                    paraph.append(mot)
            p=" ".join(paraph)
        return p.replace('\n',' ')
    """
    
    def cut_text(self,p):
        
        """ Cut text into sentences """
        undesirable_chars=['?','http','www','@']
        if (not True in [i in p for i in undesirable_chars]) and (len(p)>=100) and (len(p.split())>=7):
            
            phrases=self.remove_non_alpha(p)    
            phrases=phrases.replace('.',' ')
            phrases=phrases.replace('-',' ')
            phrases=phrases.replace("?"," ")
            phrases=re.sub(' +', ' ', phrases)
            phrases=re.sub(r'([0-9]+(?=[a-z])|(?<=[a-z])[0-9]+)',"",phrases)
            phrases=phrases.lower()
            self.parser.append(re.sub(' +', ' ', phrases))
            
        return None 
    
    def cut_text_raw(self,p):
        """Cut raw/untreated text into sentences """
        undesirable_chars=['?','http','www','@']
        if (not True in [i in p for i in undesirable_chars]) and (len(self.remove_non_alpha(p))>=100) and (len(self.remove_non_alpha(p).split())>=7):
            self.parser_raw.append(re.sub(' +', ' ', p))
            
        return None
    
    def text_preprocessing_pdf(self,p):
        """ Pipeline of sentences-preprocessing using np.vectorize for faster results """
        #remover_end_paragraphs=np.vectorize(self.remove_end_paragraphs,otypes=[str])
        cleaner=np.vectorize(self.remove_non_alpha,otypes=[str])
        cut_text=np.vectorize(self.cut_text,otypes=[str])
        cut_text_raw=np.vectorize(self.cut_text_raw,otypes=[str])
        assert len(self.parser)==len(self.parser_raw), "Length of the treated sentence treated list does not match length of raw text list: {} / {}".format(len(self.parser),len(self.parser_raw))
        cut_text_raw(p)
        p=cleaner(p)
        cut_text(p)
        return p


    
    


# In[ ]:




