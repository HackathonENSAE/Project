#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import re
import sys
import numpy as np
import nltk
from tqdm import tqdm
from nltk import tokenize as tkn

def remove_non_alpha(text):
    
    """ Remove non alpha-decimal caracters that are not dot or linebreaker """
    
    removelist="-\.\/\?\@"
    re_alpha_numeric1=r"[^0-9a-zA-Z"+removelist+" ]"
    clean_text=re.sub(re_alpha_numeric1,'',text)
    clean_text=clean_text.replace('/',' ')
    clean_text=re.sub(' +', ' ', clean_text)
    return clean_text
    
    