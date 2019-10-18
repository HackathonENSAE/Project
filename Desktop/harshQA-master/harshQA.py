#!/usr/bin/env python
# coding: utf-8

# # Pipeline for Question Answering -  ESG Assessment Projects BNP

# Pipeline that launch all nlp models ! (property of BNP Paribas Risk Air Team)

""" 
If you never installed punkt and wordnet:
nltk.download('punkt') 
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
!python -m textblob.download_corpora
!python -m spacy download en

"""

import json
import os
import re
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

import sys
import uuid
import time
import pandas as pd
import numpy as np
import enchant
import nltk
import torch
import torch.nn as nn
import tensorflow as tf #for tfrecordfiles
from tqdm import tqdm
import sklearn.feature_extraction.text as skf
from sklearn import decomposition

from data.Infersent.model import InferSent
from sentence_transformers import SentenceTransformer
from tika import parser
from nltk import tokenize as tkn
from string import digits
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob



flags = tf.flags

FLAGS = flags.FLAGS



flags.DEFINE_integer(
    "model",5,
    "** Select a model \n **Model without pre-clustering: \n\t 1-Infersent_glove [Pretrained] (10 min/corpus) \n\t 2-Bert [Pretrained on SQUAD] (30 min/corpus) \n \t 3-Tf_Idf_Lemmatizer [Trained on our corpus] (5 min/ corpus) \n**Model with pre-clustering\\t 4-Tf-Idf_Bert [Pretrained on SQuAD] (3 min/query)\n\t 5-Tf-Idf_Bert_enhanced [Finetuned on MsMarco] (1:30 min/query)\n\t 6- All \n")
"""
*** 
** Model without pre-clustering:
        1-Infersent_glove [Pretrained] (10 min/corpus)
        2-Bert [Pretrained on SQUAD] (30 min/corpus)
        3-Tf_Idf_Lemmatizer [Trained on our corpus] (5 min/ corpus)
        
* *Model with pre-clustering
        4-Tf-Idf_Bert [Pretrained on SQuAD] (3 min/query)
        5-Tf-Idf_Bert_enhanced [Finetuned on MsMarco] (1:30 min/query)
        6- All

** The settings of our test was: *** 
  Run on CPU
  size_cluster=50
  Corpus of text was 1500 sentences(300 pages) and 15 queries
  Corpus of domain_vocab was 3000 sentences (600 pages) 
  The timespeed of pdf converter is approximately 10s/1000 pages\
  The best results were achieved with model 5 harshQA (see harshQAeval) )
"""

flags.DEFINE_boolean(
    "demo",False,
    "Demo mode with your own pdfs")


flags.DEFINE_integer(
    "size_cluster",35,
    "size of the clusters of candidate to feed in neural network.")

flags.DEFINE_string(
    "domain", None ,
    "Domain folder name to process Q&A")

flags.DEFINE_string(
    "retrieved_company",None,
    "Company folder name to query")

flags.DEFINE_string(
    "pdf_directory",
    './utils/pdf_files/',
    "Path of the pdf directory")

flags.DEFINE_string(
    "output_folder",
    './output/tfrecord',
    "Path to save the tf records ")

flags.DEFINE_string(
    "vocab_file",
    "./data/bert/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt",
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "infersent_path",
    "./data/fastText/crawl-300d-2M.vec",
    "Path of the .vec file of GloVe or FastText")

flags.DEFINE_string(
    "infersent_model",
    "./data/encoder/infersent2.pkl",
    "Path of the .pkl file of infersent model")    

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 128,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")
    
flags.DEFINE_integer(
    "top_n",5,
    "Number of doc to retrieve per query")

flags.DEFINE_boolean(
    "bigram",False,
    "Wether to use bigram on unigram as langage models, (at the moment only true is possible but check choose_args function to tune it)")
    
flags.DEFINE_float(
    "threshold",0.00,
    "Threshold of similarity to apply (not advised)")


# In[16]:


## Required parameters
flags.DEFINE_string(
    "data_dir",
    "./output/tfrecord",
    "The input data dir. Should contain the .tfrecord files and the supporting "
    "query-docids mapping files.")

flags.DEFINE_string(
    "output_dir",
    "./output/output_QA",
    "The output directory where the model checkpoints will be written after train "
    "Will also store the raw tsv predictions ")

flags.DEFINE_string(
    "bert_config_file", "./data/bert_msmarco/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_boolean(
    "msmarco_output", True,
    "Whether to write the predictions to a MS-MARCO-formatted file.")

flags.DEFINE_string(
    "init_checkpoint",
    "./data/bert_msmarco/model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 35, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 35, "Total batch size for eval.")
#32 former settings
flags.DEFINE_float("learning_rate", 1e-6, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 400000,
                     "Total number of training steps to perform.")

flags.DEFINE_integer("max_eval_examples", None,
                     "Maximum number of examples to be evaluated.")

flags.DEFINE_integer(
    "num_warmup_steps", 40000,
    "Number of training steps to perform linear learning rate warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 100,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# # Code dependencies of the project

#Code implemented reader and BERT reranking tuning: 
from utils.text_cleaner import remove_non_alpha
from harshQA_pdf_reader.reader import pdfconverter
from harshQA_reranker.tokenization import* 
from harshQA_reranker.tfrecord_QA import *
import harshQA_reranker.metrics as metrics
import harshQA_reranker.modeling as modeling
import harshQA_reranker.optimization as optimization 

#Libraries Bert,Infersent in case you want to try other models
from harshQA_retrievers.m1__Infersent import m1_Infersent
from harshQA_retrievers.m2__Bert import m2_Bert
from harshQA_retrievers.m3__Tfidf import m3_Tfidf


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()
  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, log_probs)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    #tf.logging.info("*** Features ***")
    #for name in sorted(features.keys()):
      #tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    (total_loss, per_example_loss, log_probs) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    scaffold_fn = None
    initialized_variable_names = []
    
    #Initializes current variables with tensors loaded from given checkpoint.
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    

    #tf.logging.info("**** Trainable Variables ****")
    """
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    """

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              "log_probs": log_probs,
              "label_ids": label_ids,
          },
          scaffold_fn=scaffold_fn)

    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(dataset_path, seq_length, is_training,
                     max_eval_examples=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""

    batch_size = params["batch_size"]
    output_buffer_size = batch_size * 1000

    def extract_fn(data_record):
      features = {
          "query_ids": tf.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "doc_ids": tf.FixedLenSequenceFeature(
              [], tf.int64, allow_missing=True),
          "label": tf.FixedLenFeature([], tf.int64),
      }
      sample = tf.parse_single_example(data_record, features)

      query_ids = tf.cast(sample["query_ids"], tf.int32)
      doc_ids = tf.cast(sample["doc_ids"], tf.int32)
      label_ids = tf.cast(sample["label"], tf.int32)
      input_ids = tf.concat((query_ids, doc_ids), 0)

      query_segment_id = tf.zeros_like(query_ids)
      doc_segment_id = tf.ones_like(doc_ids)
      segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)

      input_mask = tf.ones_like(input_ids)
      """
      #input_ids=vecteur de 1 de la taille query*(docs+1)
      #segment_ids= vecteur de 0 taille des querries concatener avec vecteur 1 taille doc_ids
      #input_mask= que des 1 de la taille querries+docs
      """
      features = {
          "input_ids": input_ids,
          "segment_ids": segment_ids,
          "input_mask": input_mask,
          "label_ids": label_ids,
      }
      return features

    dataset = tf.data.TFRecordDataset([dataset_path])
    dataset = dataset.map(
        extract_fn, num_parallel_calls=4).prefetch(output_buffer_size)

    if is_training:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=1000)
    else:
      if max_eval_examples:
        # Use at most this number of examples (debugging only).
        dataset = dataset.take(max_eval_examples)
        # pass

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes={
            "input_ids": [seq_length],
            "segment_ids": [seq_length],
            "input_mask": [seq_length],
            "label_ids": [],
        },
        padding_values={
            "input_ids": 0,
            "segment_ids": 0,
            "input_mask": 0,
            "label_ids": 0,
        },
        drop_remainder=True)

    return dataset
  return input_fn


def run_msmarco():
  #print("okok FLAGS.bert_config_file",FLAGS.bert_config_file)
  #tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.io.gfile.makedirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=2,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    #tf.logging.info("***** Running training *****")
    #tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    #tf.logging.info("  Num steps = %d", FLAGS.num_train_steps)
    train_input_fn = input_fn_builder(
        dataset_path=FLAGS.data_dir + "/dataset_train.tf",
        seq_length=FLAGS.max_seq_length,
        is_training=True)
    estimator.train(input_fn=train_input_fn,
                    max_steps=FLAGS.num_train_steps)
    #tf.logging.info("Done Training!")

  if FLAGS.do_eval:
    for set_name in ["eval"]:
      #tf.logging.info("***** Running evaluation *****")
      #tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
      max_eval_examples = None
      if FLAGS.max_eval_examples:
        max_eval_examples = FLAGS.max_eval_examples * FLAGS.size_cluster

      eval_input_fn = input_fn_builder(
          dataset_path=FLAGS.data_dir + "/dataset_" + set_name + ".tf",
          seq_length=FLAGS.max_seq_length,
          is_training=False,
          max_eval_examples=max_eval_examples)

      #tf.logging.info("Getting results ...")

      if FLAGS.msmarco_output:
        msmarco_file = tf.io.gfile.GFile(
            FLAGS.output_dir + "/msmarco_predictions_" + set_name + ".tsv", "w")
        query_docids_map = []
        with tf.io.gfile.GFile(
            FLAGS.data_dir + "/query_doc_ids_" + set_name + ".txt") as ref_file:
          for line in ref_file:
            query_docids_map.append(line.strip().split("\t"))

      result = estimator.predict(input_fn=eval_input_fn,
                                 yield_single_examples=True)
      #start_time = time.time()
      results = []
      
      example_idx = 0
      total_count = 0
      for item in result:
        results.append((item["log_probs"], item["label_ids"]))
        #if total_count % 10000 == 0:
          #tf.logging.info("Read {} examples in {} secs".format(
              #total_count, int(time.time() - start_time)))
        #print("***results***",len(results),results)
        if len(results) == FLAGS.size_cluster:

          log_probs, labels = zip(*results)
          log_probs = np.stack(log_probs).reshape(-1, 2)
          #print("probs=",np.exp(log_probs))
          
          labels = np.stack(labels)

          scores = log_probs[:, 1]
          pred_docs = scores.argsort()[::-1]
          #print("scores",scores)
          scores_sorted=np.sort(np.exp(scores))[::-1]
          #print("scores_sorted",scores_sorted,scores_sorted[0])
          #print("pred_docs=",pred_docs)
          #print("logprobs=",log_probs)

          if FLAGS.msmarco_output:
            start_idx = example_idx * FLAGS.size_cluster
            end_idx = (example_idx + 1) * FLAGS.size_cluster
            query_ids, doc_ids = zip(*query_docids_map[start_idx:end_idx])
            assert len(set(query_ids)) == 1, "Query ids must be all the same."
            query_id = query_ids[0]
            rank = 1
            for doc_idx in pred_docs:
              doc_id = doc_ids[doc_idx]
              # Skip fake docs, as they are only used to ensure that each query
              # has proper number of docs.
              if doc_id != "00000000":
                msmarco_file.write(
                    "\t".join((query_id, doc_id, str(rank), str(scores_sorted[rank-1]) )) + "\n")
                rank += 1

          example_idx += 1
          results = []

        total_count += 1

      if FLAGS.msmarco_output:
        msmarco_file.close()

class m5_harshQA(BaseEstimator):
    """
    A scikit-learn estimator for TfidfRetriever. Trains a tf-idf matrix from a corpus
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

    def __init__(self, TF_emb, q_emb, content, content_doc,stemmed_content, vocab, l_querries,l_querries_raw, by_querries=True, top_n=5,threshold_w=0.002, portion_w=1.0,rank=500,verbose=False):

        self.top_n = top_n
        self.verbose = verbose
        self.TF_emb=TF_emb
        self.threshold_importance=threshold_w
        self.prop_w=portion_w
        self.rk=rank
        self.by_querries=by_querries
        self.querries=l_querries.copy() #(dataframe querries with word importance)
        self.querries_raw=l_querries_raw.copy()
        self.content=content
        self.content_doc=content_doc
        self.stemmed_content=stemmed_content
        self.vocab=vocab
        self.dic_emb={}
        self.q_emb=q_emb
    
    def select_term_Naystrom(self,T):

        #T is the Tf-Idf Matrix 
        #s is the fraction of important words that we want to retrieve (s=1.0 generally)
        print('begin of term selection')
        #Retrieve all the stems of tf-idf vocabulary
        terms=self.vocab
        
        #First we remove digits of the terms candidate
        n=int(self.prop_w*len(terms))
        idx_words=[]
        for i,term in enumerate(terms):
            try: float(term)
            except: idx_words.append(i)

        #Secondly we set a threshold to select words that appear at least each p documents

        #Set threshold to select words that appear at least each p documents 
        self.freq_term=np.zeros((len(self.content),len(terms)))
        
        for j,stem in enumerate(self.stemmed_content):
            for i,c in enumerate(terms):
                if c in stem:
                    self.freq_term[j,i]+=1
                    
        self.freq_term_by_doc=np.apply_along_axis(lambda x: np.mean(x),0,self.freq_term)
        ids=np.where(self.freq_term_by_doc>self.threshold_importance )[0]
        idx_words=[ i for i in ids if i in idx_words]


        #Init selection of terms with most correlation 
        if self.prop_w!=1.0:
            prob=np.sum(np.abs(T)>0,axis=0)
            prob=prob/np.sum(prob)
            prob=np.squeeze(prob)
            idx=[]
            m=len(idx_words)

            for i in range(n):

                p=int(np.random.choice(m,1,prob[0]))
                idx.append(int(idx_words[p]))
                prob[idx_words[p]]=0
                prob=prob/np.sum(prob)

            return idx,np.squeeze(T[:,idx])
        
        else:
            idx=idx_words
            self.idx=idx
            return idx,np.squeeze(T[:,idx])
        print('end of term selection')
    
    

    def fit(self, X, y=None): 

        return self

    def transform(self): #generate new enhanced tf-idf-farahat features ( co-occurence weight frequency matrix) 
        
        idx,A=self.select_term_Naystrom(self.TF_emb.toarray())   
        
        #Full rank approximation
        if self.rk==-1:
            self.rk=len(idx)
                
        qemb=self.q_emb
        print('begin of generate kernel')
        
        X=self.TF_emb.toarray().transpose()
        X=np.c_[X,qemb]

        L=np.eye(X.shape[0])*np.sqrt(X.shape[0])
        L_inv=np.linalg.inv(L)
        G=L_inv@X@X.transpose()@L_inv
        Gs=G[idx,:]
        Gs=Gs[:,idx]
        S,V,D=np.linalg.svd(Gs, full_matrices=True)
        Ssub,Vsub=S[:,:self.rk], np.diag(V)[:self.rk,:self.rk]
        #Ssub,Vsub=S,np.diag(V)
        #G=Ssub@Vsub$Ssub.transpose() but this operation is not necessary, we just need the decomposition of G

        D_sub_inv=np.diag(np.apply_along_axis(lambda x: 1/np.sqrt(x) , 0, V[:self.rk]))
        #D_sub_inv=np.diag(np.apply_along_axis(lambda x: 1/np.sqrt(x) , 0, V))
        W=(((D_sub_inv@Ssub.transpose())@X[idx,:])@(X.transpose()))@X
        self.TF_FARAHAT_emb=np.apply_along_axis(lambda x: x/np.sqrt(np.sum(x**2)), 1,W.transpose())
        print('end of generate kernel')
        return self

    def predict(self,metadata,repo):
        
        self.content_repo=self.content_doc[repo]
        questions=self.querries['query'].tolist()
        W_=self.TF_FARAHAT_emb
        scores=W_[-len(questions):]@(W_[metadata[0]:metadata[1]].transpose())
        rank=np.apply_along_axis(lambda x:np.argsort(-x),1,scores)
        raw_or_treated=1 #raw text
        
        
        all_answers=[]
        all_scores=[]
        all_indices=[]
        all_models=[]
        all_ranks=[]
        all_querries=[]
        self.all_answers_retrieved_raw=[]
        self.all_answers_retrieved=[]
        all_querries_treated=[]
        all_question_ids=[]
        special_answer=np.zeros_like(questions,dtype=int)
        
        
        
        for question in range(len(questions)):
            cluster_ids=[]
            answers=[]
            answers_raw=[]
            dic_answers={}
            count_answers=1
            #print('*****  Question {} : {} {}'.format(question,questions[question],'********'))
            rk=0
            count=0
            #print('\n----------------\n')
            w_important=[w for w in questions[question].split(" ") if (w==w.upper() and len(w)>1) ]
            #print('Important words=',w_important,rank.shape[1])
            #Display best candidates according to clustering
            
            ### Special words retrieval (Tf-Idf search) ###
            if w_important !=[]:
                for w in w_important:
                    try:
                        w=m3_Tfidf(transform_text=False).tokenize(w)[0]
                        w=w.lower()
                        best_id=[i for i in range(rank.shape[1]) if w in self.content_repo[0][rank[question,i]].split(' ')]
                    except:
                        print("an error occured with the important word{}".format(word))
                    
                    #print(",best_id",w,best_id)
                    if best_id!=[]:
                        id_match=best_id[0]
                        result=self.content_repo[raw_or_treated][rank[question,id_match]]
                        #print('Special Anserw : {}'.format(self.content_repo[raw_or_treated][rank[question,id_match]]))
                        #print('\n----------------\n')

                        all_question_ids.extend([question])
                        all_indices.extend([rank[question,id_match]])
                        all_scores.extend([scores[question,rank[question,id_match]]])
                        all_answers.extend([result])
                        all_ranks.append(rk+1)
                        all_models.append("Tf_Idf_Cluster_Bert_Finetuned_MsMarco")
                        all_querries.append(self.querries_raw[question])
                        rk+=1
                        special_answer[question]+=1
                        count_answers+=1
            
            ###Normal Loop Spherical Kmeans Clustering and Applying Bert encoder###
            #1 Get clusters
            #print('begin loop cluster retriever')
            for answ in range(FLAGS.size_cluster):
                while True:
                    answer_raw=self.content_repo[raw_or_treated][rank[question,count]]
                    answer=self.content_repo[1-raw_or_treated][rank[question,count]]
                    dic_answers[answer]=dic_answers.get(answer,0)+1
                    if dic_answers[answer]==1:
                        break
                    cluster_ids.append(metadata[0]+rank[question,count])
                    count+=1
                
                answers.append(answer)
                answers_raw.append(answer_raw)

                #print('Anserw n°{} : {}'.format(count_answers,answer))
                #print('\n----------------\n')
            self.all_answers_retrieved.append(answers)
            self.all_answers_retrieved_raw.append(answers_raw)
            
            #print('end loop cluster retriever')

            #2) Use Bert encoder inside clusters + cosine similarity retrieval
            Qst=questions[question].lower()
            newQst=remove_non_alpha(Qst)
            newQst=newQst.replace('.','')
            all_querries_treated.append(newQst)
            
        #Prediction with tensorflow app using bert finetuned on re-ranking MsMarco dataset
        
        o=len(all_querries_treated)
        tokenizer = FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)
        docs=[tuple(array) for array in self.all_answers_retrieved]
        query_id=[str(i) for i in range(o)]
        doc_ids=[tuple(array) for array in list(np.tile([str(i) for i in range(FLAGS.size_cluster) ],o).reshape(o,-1))]
        labels=[0 for i in range(FLAGS.size_cluster)]
        print("********* CLUSTER OF {}  DOCS ********** {} =  ".format(FLAGS.size_cluster, self.all_answers_retrieved_raw))
        
        convert_eval_dataset(tokenizer,
                             FLAGS.output_folder,
                             FLAGS.output_folder + '/query_doc_ids_' + 'eval' + '.txt',
                             FLAGS.max_seq_length,
                             FLAGS.max_query_length,
                             all_querries_treated,
                             docs,
                             labels,
                             query_id,
                             doc_ids)
        
        #print('begin reranking prediction with bert class')       
        print('entering run_msmarco')
        run_msmarco()
        print('end run_msmarco')
        #Transform tensorflow results to a nice DataFrame

        ranker=pd.read_csv(FLAGS.output_dir+'/msmarco_predictions_eval.tsv', sep='\t',names=['Q_id','Doc_id','Rank','Probs'])
        self.ranker=ranker[ranker.Rank<=5]
        
        for question in range(o):
            rk=0
            ranker_sub=self.ranker[self.ranker['Q_id']==question]
            scores_bert=ranker_sub['Probs'].tolist()
            indices=ranker_sub['Doc_id'].tolist()
            text=[self.all_answers_retrieved_raw[question][i] for i in indices]

            #print('end reranking prediction with bert class')
            for i,c in enumerate(text):

                #print('Anserws n° {} : {}'.format(i+1,c))
                #print('\n----------------\n')
                #print('rk=',rk+special_answer[question])
                all_ranks.append(rk+1+special_answer[question])
                all_models.append("Tf_Idf_Cluster_Bert_Finetuned_MsMarco")
                all_querries.append(self.querries_raw[question])
                all_question_ids.extend([question])
                rk+=1
            
            all_answers.extend(text)
            all_scores.extend(scores_bert)
            #print("ranker_sub.values",ranker_sub.values)
            #print("scores_bert",scores_bert)
            all_indices.extend(rank[question,:FLAGS.size_cluster][indices])
        #all_indices.extend(np.array(range(len(self.content_repo[0])))[rank[question,:self.top_n]])
        print('begin saving csv BERTTUNED')
        final=pd.DataFrame(np.c_[all_question_ids,all_querries,all_models,all_ranks,all_indices,all_answers, all_scores],columns=['Q_ids','Question','Model','Rank','Doc_index','Answer','Score'])
        print('end saving csv BERTTUNED')
        return final.sort_values(by=['Q_ids','Rank'],ascending=True).drop(columns=['Q_ids'])

    



class QApipeline():
    """
    ## kwargs :
    path_to_directory (str) : Directory to train language model tfidf
    ngram_range=(1,3) (tuple) : ngram range for tfidf words
    max_df=0.85 (float) : When building the vocabulary ignore terms that have a document frequency strictly 
                        ## higher than the given threshold (corpus-specific stop words). 
                        ## If float, the parameter represents a proportion of documents, integer absolute counts. 
                        ## This parameter is ignored if vocabulary is not None
    stop_words (str) : Language for tfidf (default 'english')
    paragraphs=None
    top=3 (int) : How many paragraphs to retrieve for querry matching (can be modified by threshold)
    verbose=False : print time of execution to build tfidf matrix and print errors
    MODEL_PATH = infersent_model 
    W2V_PATH = infersent_path  
    """
    
    def __init__(self,**kwargs):
        
        
        #new kwargs: 'threshold' (float between 0.5 and 1.0)
        
        self.kwargs_Tf_Idf = {key: value for key, value in kwargs.items()
                         if key in m3_Tfidf.__init__.__code__.co_varnames}

        self.kwargs_converter = {key: value for key, value in kwargs.items()
                            if key in pdfconverter.__init__.__code__.co_varnames}
        
        self.kwargs_Infersent={key: value for key, value in kwargs.items()
                         if key in m1_Infersent.__init__.__code__.co_varnames}
        self.kwargs_Bert={key: value for key, value in kwargs.items()
                         if key in m2_Bert.__init__.__code__.co_varnames}
        self.kwargs_others={key: value for key, value in kwargs.items()
                            if ((key not in pdfconverter.__init__.__code__.co_varnames) and (key not in m3_Tfidf.__init__.__code__.co_varnames))}
        
        
        
        self.usemodel=kwargs['usemodel']
        self.threshold=kwargs['threshold']
        print('************ READER *************')
        print("Reading pdfs doc on location: ",FLAGS.pdf_directory+FLAGS.domain+'/'+FLAGS.retrieved_company+'/pdfs/')
        self.df=pdfconverter(kwargs['directory_path'],FLAGS.retrieved_company).transform()
        self.l_questions=kwargs['l_questions']
        self.Qst_raw=self.l_questions.copy()
        self.sentences_chunk=kwargs['sentences_chunk']
        
        #Build vocabulary
        self.content =[] 
        self.voc=[]
        np.vectorize(lambda x: self.voc.extend(self.string_retriever(x.split())),otypes=[object])(np.array(self.content))
        self.voc=set(self.voc)
        
        #Extract all contents of repositories and index it 
        self.content=[]
        self.content_raw=[]
        self.contents_doc=[]
        self.borders=[0]
        print('********* DOCUMENTS RETRIEVED **********')
        for j,repo in enumerate(sorted(list(set(self.df.directory_index)))):
            dic_s=[{},{}]
            remove_idx=[[],[]]
            content_doc=[]
            content_doc_raw=[]
            title=self.df[self.df.directory_index==repo].directory.tolist()[0]
            self.df[self.df.directory_index==repo]['raw paragraphs'].apply(lambda x: self.update_dic(x,dic_s,0,remove_idx))
            self.df[self.df.directory_index==repo]['paragraphs'].apply(lambda x: self.update_dic(x,dic_s,1,remove_idx))
            self.df[self.df.directory_index==repo]['raw paragraphs'].apply(lambda x: content_doc_raw.extend(x))
            self.df[self.df.directory_index==repo]['paragraphs'].apply(lambda x: content_doc.extend(x))
            
            #Use treated text length to cap text size and raw_text size
            remove_idx=list(set(remove_idx[0]+remove_idx[1]))
            content_doc=np.delete(np.array(content_doc),remove_idx)
            content_doc_raw=np.delete(np.array(content_doc_raw),remove_idx)
            
            content=[content_doc[i] for i in range(len(content_doc)) if (len(content_doc[i])>=50 )]
            content_raw=[content_doc_raw[i] for i in range(len(content_doc)) if (len(content_doc[i])>=50)]
            self.borders.append(len(content))
            try:
                print("FOLDER : {} , {} sentences".format(self.df.directory.unique()[j],len(content)))
            except:
                print("FOLDER : {} , {} sentences".format("Similar reports",len(content)))
            self.content.extend(list(content))
            self.content_raw.extend(list(content_raw))
            self.contents_doc.append([content,content_raw])
            
        self.borders=list(np.cumsum(self.borders))
        if self.sentences_chunk>1:

            self.content=[ ' '.join(x) for x in zip(self.content[0::2], self.content[1::2]) ]
            self.content_raw=[ ' '.join(x) for x in zip(self.content_raw[0::2], self.content_raw[1::2]) ]
            for i,(L1,L2) in enumerate(self.contents_doc):
                self.contents_doc[i][0]=[ ' '.join(x) for x in zip(L1[0::2], L1[1::2]) ]
                self.contents_doc[i][1]=[ ' '.join(x) for x in zip(L2[0::2], L2[1::2]) ]
            self.borders=[int(i/2) for i in self.borders]

        #REPLACE DIGITS WITH SPECIAL TOKEN 
        for i,c in enumerate(self.borders[:-1]):
            a=self.borders[i]
            content=self.contents_doc[i][0]
            content_raw=self.contents_doc[i][1]

            for z,s in enumerate(content):
                l=s.split(" ")
                for j,w in enumerate(l):
                    try:
                        float(w)
                        l[j]="XXX"
                    except:
                        l[j]=w
                try:
                    self.content[a+z]=" ".join(l)
                except:
                    print("error! ",a,z,self.borders,len(self.content))
                self.contents_doc[i][0][z]=" ".join(l)
                
            for z,s in enumerate(content_raw):
                self.contents_doc[i][1][z]=s

                
    def update_dic(self,l,d,pos,r_index):
        for i,c in enumerate(l):
            
            d=d[pos].copy()
            d[c]=d.get(c,0)+1
            d[pos]=d
            if d[pos][c]>1:
                r_index[pos].append(i)
        return None
    
    def fit(self):
        MODELS=['INFERSENT - GLOVE','BERT PRETRAINED','TFIDF - LEMMATIZER & BIGRAM','BERT & TFIDF SHORT TEXT CLUSTERING','BERT FINETUNED & TFIDF SHORT TEXT CLUSTERING',' ALL MODELS (EVAL MODE)']
        print('********* MODEL {} **********'.format(MODELS[self.usemodel-1]))
        #We need to init Infersent at least for the querry builder
        self.inferst = m1_Infersent(**self.kwargs_Infersent)
        
        #Then use selected model to fit the data
        if self.usemodel in [1,6]:
            #Fit Infersent
            print('beginning of infersent fit')
            self.inferst.fit(self.content)
            self.inferst.transform(self.content)
            print('end of infersent fit')
        if self.usemodel in [2,6]:
            #Fit Bert pretrained
            print('beginning of BERT fit')
            self.bert=m2_Bert(**self.kwargs_Bert)
            self.bert.fit(self.content)#no finetuning for Bert
            self.bert.transform(self.content)
            print('end of BERT fit')
            
        if self.usemodel in [4,5,6]:
            #Fit Tf-Idf for mixture models or single tf-idf Lemmatize Bigram
            print('beginning of tf_idf fit')
            self.retriever = m3_Tfidf(**self.kwargs_Tf_Idf)
            self.retriever.fit(self.content)
            self.retriever.transform(self.content)
            print('end of tf_idf fit')
        
        
        if self.usemodel in [5,6]:
            #Fit TF-Idf BERT pretrained
            print("beginning of tf_idf bert (finetuned) fit")
            TF=self.retriever.tfidf_matrix
            dataQu=self.generate_querries(self.l_questions)
            querries=dataQu['words_sort'].tolist()
            for i,q in enumerate(querries):
                q=' '.join(list(q))
                q=q.lower()
                q=remove_non_alpha(q)
                q=q.replace('.','')
                querries[i]=q

            q_emb=self.retriever.vectorizer.transform(querries).toarray().transpose()
            content=self.content
            stemmed_content=[self.retriever.tokenize(s) for s in self.content]
            voc=self.retriever.vectorizer.get_feature_names()
            
            self.tf_idf_farahat=m5_harshQA(TF,q_emb,content,self.contents_doc.copy(),stemmed_content,voc,dataQu,self.Qst_raw)
            self.tf_idf_farahat.transform()
            print("end of tf_idf bert (finetuned) fit")
        return self
    
    
        #Initialisation of Tf-Idf-Farahat
    
    def predict(self,Qst,VE_type='DP',VE_cdt='',range_chunks=(5,20)):
        """
        kwargs:
        ##VE_type: 'DP' for Detect Presence of 'VE' for Value extraction
        ##Qst: Querry
        ##VE_cdt : null
        ##range_chunks: (tuple object) lower and upper bounds for text length
        """
    
        #Ask for the repository to question
        df_show=self.df[['directory']].drop_duplicates(subset=['directory']).reset_index(drop=True)
        repo=0
        
        min_,max_=range_chunks
        
        #Apply corpus transformations to querry for parcimony
        Qst=[q.lower() for q in self.Qst_raw]
        newQst=[remove_non_alpha(q) for q in Qst]
        newQst=[q.replace('.','') for q in newQst]
        
        
        #Infersent and Bert Loop Retriever ------ (look for most similar sentence using Infersent / Bert embeddings )
        self.dataframe=[]
        self.dataframe_infersent=[]
        self.dataframe_bert=[]
        self.dataframe_tf_farahat=[]
        self.dataframe_tf_farahat_tuned=[]

        if self.usemodel in [1,6] :
            
            #Infersent ranking
            print("beginning of infersent predict")
            all_scores=[]
            all_models=[]
            all_querries=[]
            all_ranks=[]
            all_indices=[]
            all_answers=[]
            for i,qu in enumerate(newQst):
                
                indices,scores=self.inferst.predict(qu,[self.borders[repo],self.borders[repo+1]])
                p=len(indices)
                try :
                    bbb=indices[1]
                except:
                    print('Fail')
                all_scores.extend(scores.loc[indices].values[:,0])
                all_answers.extend([ self.contents_doc[repo][1][i] for i in indices])
                all_models.extend(['Infersent Bi-Lstm']*p)
                all_ranks.extend(list(range(1,p+1)))
                all_querries.extend([self.Qst_raw[i]]*p)
                all_indices.extend(indices)
            
            self.dataframe_infersent=pd.DataFrame(np.c_[all_querries,all_models,all_ranks,all_indices,all_answers,all_scores],columns=['Question','Model','Rank','Doc_index','Answer','Score'])
            try :
                series=self.dataframe_infersent.iloc[0]
            except:
                print("PROBLEM DATA INFERSENT:",all_scores,all_answers,all_models,all_ranks,all_querries,all_indices)
            print("end of infersent predict")
                
        if self.usemodel in [2,6] :
            #Bert ranking
            print("beginning of bert predict")
            all_scores=[]
            all_models=[]
            all_querries=[]
            all_ranks=[]
            all_indices=[]
            all_answers=[]
            for i,qu in enumerate(newQst):
                indices,scores=self.bert.predict(qu,[self.borders[repo],self.borders[repo+1]])
                p=len(indices)
                all_scores.extend(scores.loc[indices].values[:,0])
                all_answers.extend([ self.contents_doc[repo][1][i] for i in indices])
                all_models.extend(['Bert - transformer']*p)
                all_ranks.extend(list(range(1,p+1)))
                all_querries.extend([self.Qst_raw[i]]*p)
                all_indices.extend(indices)
            self.dataframe_bert=pd.DataFrame(np.c_[all_querries,all_models,all_ranks,all_indices,all_answers,all_scores],columns=['Question','Model','Rank','Doc_index','Answer','Score'])
            print("end of bert predict")
    
            
        #Tf-Idf Farahat Ranking
        
        if self.usemodel in [5,6] :
            print('beginning of tf_idf bert (finetuned) predict')
            self.dataframe_tf_farahat_tuned=self.tf_idf_farahat.predict([self.borders[repo],self.borders[repo+1]],repo)
            print("enf of tf_idf bert (finetuned) predict")
            
        
        #Tf-Idf Loop Retriever (look for a variable sized textchunk which match the querry using Tf-Idf)  ------
        if self.usemodel==3:
            print("beginning of tf_idf predict")
            all_scores=[]
            all_models=[]
            all_querries=[]
            all_ranks=[]
            all_indices=[]
            all_answers=[]
            self.Q_emb=self.retriever.vectorizer.transform(newQst)
            dataQu=self.generate_querries(self.l_questions)
            querries=dataQu['words_sort'].tolist()
            for i,q in enumerate(querries):
                q=' '.join(list(q))
                q=q.lower()
                q=remove_non_alpha(q)
                q=q.replace('.','')
                querries[i]=q

            for iq,qu in enumerate(newQst):

                closest_docs_indices,scores = self.retriever.predict(querries[iq],[self.borders[repo],self.borders[repo+1]])
                self.test=[self.contents_doc[repo][1][u] for u in closest_docs_indices]
                self.test2=[self.contents_doc[repo][0][u] for u in closest_docs_indices]
                selected_extract=[self.contents_doc[repo][0][u] for u in closest_docs_indices]
                selection=pd.DataFrame(selected_extract,index=closest_docs_indices,columns=['extract'])

                extract_chunks=[]
                scores_chunks=[]
                contexts=[]

                #Searching for optimal textchunk size that match the querry 
                for ind in selection.index:
                    extract=selection.loc[ind,'extract']
                    temp_scores=[]
                    temp=[]

                    for chunksize in range(min_,max_+1):
                        array=extract.split(' ')
                        if len(array)<chunksize and len(array)>min_:
                            temp.append(extract)
                            embed=self.retriever.vectorizer.transform([extract])
                            temp_scores.append(embed.dot(self.Q_emb[iq].T).toarray()[0,0])
                        else:
                            for j in range(len(array)):
                                if j+chunksize<=len(array)-1:
                                    smallchunk=' '.join(array[j:j+chunksize]).lower()
                                    embed=self.retriever.vectorizer.transform([smallchunk])
                                    temp_scores.append(embed.dot(self.Q_emb[iq].T).toarray()[0,0])
                                    temp.append(smallchunk)

                    if temp_scores!=[]:
                        #print("temp_score=",temp_scores)
                        temp_scores=np.array(temp_scores)
                        pos,score=np.argmax(temp_scores),np.max(temp_scores)
                        scores_chunks.append(score)
                        extract_chunks.append(temp[pos])
                        contexts.append(ind)
                #print("scores_chunks=",scores_chunks)
                scores_chunks=np.array(scores_chunks)
                pos=(scores_chunks>self.threshold)
                scores_chunks=scores_chunks[pos]
                selected_indices=np.array(contexts)[pos]
                selected_extracts=np.array(extract_chunks)[pos]
                #selected_contexts=np.array(selection[selection.index==selected_indices]["extract"])
                selected_contexts=np.array(self.contents_doc[repo][1])[selected_indices]
                p=len(scores_chunks)

                all_models.extend(['Tf-Idf + Lemmatizer']*p)
                all_scores.extend(scores_chunks)
                all_querries.extend([self.Qst_raw[iq]]*p)
                idx_sort=np.argsort(-scores_chunks)
                all_ranks.extend((np.ones(p)+np.arange(p))[np.argsort(idx_sort)])
                all_indices.extend(selected_indices)
                all_answers.extend(selected_contexts)
                #print("mod,scores,querr,idx,rks,answ",len(all_models),len(all_scores),len(all_querries),len(all_indices),len(all_ranks),len(all_answers))

            #dataframe=pd.DataFrame(np.c_[selected_extracts,selected_contexts,selected_indices,scores_chunks],columns=['chunk','context','doc index','scores']).sort_values(by='scores',ascending=False)
            #['Question','Model','Rank','Doc_index','Answer','Score']
            self.dataframe=pd.DataFrame(np.c_[all_querries,all_models,all_ranks,all_indices,all_answers,all_scores],columns=['Question','Model','Rank','Doc_index','Answer','Score']).sort_values(by=['Question','Score'],ascending=False)
            print("end of tf_idf predict")
        
        
        #return closest_passages_chunks[pos],scores_chunks[pos]
        if FLAGS.model==6:
            return self.dataframe,self.dataframe_infersent,self.dataframe_bert,self.dataframe_tf_farahat,self.dataframe_tf_farahat_tuned
        else:
            output=[self.dataframe_infersent,self.dataframe_bert,self.dataframe,self.dataframe_tf_farahat,self.dataframe_tf_farahat_tuned]
            for i in range(len(output)):
                if i!=FLAGS.model-1:
                    output[i]=[]
                else:
                    output[i]=output[i].apply(self.add_ctxt,axis=1)
            return [output[j] for j in range(5)]

    def string_retriever(self,sentence_list):
        return [w  for w in sentence_list if not w.isdigit()]
    
    def add_ctxt(self,row):
        try:
            row['Context_Answer']=' '.join([self.contents_doc[0][1][int(row.Doc_index)-1],row.Answer,self.contents_doc[0][1][int(row.Doc_index)+1]])
        except:
            print('No context for index:',row.Doc_index)
            row['Context Answer']= ' '
        return row
    
    def generate_querries(self,querries):
        
        querries=[remove_non_alpha(q) for q in querries]    
        sentences=[s[0].lower()+s[1:].replace('?','')  for s in querries]
        i=0
        important_words=[]
        unsorted_words=[]

        for qu in sentences:

            #Get scores from infersent visualization function (max-pooling et each layer)
            tensor,vector,scores,words=self.inferst.infersent.visualize(qu)
            scores=np.array(scores[1:len(scores)-1])
            words=np.array(words[1:len(words)-1])

            #Remove stopwords from querries and attributed scores
            pos=[i for i,c in enumerate(list(words)) if c not in self.retriever.stop_words_list]
            words=words[pos]
            scores=np.array(scores)[pos]
            scores=scores/np.sum(scores)

            #Sort query words by word importance keeping idx in memory to unsort it back
            data=pd.DataFrame(np.c_[words,scores],columns=['word','score'])
            idx=np.argsort(-np.array(data.score.values,dtype='float64'))
            idx_unsort=np.argsort(idx)
            data=data.sort_values(by=['score'],ascending=False)
            new_words=data.word.values
            new_scores=np.array(data.score.values,dtype=float)

            #Keeping a set of words that satisfy 70% of cumulative importance
            score_cum=np.cumsum(new_scores)
            pos=score_cum<0.75
            lw=new_words[pos]
            ls=new_scores[pos]
            lw_unsort=words[pos[idx]]

            important_words.append(lw)
            unsorted_words.append(lw_unsort)
            i+=1
        
        array=np.zeros((len(querries),3),dtype=object)
        for i,c in enumerate(querries):
            array[i]=np.array([c,important_words[i],unsorted_words[i]])
            
        return pd.DataFrame(array,columns=['query','words_sort','words_unsort'])

        
            
        

        


# ### Tf-Idf/ FastText or Glove / Bert

# In[12]:


def choose_args():

    if FLAGS.demo:
        FLAGS.model=None
        FLAGS.retrieved_company=None
        FLAGS.domain=None
        
        for r,d,f in os.walk(FLAGS.pdf_directory):
            print([item for item in d if "." not in item ])
            break
        FLAGS.domain=str(input('select a domain\t'))
        for r,d,f in os.walk(FLAGS.pdf_directory+FLAGS.domain):
            print([item for item in d if "."  not in item ])
            break
        FLAGS.retrieved_company=str(input('select a company\t'))
        
        
        print('****Models implemented*****')
        models=[]
        models.append("Infersent_glove [Pretrained] (5 min/ company on approx 350 pages)")
        models.append("Bert [Pretrained on SQuAD] (30min/company on approx 350 pages) ")
        models.append("Tf_Idf_stemming [Trained on our corpus] (1 min/ company on approx 350 pages)")
        models.append("Short text Clustering + BERT Pretrained (30min/company on approx 350 pages)")
        models.append("Short text Clustering + BERT TUNED on RERANKING : harshQA :)  (20min/company on approx 350 pages)")
        models.append("EVAL / All models")
        df=pd.DataFrame(models,index=list(range(1,len(models)+1)),columns=['Models for ESG'])
      
        
        print(df.to_string())
        FLAGS.model=int(input("select a model entering his index\n"))
       
    
    if not FLAGS.demo:              
        path_q=FLAGS.pdf_directory+FLAGS.domain+'/'+"Queries.txt"
        file= open(path_q,"r+")  
        text=file.read().replace("  ","")
        queries=text.split("\n")
        queries=[q for q in queries if len(q)>1]
        file.close()
    else:
        s=str(input("enter a query \n \t"))
        queries=[s]
    
    
    args_Infersent={'directory_path':FLAGS.pdf_directory+FLAGS.domain+'/','w2v_path': FLAGS.infersent_path, 'model_path': FLAGS.infersent_model ,'top_n':FLAGS.top_n,'threshold':FLAGS.threshold,'usemodel':1,'ngram_range':(1, 1),'lemmatize':False,'transform_text':True,'l_questions':queries,'sentences_chunk':1}
    args_Bert={'directory_path':FLAGS.pdf_directory+FLAGS.domain+'/','w2v_path': FLAGS.infersent_path, 'model_path': FLAGS.infersent_model ,'top_n':FLAGS.top_n,'threshold':FLAGS.threshold,'usemodel':2,'ngram_range':(1, 1),'lemmatize':False,'transform_text':True,'l_questions':queries,'sentences_chunk':1}
    args_Tf_Idf={'directory_path':FLAGS.pdf_directory+FLAGS.domain+'/','w2v_path': FLAGS.infersent_path, 'model_path': FLAGS.infersent_model ,'top_n':FLAGS.top_n,'threshold':FLAGS.threshold,'usemodel':3,'ngram_range':(1, 2),'lemmatize':True,'transform_text':True,'l_questions':queries,'sentences_chunk':1}
    args_TfBERT={'directory_path':FLAGS.pdf_directory+FLAGS.domain+'/','w2v_path': FLAGS.infersent_path, 'model_path': FLAGS.infersent_model ,'top_n':FLAGS.top_n,'threshold':FLAGS.threshold,'usemodel':4,'ngram_range':(1, 1),'lemmatize':False,'transform_text':True,'l_questions':queries,'sentences_chunk':1}
    args_TfBERT_enhanced={'directory_path':FLAGS.pdf_directory+FLAGS.domain+'/','w2v_path': FLAGS.infersent_path, 'model_path': FLAGS.infersent_model ,'top_n':FLAGS.top_n,'threshold':FLAGS.threshold,'usemodel':5,'ngram_range':(1, 1),'lemmatize':False,'transform_text':True,'l_questions':queries,'sentences_chunk':1}
    args_All_transforms={'directory_path':FLAGS.pdf_directory+FLAGS.domain+'/','w2v_path': FLAGS.infersent_path, 'model_path': FLAGS.infersent_model ,'top_n':FLAGS.top_n,'threshold':FLAGS.threshold,'usemodel':6,'ngram_range':(1, 1),'lemmatize':False,'transform_text':True,'l_questions':queries,'sentences_chunk':1}

    if FLAGS.model==1:
        return args_Infersent
    elif FLAGS.model==2:
        return args_Bert
    elif FLAGS.model==3:
        return args_Tf_Idf
    elif FLAGS.model==4:
        return args_TfBERT
    elif FLAGS.model==5:
        return args_TfBERT_enhanced
    else:
        return args_All_transforms

def main(_):

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    assert FLAGS.pdf_directory!=None, 'Enter a pdf_directory path'
    assert  FLAGS.domain!=None , 'Enter a domain'
    assert  FLAGS.retrieved_company!=None, "Enter a company"


    writer = tf.python_io.TFRecordWriter(FLAGS.output_folder + '/dataset_' + 'eval' + '.tf')
    dic_suffix={1:'FASTEXT',2:'BERT',3:'TFIDF',4:'BERTCLUST',5:'BERTCLUST_TUNED',6:'EVAL'}
    args=choose_args()
    QAmodel=QApipeline(**args) 
    QAmodel.fit()
    
    data1,data2,data3,data4,data5=QAmodel.predict(args['l_questions'], VE_type='DP',range_chunks=(8,35))
    
    
    list_questions_results=QAmodel.l_questions
    dic_questions={}
    for i,s in enumerate(list_questions_results):
        dic_questions[s]=i+1

    frames=[data1,data2,data3,data4,data5]
    result = pd.concat([data for data in frames if len(data)>0])
    result['Rank']=result['Rank'].map(lambda x: x[0])
    result['Score']=result['Score'].map(lambda x: np.round(float(x),4))
    result['Q_id']=result.apply(lambda x: dic_questions[x.Question],axis=1)
    result['Company']=[FLAGS.retrieved_company]*len(result)

    results=result.sort_values(by=['Q_id','Company','Model','Rank']).reset_index(drop=True)[['Question','Q_id','Company','Model','Answer','Rank','Score','Doc_index']]
    #results=results.apply(func_add_context,axis=1)
    if not FLAGS.demo:
        results.to_csv(FLAGS.pdf_directory+FLAGS.domain+'/'+FLAGS.retrieved_company+'/results/' +FLAGS.retrieved_company+dic_suffix[model]+'.csv')
    else:
        print('*******************************  RESULTS    ***************************************')
        print([ i for i in zip(results.Score.tolist(),result.Answer.tolist())])


if __name__ == "__main__":
  tf.compat.v1.app.run()


