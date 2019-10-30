import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
import scipy.io as sio
import numpy as np
import json
from pprint import pprint
from scipy.spatial.distance import pdist

import os
import collections
import cPickle as pickle

def load_pickle(filename):
    fp = open(filename, "rb")
    return pickle.load(fp)

def load_year_words(word_file, years):
    word_pickle = load_pickle(word_file)
    word_lists = {}
    if not years[0] in word_pickle:
        if type(word_pickle) == dict or type(word_pickle) == collections.Counter:
            word_pickle = sorted(word_pickle, key = lambda word : word_pickle[word], reverse=True) 
        for year in years:
            word_lists[year] = word_pickle
    else:
        for year in years:
            word_list = word_pickle[year]
            if type(word_list) == dict or type(word_list) == collections.Counter:
                word_list = sorted(word_list.keys(), key = lambda word : word_list[word], reverse=True) 
            word_lists[year] = word_list
    return word_lists

def load_target_context_words(years, word_file, num_target, num_context):
    if num_context == 0:
        num_context = None
    if num_target == -1:
        num_target = num_context
    elif num_context == -1:
        num_context = num_target
    if num_target > num_context and num_context != None:
        raise Exception("Cannot have more target words than context words")
    word_lists = load_year_words(word_file, years)
    target_lists = {}
    context_lists = {}
    if num_context != -1:
        for year in years:
            context_lists[year] = word_lists[year][:num_context]
            target_lists[year] = word_lists[year][:num_target]
    else:
        context_lists = target_lists = word_lists
    return target_lists, context_lists

years = range(1850, 2000 + 1, 10)

target_lists,context_lists = load_target_context_words(years, "/home/beck/Repositories/Data/coha-word_sgns/sgns/years-vocab.pkl", -1, -1)
print(target_lists)
print()
#print(context_lists)