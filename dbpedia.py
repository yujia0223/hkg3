# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 11:13:06 2021

@author: admin
"""
import numpy as np 
import pandas as pd
from hlda.sampler import HierarchicalLDA
from datetime import datetime

from evaluationdb import cluster_entities
import networkx as nx



# load the raw data from directory
raw_data = pd.read_csv('data/dbpedia/triples.txt', delimiter = '\t')
raw_data = raw_data.to_numpy()
print('---------- load the data ----------')

# load the labels for evaluation
reference_data = pd.read_csv('data/dbpedia/classes.txt', delimiter = '\t')
subjects = set(raw_data[:,0])
for i in range(len(reference_data)):
    tmpt = reference_data.subject[i]
    if tmpt not in subjects:
        reference_data = reference_data.drop(i)

# preprocessing to get the documents and set of relation and tail       
relation_list = []
all_doc = []
vocab = set()
all_tail_doc = []
tail_vocab = set()
new_data = []

for s in reference_data.subject:
    if (raw_data[:,0] == s).any():
        index = np.where(raw_data[:,0]==s)[0]
        new_doc = []
        new_tail_doc = []
        new_data.append(raw_data[index])
    
        for i in index:
            new_doc.append(raw_data[i,1])
            new_tail_doc.append(raw_data[i,2])
    
    #         print(raw_data[i,1])
    #         print(raw_data[i,0])
        all_doc.append(new_doc)
        all_tail_doc.append(new_tail_doc)
        vocab.update(new_doc)
        tail_vocab.update(new_tail_doc)


vocab = sorted(list(vocab))
tail_vocab = sorted(list(tail_vocab))

print(vocab[0:10])
vocab_index = {}
for i, w in enumerate(vocab):
    vocab_index[w] = i
    
tail_vocab_index = {}
for i, w in enumerate(tail_vocab):
    tail_vocab_index[w] = i

# get the corpus of relation and tail    
new_corpus = []
for doc in all_doc:
    new_doc = []
    for word in doc:
        word_idx = vocab_index[word]
        new_doc.append(word_idx)
    new_corpus.append(new_doc)

tail_corpus = []
for doc in all_tail_doc:
    new_doc = []
    for word in doc:
        word_idx = tail_vocab_index[word]
        new_doc.append(word_idx)
    tail_corpus.append(new_doc)
print(len(tail_corpus))
print(len(tail_vocab))
for i in range(len(new_corpus)):
    if len(new_corpus[i]) != len(tail_corpus[i]):
        print('ss')

# get the all new triples facts 
new_data_df = pd.DataFrame()
for i in range(len(new_data)):
    temp = pd.DataFrame(new_data[i])
    new_data_df = new_data_df.append(temp,ignore_index=True)
print(new_data_df)

# get the all entities
union_list = list(set().union(tail_vocab, list(reference_data.subject.values)))
print(len(union_list))
intersection_set = set.intersection(set(tail_vocab), set(list(reference_data.subject.values)))
intersection_list = list(intersection_set)
print(len(intersection_list))
# print(data)

# hyper-parameter setting
n_samples = 100  #1000    # no of iterations for the sampler
alpha = 5#10.0          # smoothing over level distributions
gamma = 1  #1.0           # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
eta = 0.1             # smoothing over topic-word distributions
eta_tail = 5

num_levels = 5       # the number of levels in the tree
display_topics = 10 #100  # the number of iterations between printing a brief summary of the topics so far
n_words = 3           # the number of most probable words to print for each topic after model estimation
with_weights = True  # whether to print the words with the weights

ARI_1, NMI_1, HMS_1 = [], [], []
ARI_2, NMI_2, HMS_2 = [], [], []

for iteration in range(5):
    start_time = datetime.now()
    # do your work here
    hlda = HierarchicalLDA(new_corpus, tail_corpus, vocab, tail_vocab, alpha=alpha, gamma=gamma, eta=eta, eta_tail = eta_tail, num_levels=num_levels)
    hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)
    end_time = datetime.now()
    print('=======Duration: {}'.format(end_time - start_time))

    #print(len(new_corpus))
    leaves = hlda.document_leaves
    # print(hlda.document_leaves)
    results = hlda.results
    allnodes = hlda.allnodes_dict

    # evaluation
    cluster_index = np.zeros((len(new_corpus),num_levels))
    for i in range(len(new_corpus)):

        cluster_index[i,num_levels-1] = hlda.document_leaves[i].node_id
        s = hlda.document_leaves[i]
        for j in range(num_levels-1):
            s = s.parent
            cluster_index[i,num_levels-1-j-1] = s.node_id
    print(len(set(cluster_index[:,1])))
    print(len(set(cluster_index[:,2])))
    print(len(set(cluster_index[:,3])))
    print(len(set(cluster_index[:,4])))
    path_df = pd.DataFrame(cluster_index)
    # path_df.to_csv('paths_{0}.csv'.format(iteration,alpha,gamma,eta, eta_tail))
    print('experiments times:', iteration)

    for level in range(1,num_levels):
        print('==============level {0} evaluation============='.format(level))
        ARI, homogeneity_score, completeness_score,v_measure_score, NMI = cluster_entities(cluster_index, level, reference_data)
        ARI_1.append(ARI)
        NMI_1.append(NMI)
        HMS_1.append(homogeneity_score)
    path_df.to_csv('paths_{0}_{1}_{2}_{3}_{4}_{5}.csv'.format(iteration,alpha,gamma,eta, eta_tail,NMI))
    #save the tree
    results_customers_id = hlda.results_customers
    results_customers_labels = hlda.labels

    g = nx.DiGraph()
    for i in range(len(results_customers_labels)):
        g.add_node(results_customers_labels[i][0], label=results_customers_labels[i][1], shape='circle')
    g.add_edges_from(results_customers_id)
    p = nx.drawing.nx_pydot.to_pydot(g)
    p.write_png('results/5_level_results_customers_{0}_{1}_{2}_{3}_{4}_{5}.png'.format(iteration,alpha,gamma,eta, eta_tail,NMI))
    # p.write_pdf('results/5_level_results_customers_{}.pdf'.format(iteration))
