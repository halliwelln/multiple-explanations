#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from scipy import sparse

def parse_ttl(file_name, max_padding):
    
    lines = []

    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line)

    ground_truth = []
    traces = []
    weights = []

    for idx in range(len(lines)):

        if "graph us:construct" in lines[idx]:

            split_source = lines[idx+1].split()

            source_rel = split_source[1].split(':')[1]

            source_tup = [split_source[0],source_rel,split_source[2]]

            weight = float(lines[idx+2].split()[2][1:5])

        exp_triples = []

        if 'graph us:where' in lines[idx]:

            while lines[idx+1] != '} \n':

                split_exp = lines[idx+1].split()

                exp_rel = split_exp[1].split(':')[1]

                exp_triple = [split_exp[0],exp_rel,split_exp[2]]

                exp_triples.append(exp_triple)

                idx+=1

        if len(source_tup) and len(exp_triples):

            if len(exp_triples) < max_padding:

                while len(exp_triples) != max_padding:

                    pad = np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT'])
                    exp_triples.append(pad)

            ground_truth.append(np.array(source_tup))
            traces.append(np.array(exp_triples))
            weights.append(weight)
            
    return np.array(ground_truth),np.array(traces),np.array(weights)

def get_data(data,rule):

    if rule == 'full_data':

        triples,traces = concat_triples(data, data['rules'])
        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()

    else:
        triples,traces = concat_triples(data, [rule])
        entities = data[rule+'_entities'].tolist()
        relations = data[rule+'_relations'].tolist()

    return triples,traces,entities,relations

def concat_triples(data, rules):

    triples = []
    traces = []

    for rule in rules:

        triple_name = rule + '_triples'
        traces_name = rule + '_traces'

        triples.append(data[triple_name])
        traces.append(data[traces_name])

    triples = np.concatenate(triples, axis=0)
    traces = np.concatenate(traces, axis=0)
    
    return triples, traces

def array2idx(dataset,ent2idx,rel2idx):
    
    if dataset.ndim == 2:
        
        data = []
        
        for head, rel, tail in dataset:
            
            head_idx = ent2idx[head]
            tail_idx = ent2idx[tail]
            rel_idx = rel2idx[rel]
            
            data.append((head_idx, rel_idx, tail_idx))

        data = np.array(data)

    elif dataset.ndim == 3:
        
        data = []

        for i in range(len(dataset)):
            
            temp_array = []
        
            for head,rel,tail in dataset[i,:,:]:

                head_idx = ent2idx[head]
                tail_idx = ent2idx[tail]
                rel_idx = rel2idx[rel]

                temp_array.append((head_idx,rel_idx,tail_idx))

            data.append(temp_array)
            
        data = np.array(data).reshape(-1,dataset.shape[1],3)

    return data

def idx2array(dataset,idx2ent,idx2rel):
    
    if dataset.ndim == 2:
        
        data = []
        
        for head_idx, rel_idx, tail_idx in dataset:
            
            head = idx2ent[head_idx]
            tail = idx2ent[tail_idx]
            rel = idx2rel[rel_idx]
            
            data.append((head, rel, tail))

        data = np.array(data)

    elif dataset.ndim == 3:
        
        data = []

        for i in range(len(dataset)):
            
            temp_array = []
        
            for head_idx, rel_idx, tail_idx in dataset[i,:,:]:

                head = idx2ent[head_idx]
                tail = idx2ent[tail_idx]
                rel = idx2rel[rel_idx]

                temp_array.append((head,rel,tail))

            data.append(temp_array)
            
        data = np.array(data).reshape(-1,dataset.shape[1],3)

    return data

def distinct(a):
    _a = np.unique(a,axis=0)
    return _a

def get_adj_mats(data,num_entities,num_relations):

    adj_mats = []

    for i in range(num_relations):

        data_i = data[data[:,1] == i]

        if not data_i.shape[0]:
            indices = tf.zeros((1,2),dtype=tf.int64)
            values = tf.zeros((indices.shape[0]))

        else:

            # indices = tf.concat([
            #         tf.gather(data_i,[0,2],axis=1),
            #         tf.gather(data_i,[2,0],axis=1)],axis=0)
            indices = tf.gather(data_i,[0,2],axis=1)

            indices = tf.py_function(distinct,[indices],indices.dtype)
            values = tf.ones((indices.shape[0]))

        sparse_mat = tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=(num_entities,num_entities)
            )

        sparse_mat = tf.sparse.reorder(sparse_mat)

        sparse_mat = tf.sparse.reshape(sparse_mat, shape=(1,num_entities,num_entities))

        adj_mats.append(sparse_mat)

    return adj_mats

def get_negative_triples(head, rel, tail, num_entities, random_state=123):

    cond = tf.random.uniform(tf.shape(head), 0, 2, dtype=tf.int64, seed=random_state)
    rnd = tf.random.uniform(tf.shape(head), 0, num_entities-1, dtype=tf.int64, seed=random_state)
    
    neg_head = tf.where(cond == 1, head, rnd)
    neg_tail = tf.where(cond == 1, rnd, tail)

    return neg_head, neg_tail
