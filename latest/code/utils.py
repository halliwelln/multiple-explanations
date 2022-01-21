#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

def get_longest_trace(data, rule):

    if rule == 'full_data':
        longest_trace = data['max_trace']
    else:
        longest_trace = data[rule + '_longest_trace']

    return longest_trace

def graded_precision_recall(
    true_exp,
    pred_exp,
    true_weight,
    unk_ent_id,
    unk_rel_id,
    unk_weight_id):

    '''
    pred_exp: numpy array without padding
    true_exp: numpy array with padding

    '''
    
    n = len(pred_exp)
    
    unk = np.array([[unk_ent_id, unk_rel_id, unk_ent_id]])

    #first compute number of triples in explanation (must exclude padded triples)
    num_explanations = 0

    #number of triples per explanation
    num_true_triples = []

    for i in range(len(true_exp)):

        current_trace = true_exp[i]

        num_triples = (current_trace != unk).all(axis=1).sum()

        if  num_triples > 0:

            num_explanations += 1
            num_true_triples.append(num_triples)

    num_true_triples = np.array(num_true_triples)

    relevance_scores = np.zeros(num_explanations)

    for i in range(n):

        current_pred = pred_exp[i]

        for j in range(num_explanations):

            unpadded_traces = remove_padding_np(true_exp[j],unk_ent_id,unk_rel_id)

            unpadded_weights = true_weight[j][true_weight[j] != unk_weight_id]

            indices = (unpadded_traces == current_pred).all(axis=1)

            sum_weights = sum([float(num) for num in unpadded_weights[indices]])

            relevance_scores[j] += sum_weights

    precision_scores = relevance_scores / (n * .9)
    recall_scores = relevance_scores /  (num_true_triples * .9)

    nonzero_indices = (precision_scores + recall_scores) != 0

    if np.sum(nonzero_indices) == 0:
        f1_scores = [0.0]
    else:

        nonzero_precision_scores = precision_scores[nonzero_indices]
        nonzero_recall_scores = recall_scores[nonzero_indices]

        f1_scores = 2 * (nonzero_precision_scores * \
            nonzero_recall_scores) / (nonzero_precision_scores + nonzero_recall_scores)

    #f1_scores = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores + .000001)

    f1 = np.max(f1_scores)
    precision = np.max(precision_scores)
    recall = np.max(recall_scores)

    return precision, recall, f1

def pad_trace(trace,longest_trace,max_padding,unk):

    #unk = np.array([['UNK_ENT','UNK_REL','UNK_ENT']])
    
    unk = np.repeat(unk,[max_padding],axis=0)
    
    unk = np.expand_dims(unk,axis=0)
    
    while trace.shape[0] != longest_trace:
        trace = np.concatenate([trace,unk],axis=0)
        
    return trace

def pad_weight(trace,longest_trace,unk_weight):

    while trace.shape[0] != longest_trace:
        trace = np.concatenate([trace,unk_weight],axis=0)

    return trace

def f1(precision,recall):
    return 2 * (precision*recall) / (precision + recall)

def jaccard_score_np(true_exp,pred_exp):
        
    num_true_traces = true_exp.shape[0]
    num_pred_traces = pred_exp.shape[0]

    count = 0
    for pred_row in pred_exp:
        for true_row in true_exp:
            if (pred_row == true_row).all():
                count +=1

    score = count / (num_true_traces + num_pred_traces-count)
    
    return score

def jaccard_score_tf(true_exp,pred_exp):

    num_true_traces = tf.shape(true_exp)[0]
    num_pred_traces = tf.shape(pred_exp)[0]

    count = 0
    for i in range(num_pred_traces):

        pred_row = pred_exp[i]

        for j in range(num_true_traces):

            true_row = true_exp[j]

            count += tf.cond(tf.reduce_all(pred_row == true_row), lambda :1, lambda:0)

    score = count / (num_true_traces + num_pred_traces-count)
    
    return score

def remove_padding_np(exp,unk_ent_id, unk_rel_id,axis=1):

    #unk = np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT'])
    unk = np.array([unk_ent_id, unk_rel_id, unk_ent_id],dtype=object)

    exp_mask = (exp != unk).all(axis=axis)

    masked_exp = exp[exp_mask]

    return masked_exp

def remove_padding_tf(exp,unk_ent_id, unk_rel_id,axis=-1):

    #unk = tf.convert_to_tensor(np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT']))
    unk = tf.cast(
        tf.convert_to_tensor([unk_ent_id, unk_rel_id, unk_ent_id]),
        dtype=exp.dtype)

    exp_mask = tf.reduce_all(tf.math.not_equal(exp, unk),axis=axis)

    masked_exp = tf.boolean_mask(exp,exp_mask)

    return masked_exp

def max_jaccard_np(current_traces,pred_exp,true_weight,
    unk_ent_id,unk_rel_id,unk_weight_id,return_idx=False):

    ''''
    pred_exp must have shape[0] >= 1

    pred_exp: 2 dimensional (num_triples,3)

    '''
    
    jaccards = []
    sum_weights = []
    
    for i in range(len(current_traces)):
        
        true_exp = remove_padding_np(current_traces[i],unk_ent_id,unk_rel_id)

        weight = true_weight[i][true_weight[i] != unk_weight_id]

        sum_weight = sum([float(num) for num in weight])

        sum_weights.append(sum_weight)

        jaccard = jaccard_score_np(true_exp, pred_exp)

        jaccards.append(jaccard)

    max_indices = np.array(jaccards) == max(jaccards)

    if max_indices.sum() > 1:
        max_idx = np.argmax(max_indices * sum_weights)
        max_jaccard = jaccards[max_idx]
    else:
        max_jaccard = max(jaccards)
        max_idx = np.argmax(jaccards)
    
    if return_idx:
        return max_jaccard, max_idx
    return max_jaccard

def max_jaccard_tf(current_traces,pred_exp,unk_ent_id,unk_rel_id):

    '''pred_exp: 2 dimensional (num_triples,3)'''
    
    jaccards = []
    
    for i in range(len(current_traces)):
        
        trace = remove_padding_tf(current_traces[i],unk_ent_id,unk_rel_id)

        jaccard = jaccard_score_tf(trace, pred_exp)

        jaccards.append(jaccard)

    return max(jaccards)

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

        triples = data['all_triples']
        traces = data['all_traces'] 
        weights = data['all_weights']

        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()

    else:
        triples,traces,weights = concat_triples(data, [rule])
        entities = data[rule+'_entities'].tolist()
        relations = data[rule+'_relations'].tolist()

    return triples,traces,weights,entities,relations

def concat_triples(data, rules):

    triples = []
    traces = []
    weights = []

    for rule in rules:

        triple_name = rule + '_triples'
        traces_name = rule + '_traces'
        weights_name = rule + '_weights'

        triples.append(data[triple_name])
        traces.append(data[traces_name])
        weights.append(data[weights_name])

    triples = np.concatenate(triples, axis=0)
    traces = np.concatenate(traces, axis=0)
    weights = np.concatenate(weights,axis=0)
    
    return triples, traces, weights

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

    elif dataset.ndim == 4:

        data = []

        for i in range(len(dataset)):

            temp_array = []

            for j in range(len(dataset[i])):

                temp_array_1 = []

                for head,rel,tail in dataset[i,j]:

                    head_idx = ent2idx[head]
                    tail_idx = ent2idx[tail]
                    rel_idx = rel2idx[rel]

                    temp_array_1.append((head_idx,rel_idx,tail_idx))

                temp_array.append(temp_array_1)

            data.append(temp_array)

        data = np.array(data)

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

    elif dataset.ndim == 4:

        data = []

        for i in range(len(dataset)):

            temp_array = []

            for j in range(len(dataset[i])):

                temp_array_1 = []

                for head_idx, rel_idx, tail_idx in dataset[i,j]:

                    head = idx2ent[head_idx]
                    tail = idx2ent[tail_idx]
                    rel = idx2rel[rel_idx]

                    temp_array_1.append((head,rel,tail))

                temp_array.append(temp_array_1)

            data.append(temp_array)

        data = np.array(data)

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
    
def train_test_split_no_unseen(
    X,
    E,
    weights=None,
    longest_trace=None,
    max_padding=None,
    unk_ent_id='UNK_ENT',
    unk_rel_id='UNK_REL',
    test_size=.25,
    seed=123,
    allow_duplication=False):

    test_size = int(len(X) * test_size)

    np.random.seed(seed)

    X_train = None
    X_train_exp = None
    X_test_candidates = X
    X_test_exp_candidates = E

    if E.ndim == 4:

        exp_entities = np.array([
            [E[:,i,j,0],E[:,i,j,2]] for i in range(longest_trace) for j in range(max_padding)]).flatten()

        exp_relations = np.array([
            [E[:,i,j,1]] for i in range(longest_trace) for j in range(max_padding)]).flatten()
        
    elif E.ndim == 3:
        exp_entities = np.array([[E[:,i,:][:,0],E[:,i,:][:,2]] for i in range(max_padding)]).flatten()

        exp_relations = np.array([E[:,i,:][:,1] for i in range(max_padding)]).flatten()
        
    entities, entity_cnt = np.unique(np.concatenate([
                                X[:,0], X[:,2], exp_entities],axis=0),return_counts=True)
    rels, rels_cnt = np.unique(np.concatenate([
                                X[:,1], exp_relations],axis=0),return_counts=True)
    
    dict_entities = dict(zip(entities, entity_cnt))
    dict_rels = dict(zip(rels, rels_cnt))
    idx_test = []
    idx_train = []
    
    all_indices_shuffled = np.random.permutation(np.arange(X_test_candidates.shape[0]))

    for i, idx in enumerate(all_indices_shuffled):
        test_triple = X_test_candidates[idx]
        test_exp = remove_padding_np(X_test_exp_candidates[idx],unk_ent_id, unk_rel_id,axis=-1)
                
        # reduce the entity and rel count of triple
        dict_entities[test_triple[0]] = dict_entities[test_triple[0]] - 1
        dict_rels[test_triple[1]] = dict_rels[test_triple[1]] - 1
        dict_entities[test_triple[2]] = dict_entities[test_triple[2]] - 1
        
        exp_entities = np.concatenate([test_exp[:,0].flatten(),
                                       test_exp[:,2].flatten()])
        
        exp_rels = test_exp[:,1]
        
        # reduce the entity and rel count of explanation
        for exp_ent in exp_entities:
            dict_entities[exp_ent] -= 1
            
        for exp_rel in exp_rels:
            dict_rels[exp_rel] -= 1
            
        ent_counts = []
        for exp_ent in exp_entities:
            count_i = dict_entities[exp_ent]
            
            if count_i > 0:
                ent_counts.append(1)
            else:
                ent_counts.append(0)
                
        rel_counts = []
        for exp_rel in exp_rels:
            count_i = dict_rels[exp_rel]
            
            if count_i > 0:
                rel_counts.append(1)
            else:
                rel_counts.append(0)
        
        #compute sums and determine if counts > 0

        # test if the counts are > 0
        if dict_entities[test_triple[0]] > 0 and \
                dict_rels[test_triple[1]] > 0 and \
                dict_entities[test_triple[2]] > 0 and \
                sum(ent_counts) == len(ent_counts) and \
                sum(rel_counts) == len(rel_counts):
            
            # Can safetly add the triple to test set
            idx_test.append(idx)
            if len(idx_test) == test_size:
                # Since we found the requested test set of given size
                # add all the remaining indices of candidates to training set
                idx_train.extend(list(all_indices_shuffled[i + 1:]))
                
                # break out of the loop
                break
            
        else:
            # since removing this triple results in unseen entities, add it to training
            dict_entities[test_triple[0]] = dict_entities[test_triple[0]] + 1
            dict_rels[test_triple[1]] = dict_rels[test_triple[1]] + 1
            dict_entities[test_triple[2]] = dict_entities[test_triple[2]] + 1
            
            for exp_ent in exp_entities:
                dict_entities[exp_ent] += 1
            
            for exp_rel in exp_rels:
                dict_rels[exp_rel] += 1
            
            idx_train.append(idx)
            
    if len(idx_test) != test_size:
        # if we cannot get the test set of required size that means we cannot get unique triples
        # in the test set without creating unseen entities
        if allow_duplication:
            # if duplication is allowed, randomly choose from the existing test set and create duplicates
            duplicate_idx = np.random.choice(idx_test, size=(test_size - len(idx_test))).tolist()
            idx_test.extend(list(duplicate_idx))
        else:
            # throw an exception since we cannot get unique triples in the test set without creating 
            # unseen entities
            raise Exception("Cannot create a test split of the desired size. "
                            "Some entities will not occur in both training and test set. "
                            "Set allow_duplication=True," 
                            "or set test_size to a smaller value.")

    X_train = X_test_candidates[idx_train]
    X_train_exp = X_test_exp_candidates[idx_train]
    
    X_test = X_test_candidates[idx_test]
    X_test_exp = X_test_exp_candidates[idx_test]
    
    #shuffle data
    
    idx_train_shuffle = np.random.permutation(np.arange(len(idx_train)))
    idx_test_shuffle = np.random.permutation(np.arange(len(idx_test)))
    
    X_train = X_train[idx_train_shuffle]
    X_train_exp = X_train_exp[idx_train_shuffle]
    
    X_test = X_test[idx_test_shuffle]
    X_test_exp = X_test_exp[idx_test_shuffle]
    
    if weights is not None:
        
        X_train_weights = weights[idx_train]
        X_test_weights = weights[idx_test]
        
        X_train_weights = X_train_weights[idx_train_shuffle]
        X_test_weights = X_test_weights[idx_test_shuffle]
                
        return X_train, X_train_exp,X_train_weights,\
               X_test, X_test_exp, X_test_weights
    
    return X_train, X_train_exp, X_test, X_test_exp
