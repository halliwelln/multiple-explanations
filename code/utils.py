#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

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

    for i in range(len(true_exp)):

        current_trace =  true_exp[i]

        if (current_trace != unk).all(axis=1).sum() > 0:

            num_explanations += 1

    relevance_scores = np.zeros(num_explanations)

    for i in range(n):

        current_pred = pred_exp[i]

        for j in range(num_explanations):

            unpadded_traces = remove_padding_np(true_exp[j],unk_ent_id,unk_rel_id)

            unpadded_weights = true_weight[j][true_weight[j] != unk_weight_id]

            indices = (unpadded_traces == current_pred).all(axis=1)

            sum_weights = sum([float(num) for num in unpadded_weights[indices]])

            relevance_scores[j] += sum_weights

    max_idx = np.argmax(relevance_scores)
    
    max_relevance_score = relevance_scores[max_idx]

    total_sum = sum([float(weight) for weight in true_weight[max_idx] if weight != unk_weight_id])

    precision = max_relevance_score /n
    recall = max_relevance_score/total_sum
    
    return precision, recall

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

def remove_padding_np(exp,unk_ent_id, unk_rel_id):

    #unk = np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT'])
    unk = np.array([unk_ent_id, unk_rel_id, unk_ent_id],dtype=object)

    exp_mask = (exp != unk).all(axis=1)

    masked_exp = exp[exp_mask]

    return masked_exp

def remove_padding_tf(exp,unk_ent_id, unk_rel_id):

    #unk = tf.convert_to_tensor(np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT']))
    unk = tf.cast(
        tf.convert_to_tensor([unk_ent_id, unk_rel_id, unk_ent_id]),
        dtype=exp.dtype)

    exp_mask = tf.reduce_all(tf.math.not_equal(exp, unk),axis=1)

    masked_exp = tf.boolean_mask(exp,exp_mask)

    return masked_exp

def max_jaccard_np(current_traces,pred_exp,unk_ent_id,unk_rel_id,return_idx=False):

    ''''
    pred_exp must have shape[0] >= 1

    pred_exp: 2 dimensional (num_triples,3)

    '''
    
    jaccards = []
    
    for i in range(len(current_traces)):
        
        true_exp = remove_padding_np(current_traces[i],unk_ent_id,unk_rel_id)

        jaccard = jaccard_score_np(true_exp, pred_exp)

        jaccards.append(jaccard)
    
    if return_idx:
        return max(jaccards), np.argmax(jaccards)
    return max(jaccards)

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

def train_test_split_no_unseen(X, test_size=100, seed=0, allow_duplication=False, filtered_test_predicates=None):

    '''Taken from https://github.com/Accenture/AmpliGraph/blob/master/ampligraph/evaluation/protocol.py'''
    
    if type(test_size) is float:
        test_size = int(len(X) * test_size)

    rnd = np.random.RandomState(seed)

    if filtered_test_predicates:
        candidate_idx = np.isin(X[:, 1], filtered_test_predicates)
        X_test_candidates = X[candidate_idx]
        X_train = X[~candidate_idx]
    else:
        X_train = None
        X_test_candidates = X

    entities, entity_cnt = np.unique(np.concatenate([X_test_candidates[:, 0], 
                                                     X_test_candidates[:, 2]]), return_counts=True)
    rels, rels_cnt = np.unique(X_test_candidates[:, 1], return_counts=True)
    dict_entities = dict(zip(entities, entity_cnt))
    dict_rels = dict(zip(rels, rels_cnt))
    idx_test = []
    idx_train = []
    
    all_indices_shuffled = rnd.permutation(np.arange(X_test_candidates.shape[0]))

    for i, idx in enumerate(all_indices_shuffled):
        test_triple = X_test_candidates[idx]
        # reduce the entity and rel count
        dict_entities[test_triple[0]] = dict_entities[test_triple[0]] - 1
        dict_rels[test_triple[1]] = dict_rels[test_triple[1]] - 1
        dict_entities[test_triple[2]] = dict_entities[test_triple[2]] - 1

        # test if the counts are > 0
        if dict_entities[test_triple[0]] > 0 and \
                dict_rels[test_triple[1]] > 0 and \
                dict_entities[test_triple[2]] > 0:
            
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
            idx_train.append(idx)
            
    if len(idx_test) != test_size:
        # if we cannot get the test set of required size that means we cannot get unique triples
        # in the test set without creating unseen entities
        if allow_duplication:
            # if duplication is allowed, randomly choose from the existing test set and create duplicates
            duplicate_idx = rnd.choice(idx_test, size=(test_size - len(idx_test))).tolist()
            idx_test.extend(list(duplicate_idx))
        else:
            # throw an exception since we cannot get unique triples in the test set without creating 
            # unseen entities
            raise Exception("Cannot create a test split of the desired size. "
                            "Some entities will not occur in both training and test set. "
                            "Set allow_duplication=True," 
                            "remove filter on test predicates or "
                            "set test_size to a smaller value.")
  
    
    if X_train is None:
        X_train = X_test_candidates[idx_train]
    else:
        X_train_subset = X_test_candidates[idx_train]
        X_train = np.concatenate([X_train, X_train_subset])
    X_test = X_test_candidates[idx_test]
    
    X_train = rnd.permutation(X_train)
    X_test = rnd.permutation(X_test)

    return X_train, X_test
