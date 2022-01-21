#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

def get_pred(adj_mats,num_relations,tape,pred,top_k):
    
    scores = []
    
    for i in range(num_relations):
        
        adj_mat_i = adj_mats[i]
        
        for idx,score in enumerate(tape.gradient(pred,adj_mat_i.values).numpy()):
            if tf.abs(score) >= 0:
                scores.append((idx,i,score))

    top_k_scores = sorted(scores, key=lambda x : x[2],reverse=True)[:top_k]

    pred_triples = []
    
    for idx,rel,score in top_k_scores:
        
        indices =  adj_mats[rel].indices.numpy()[idx,1:]

        head,tail = indices

        pred_triple = [head,rel,tail]

        pred_triples.append(pred_triple)

    return np.array(pred_triples)

if __name__ == '__main__':

    import os
    import utils
    import random as rn
    import RGCN
    import argparse

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str,
        help='paul, french_royalty, etc')
    parser.add_argument('rule',type=str,
        help='spouse,brother,...,full_data')
    parser.add_argument('top_k',type=int,help='top k explanations')
    parser.add_argument('embedding_dim',type=int)

    args = parser.parse_args()

    DATASET = args.dataset
    RULE = args.rule
    TOP_K = args.top_k
    EMBEDDING_DIM = args.embedding_dim

    data = np.load(os.path.join('..','data',DATASET+'.npz'))

    triples,traces,weights,entities,relations = utils.get_data(data,RULE)

    MAX_PADDING = 2
    LONGEST_TRACE = utils.get_longest_trace(data, RULE)

    X_train_triples, X_train_traces,_,X_test_triples, X_test_traces, _ = utils.train_test_split_no_unseen(
        X=triples,E=traces,weights=weights,longest_trace=LONGEST_TRACE,max_padding=MAX_PADDING,
        test_size=.25,seed=SEED)

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    OUTPUT_DIM = EMBEDDING_DIM

    ALL_INDICES = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    idx2ent = dict(zip(range(NUM_ENTITIES),entities))
    idx2rel = dict(zip(range(NUM_RELATIONS),relations))

    train2idx = utils.array2idx(X_train_triples,ent2idx,rel2idx)
    trainexp2idx = utils.array2idx(X_train_traces,ent2idx,rel2idx)

    test2idx = utils.array2idx(X_test_triples,ent2idx,rel2idx)
    testexp2idx = utils.array2idx(X_test_traces,ent2idx,rel2idx)

    ADJACENCY_DATA = tf.concat([
        train2idx,
        trainexp2idx.reshape(-1,3),
        test2idx,
        testexp2idx.reshape(-1,3)
        ],axis=0
    )

    UNK_ENT_ID = ent2idx['UNK_ENT']
    UNK_REL_ID = rel2idx['UNK_REL']

    model = RGCN.get_RGCN_Model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED
    )

    model.load_weights(os.path.join('..','data','weights',DATASET,DATASET+'_'+RULE+'.h5'))

    ADJ_MATS = utils.get_adj_mats(ADJACENCY_DATA,NUM_ENTITIES,NUM_RELATIONS)

    tf_data = tf.data.Dataset.from_tensor_slices(
            (test2idx[:,0],test2idx[:,1],test2idx[:,2],testexp2idx)).batch(1)

    pred_exps = []

    for head, rel, tail, true_exp in tf_data:

        with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape:

            tape.watch(ADJ_MATS)
    
            pred = model([
                ALL_INDICES,
                tf.reshape(head,(1,-1)),
                tf.reshape(rel,(1,-1)),
                tf.reshape(tail,(1,-1)),
                ADJ_MATS
                ]
            )

        pred_exp = get_pred(ADJ_MATS,NUM_RELATIONS,tape,pred,TOP_K)

        pred_exps.append(pred_exp)

    preds = np.array(pred_exps)

    best_preds = utils.idx2array(preds,idx2ent,idx2rel)

    print(f'Embedding dim: {EMBEDDING_DIM}')
    print(f"{DATASET} {RULE}")

    np.savez(os.path.join('..','data','preds',DATASET,'explaine_'+DATASET+'_'+RULE+'_preds.npz'),
        preds=best_preds
        )

    print('Done.')
