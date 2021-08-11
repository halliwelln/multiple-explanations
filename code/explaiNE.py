#!/usr/bin/env python3

import numpy as np

def get_pred(adj_mats,num_relations,tape,pred,true_exp):

    trace_length = true_exp.shape[0]
    
    scores = []
    
    for i in range(num_relations):
        
        adj_mat_i = adj_mats[i]
        
        for idx,score in enumerate(tape.gradient(pred,adj_mat_i.values).numpy()):
            if score:
                scores.append((idx,i,score))
                
    top_k_scores = sorted(scores, key=lambda x : x[2],reverse=True)[:trace_length]
    
    pred_triples = []
    
    for idx,rel,score in top_k_scores:
        
        indices =  adj_mats[rel].indices.numpy()[idx,1:]

        head,tail = indices

        pred_triple = [head,rel,tail]

        pred_triples.append(pred_triple)

    return np.array(pred_triples)

if __name__ == '__main__':

    import tensorflow as tf
    import os
    import utils
    import random as rn
    import RGCN
    import argparse
    from sklearn.model_selection import KFold

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str,
        help='paul, royalty')
    parser.add_argument('rule',type=str,
        help='spouse,brother,...,full_data')
    parser.add_argument('embedding_dim',type=int)

    args = parser.parse_args()

    DATASET = args.dataset
    RULE = args.rule
    EMBEDDING_DIM = args.embedding_dim

    data = np.load(os.path.join('..','data',DATASET+'.npz'))

    triples,traces,weights,entities,relations = utils.get_data(data,RULE)

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    OUTPUT_DIM = EMBEDDING_DIM

    ALL_INDICES = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    idx2ent = dict(zip(range(NUM_ENTITIES),entities))
    idx2rel = dict(zip(range(NUM_RELATIONS),relations))

    triples2idx = utils.array2idx(triples,ent2idx,rel2idx)
    traces2idx = utils.array2idx(traces,ent2idx,rel2idx)

    model = RGCN.get_RGCN_Model(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        seed=SEED
    )

    model.load_weights(os.path.join('..','data','weights',DATASET,DATASET+'_'+RULE+'.h5'))

    kf = KFold(n_splits=3,shuffle=True,random_state=SEED)

    cv_scores = []
    cv_preds = []
    test_indicies = []

    for train_idx,test_idx in kf.split(X=triples):

        pred_exps = []
        cv_jaccard = 0.0

        train2idx = triples2idx[train_idx]#utils.array2idx(triples[train_idx],ent2idx,rel2idx)
        trainexp2idx = traces2idx[train_idx]#utils.array2idx(traces[train_idx],ent2idx,rel2idx)

        test2idx = triples2idx[test_idx]#utils.array2idx(triples[test_idx],ent2idx,rel2idx)
        testexp2idx = traces2idx[test_idx]#utils.array2idx(traces[test_idx],ent2idx,rel2idx)

        ADJACENCY_DATA = tf.concat([
            train2idx,
            trainexp2idx.reshape(-1,3),
            test2idx,
            testexp2idx.reshape(-1,3)
            ],axis=0
        )

        adj_mats = utils.get_adj_mats(ADJACENCY_DATA,NUM_ENTITIES,NUM_RELATIONS)

        tf_data = tf.data.Dataset.from_tensor_slices(
                (test2idx[:,0],test2idx[:,1],test2idx[:,2],testexp2idx)).batch(1)

        for head, rel, tail, true_exp in tf_data:

            true_exp = utils.remove_padding_tf(true_exp[0],UNK_ENT_ID,UNK_REL_ID)

            with tf.GradientTape(watch_accessed_variables=False,persistent=True) as tape:

                tape.watch(adj_mats)
        
                pred = model([
                    ALL_INDICES,
                    tf.reshape(head,(1,-1)),
                    tf.reshape(rel,(1,-1)),
                    tf.reshape(tail,(1,-1)),
                    adj_mats
                    ]
                )

            pred_exp = get_pred(adj_mats,NUM_RELATIONS,tape,pred,true_exp)

            pred_exps.append(pred_exp)

            jaccard = utils.max_jaccard_np(true_exp,pred_exp,UNK_ENT_ID,UNK_REL_ID)
            cv_jaccard += jaccard

        cv_preds.append(pred_exps)
        cv_scores.append(cv_jaccard / test2idx.shape[0])
        test_indicies.append(test_idx)

    best_idx = np.argmax(cv_scores)
    best_test_indices = test_indicies[best_idx]

    preds = np.array(cv_preds[best_idx],dtype=object)

    preds = utils.pad(preds,UNK_ENT_ID,UNK_REL_ID)

    best_preds = utils.idx2array(preds,idx2ent,idx2rel)

    print(f'Embedding dim: {EMBEDDING_DIM}')

    np.savez(os.path.join('..','data','preds',DATASET,'explaine_'+DATASET+'_'+RULE+'_preds.npz'),
        preds=best_preds,best_idx=best_idx,test_idx=best_test_indices
        )

    print('Done.')
