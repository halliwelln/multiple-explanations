#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import utils
import random as rn
import RGCN

def get_neighbors(data_subset,node_idx):

    head_neighbors = tf.boolean_mask(data_subset,data_subset[:,0]==node_idx)
    tail_neighbors = tf.boolean_mask(data_subset,data_subset[:,2]==node_idx)

    neighbors = tf.concat([head_neighbors,tail_neighbors],axis=0)
    
    return neighbors

def get_computation_graph(head,rel,tail,data,num_relations):

    '''Get k hop neighbors (may include duplicates)'''

    neighbors_head = get_neighbors(data,head)
    neighbors_tail = get_neighbors(data,tail)

    all_neighbors = tf.concat([neighbors_head,neighbors_tail],axis=0)

    return all_neighbors

def replica_step(head,rel,tail,explanation,num_entities, num_relations):
    
    comp_graph = get_computation_graph(head,rel,tail,ADJACENCY_DATA,num_relations)

    adj_mats = utils.get_adj_mats(comp_graph, num_entities, num_relations)

    total_loss = 0.0

    for epoch in range(NUM_EPOCHS):

        with tf.GradientTape(watch_accessed_variables=False) as tape:

            tape.watch(masks)

            masked_adjs = [adj_mats[i] * tf.sigmoid(masks[i]) for i in range(num_relations)]

            before_pred = model([
                    ALL_INDICES,
                    tf.reshape(head,(1,-1)),
                    tf.reshape(rel,(1,-1)),
                    tf.reshape(tail,(1,-1)),
                    adj_mats
                    ]
                )

            pred = model([
                    ALL_INDICES,
                    tf.reshape(head,(1,-1)),
                    tf.reshape(rel,(1,-1)),
                    tf.reshape(tail,(1,-1)),
                    masked_adjs
                    ]
                )

            loss = - before_pred * tf.math.log(pred+0.00001)

            tf.print(f"current loss {loss}")

            total_loss += loss

        grads = tape.gradient(loss,masks)
        optimizer.apply_gradients(zip(grads,masks))

    current_pred = []
    current_scores = []

    for i in range(num_relations):

        mask_i = adj_mats[i] * tf.nn.sigmoid(masks[i])

        mask_idx = mask_i.values > THRESHOLD

        non_masked_indices = tf.gather(mask_i.indices[mask_idx],[1,2],axis=1)

        if tf.reduce_sum(non_masked_indices) != 0:

            rel_indices = tf.cast(tf.ones((non_masked_indices.shape[0],1)) * i,tf.int64)

            triple = tf.concat([non_masked_indices,rel_indices],axis=1)
            
            triple = tf.gather(triple,[0,2,1],axis=1)

            score_array = mask_i.values[mask_idx] 

            current_pred.append(triple)
            current_scores.append(score_array)

    current_scores = tf.concat([array for array in current_scores],axis=0)
    top_k_scores = tf.argsort(current_scores,direction='DESCENDING')[0:2]

    pred_exp = tf.reshape(tf.concat([array for array in current_pred],axis=0),(-1,3))
    pred_exp = tf.gather(pred_exp,top_k_scores,axis=0)

    true_exp = tf.squeeze(explanation,axis=0)

    jaccard_pred_i = utils.max_jaccard_tf(true_exp,pred_exp,UNK_ENT_ID,UNK_REL_ID)

    #print('jaccard_pred_i',jaccard_pred_i)

    for mask in masks:
        mask.assign(value=init_value)

    return total_loss, jaccard_pred_i, pred_exp

def distributed_replica_step(head,rel,tail,explanation,num_entities, num_relations):

    per_replica_losses, per_replica_jaccard, current_preds = strategy.run(replica_step,
        args=(head,rel,tail,explanation,num_entities, num_relations))

    reduce_loss = per_replica_losses / NUM_EPOCHS

    #tf.print(f"reduced loss {reduce_loss}")
    #tf.print(f"reduced jaccard {per_replica_jaccard}")

    return reduce_loss,per_replica_jaccard, current_preds

if __name__ == '__main__':

    import argparse
    import tensorflow as tf

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str,
        help='paul,french_royalty, etc')
    parser.add_argument('rule',type=str,
        help='spouse,aunt,...,full_data')
    parser.add_argument('num_epochs',type=int)
    parser.add_argument('embedding_dim',type=int)
    parser.add_argument('learning_rate',type=float)

    args = parser.parse_args()

    DATASET = args.dataset
    RULE = args.rule
    NUM_EPOCHS = args.num_epochs
    EMBEDDING_DIM = args.embedding_dim
    LEARNING_RATE = args.learning_rate

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
    THRESHOLD = .01

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    idx2ent = dict(zip(range(NUM_ENTITIES),entities))
    idx2rel = dict(zip(range(NUM_RELATIONS),relations))

    UNK_ENT_ID = ent2idx['UNK_ENT']
    UNK_REL_ID = rel2idx['UNK_REL']
    
    train2idx = utils.array2idx(X_train_triples,ent2idx,rel2idx)
    trainexp2idx = utils.array2idx(X_train_traces,ent2idx,rel2idx)
    
    test2idx = utils.array2idx(X_test_triples,ent2idx,rel2idx)
    testexp2idx = utils.array2idx(X_test_traces,ent2idx,rel2idx)

    ALL_INDICES = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    with strategy.scope():

        model = RGCN.get_RGCN_Model(
            num_entities=NUM_ENTITIES,
            num_relations=NUM_RELATIONS,
            embedding_dim=EMBEDDING_DIM,
            output_dim=OUTPUT_DIM,
            seed=SEED
        )

        model.load_weights(os.path.join('..','data','weights',DATASET,DATASET+'_'+RULE+'.h5'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        init_value = tf.random.normal(
                (1,NUM_ENTITIES,NUM_ENTITIES), 
                mean=0, 
                stddev=1, 
                dtype=tf.float32, 
                seed=SEED
            )

        masks = [tf.Variable(
            initial_value=init_value,
            name='mask_'+str(i),
            trainable=True) for i in range(NUM_RELATIONS)
        ]

    ADJACENCY_DATA = tf.concat([
        train2idx,
        trainexp2idx.reshape(-1,3),
        test2idx,
        testexp2idx.reshape(-1,3)
        ],axis=0
    )

    del train2idx
    del trainexp2idx

    best_preds = []

    tf_data = tf.data.Dataset.from_tensor_slices(
            (test2idx[:,0],test2idx[:,1],test2idx[:,2],testexp2idx)).batch(1)

    dist_dataset = strategy.experimental_distribute_dataset(tf_data)

    for head,rel,tail,explanation in dist_dataset:

        loss, _, current_preds = distributed_replica_step(head,rel,
            tail,explanation,NUM_ENTITIES, NUM_RELATIONS)

        best_preds.append(current_preds)

    best_preds = [array.numpy() for array in best_preds]

    out_preds = []

    for i in range(len(best_preds)):

        preds_i = utils.idx2array(best_preds[i],idx2ent,idx2rel)

        out_preds.append(preds_i)

    out_preds = np.array(out_preds,dtype=object)

    print(f'Rule: {RULE}')
    print(f'Num epochs: {NUM_EPOCHS}')
    print(f'Embedding dim: {EMBEDDING_DIM}')
    print(f'learning_rate: {LEARNING_RATE}')
    print(f'threshold {THRESHOLD}')

    np.savez(os.path.join('..','data','preds',
        DATASET,'gnn_explainer_'+DATASET+'_'+RULE+ '_'+'_preds.npz'),
        preds=out_preds
        )

    print('Done.')

