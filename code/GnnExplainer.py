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

def get_computation_graph(head,rel,tail,k,data,num_relations):

    '''Get k hop neighbors (may include duplicates)'''

    neighbors_head = get_neighbors(data,head)
    neighbors_tail = get_neighbors(data,tail)

    all_neighbors = tf.concat([neighbors_head,neighbors_tail],axis=0)

    # if k > 1:
    #     num_indices = all_neighbors.shape[0]

    #     seen_nodes = []
        
    #     for _ in range(k-1):#-1 as we already computed 1st degree neighbors above

    #         for idx in range(num_indices):

    #             head_neighbor_idx = all_neighbors[idx,0]
    #             tail_neighbor_idx = all_neighbors[idx,2]

    #             if head_neighbor_idx not in seen_nodes:
                    
    #                 seen_nodes.append(head_neighbor_idx)

    #                 more_head_neighbors = get_neighbors(data,head_neighbor_idx)

    #                 all_neighbors = tf.concat([all_neighbors,more_head_neighbors],axis=0)

    #             if tail_neighbor_idx not in seen_nodes:

    #                 seen_nodes.append(tail_neighbor_idx)

    #                 more_tail_neighbors = get_neighbors(data,tail_neighbor_idx)

    #                 all_neighbors = tf.concat([all_neighbors,more_tail_neighbors],axis=0)

    return all_neighbors

def replica_step(head,rel,tail,explanation,num_relations):
    
    comp_graph = get_computation_graph(head,rel,tail,K,ADJACENCY_DATA,num_relations)

    adj_mats = utils.get_adj_mats(comp_graph, NUM_ENTITIES, num_relations)

    true_subgraphs = utils.get_adj_mats(tf.squeeze(explanation,axis=0),NUM_ENTITIES,num_relations)

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

    current_preds = []
    total_jaccard = 0.0

    for i in range(num_relations):

        mask_i = adj_mats[i] * tf.nn.sigmoid(masks[i])

        non_masked_indices = mask_i.indices[mask_i.values > THRESHOLD]

        if (non_masked_indices.shape[0] == 0) and (tf.math.reduce_all(true_subgraphs[i].values == tf.zeros((1,3)))):
            total_jaccard += 1.
        else:
            #total_jaccard += utils.jaccard_tf(true_subgraphs[i].indices,non_masked_indices)

        pred = non_masked_indices.numpy()

        current_preds.append(pred[:,1:])

    total_jaccard /= num_relations

    #tf.print(f"per observation jaccard: {total_jaccard}")

    for mask in masks:
        mask.assign(value=init_value)

    return total_loss, total_jaccard, current_preds

def distributed_replica_step(head,rel,tail,explanation,num_relations):

    per_replica_losses, per_replica_jaccard, current_preds = strategy.run(replica_step,
        args=(head,rel,tail,explanation,num_relations))

    reduce_loss = per_replica_losses / NUM_EPOCHS

    #tf.print(f"reduced loss {reduce_loss}")
    #tf.print(f"reduced jaccard {per_replica_jaccard}")

    return reduce_loss,per_replica_jaccard, current_preds

if __name__ == '__main__':

    import argparse
    from sklearn.model_selection import KFold
    import tensorflow as tf

    SEED = 123
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str,
        help='paul_dataset')
    parser.add_argument('rule',type=str,
        help='spouse,successor,...,full_data')
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

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)
    OUTPUT_DIM = EMBEDDING_DIM
    THRESHOLD = .01
    K = 1

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    idx2ent = dict(zip(range(NUM_ENTITIES),entities))
    idx2rel = dict(zip(range(NUM_RELATIONS),relations))

    triples2idx = utils.array2idx(triples,ent2idx,rel2idx)
    traces2idx = utils.array2idx(traces,ent2idx,rel2idx)

    UNK_ENT_ID = ent2idx['UNK_ENT']
    UNK_REL_ID = rel2idx['UNK_REL']

    relevance_scores = utils.get_relevance_scores(
        traces2idx,
        weights,
        UNK_ENT_ID,
        UNK_REL_ID)

    ALL_INDICES = tf.reshape(tf.range(0,NUM_ENTITIES,1,dtype=tf.int64), (1,-1))

    kf = KFold(n_splits=3,shuffle=True,random_state=SEED)

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

    cv_scores = []
    cv_preds = []
    test_indices = []

    for train_idx,test_idx in kf.split(X=triples):

        #test_idx = test_idx[0:2]

        preds = []

        train2idx = triples2idx[train_idx]
        trainexp2idx = traces2idx[train_idx]

        test2idx = triples2idx[test_idx]
        testexp2idx = traces2idx[test_idx]

        ADJACENCY_DATA = tf.concat([
            train2idx,
            trainexp2idx.reshape(-1,3),
            test2idx,
            testexp2idx.reshape(-1,3)
            ],axis=0
        )

        del train2idx
        del trainexp2idx

        TEST_SIZE = test2idx.shape[0]

        tf_data = tf.data.Dataset.from_tensor_slices(
            (test2idx[:,0],test2idx[:,1],test2idx[:,2],testexp2idx)).batch(1)

        dist_dataset = strategy.experimental_distribute_dataset(tf_data)

        total_jaccard = 0.0

        for head,rel,tail,explanation in dist_dataset:

            loss, jaccard, current_preds = distributed_replica_step(head,rel,tail,explanation,NUM_RELATIONS)
    
            preds.append(current_preds)

            total_jaccard += jaccard

        total_jaccard /= TEST_SIZE

        print(f"CV jaccard: {total_jaccard}")

        cv_scores.append(total_jaccard)
        cv_preds.append(preds)
        test_indices.append(test_idx)

    best_idx = np.argmax(cv_scores)
    best_test_indices = test_indices[best_idx]
    best_preds = cv_preds[best_idx]

    best_preds = []

    all_preds = np.array(cv_preds[best_idx],dtype=object)

    for i in range(len(all_preds)):

        preds_i = []

        for rel_idx in range(NUM_RELATIONS):

            triples_i = all_preds[i][rel_idx]

            if triples_i.shape[0]:
                rel_indices = (np.ones((triples_i.shape[0],1)) * rel_idx).astype(np.int64)
                concat = np.concatenate([triples_i,rel_indices],axis=1)
                preds_i.append(concat[:,[0,2,1]])
        preds_i = np.concatenate(preds_i,axis=0)
        best_preds.append(utils.idx2array(preds_i,idx2ent,idx2rel))

    best_preds = np.array(best_preds,dtype=object)

    print(f'Num epochs: {NUM_EPOCHS}')
    print(f'Embedding dim: {EMBEDDING_DIM}')
    print(f'learning_rate: {LEARNING_RATE}')
    print(f'threshold {THRESHOLD}')

    np.savez(os.path.join('..','data','preds',DATASET,'gnn_explainer_'+DATASET+'_'+RULE+'_preds.npz'),
        best_idx=best_idx, preds=best_preds,test_idx=best_test_indices
        )

    print('Done.')
