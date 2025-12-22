import os
import random 
import pickle 
import itertools 
import numpy as np
import copy

from torchpack.utils.config import configs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from eval.eval_utils import get_latent_vectors


class MemoryElement:
    def __init__(self):
        self.env_idx = None
        self.global_offset = np.zeros(2)
        self.tuple_pairs = []
        self.scores = []
        self.cluster_inds = []

class Memory:
    def __init__(self):
        self.K = configs.train.memory.num_pairs # 256
        self.memories = []
        
        self.train_tuples = []

    def __len__(self):
        return len(self.train_tuples)

    def get_tuples(self, new_dataset_len = 0):
        tuples = copy.deepcopy(list(itertools.chain.from_iterable(self.train_tuples)))

        # Adjust id, positives, non_negatives to match dataset we'll be appending to 
        for t in tuples:
            t.id = t.id + new_dataset_len
            t.positives = t.positives + new_dataset_len
            t.non_negatives = t.non_negatives + new_dataset_len

        tuples_dict = {t.id: t for t in tuples}

        return tuples_dict

    def get_anchor_clusters(self, Q, Q_pos, max_K_cluster=256, NC_thresh=10):
        ''' Get anchor clusters from current training task.
            Q: n x d, n queries in dimension of d (x,y,feature)
            Q_pos: n x 2, locations of n queries
            Return: new_C_inds, scores
        '''
        # K-means clustering for queries
        Q = np.concatenate([Q, Q_pos], axis=-1)
        K_cluster = np.minimum(max_K_cluster, Q.shape[0] // NC_thresh)
        
        C_inds = []
        kmeans = KMeans(n_clusters=K_cluster)
        kmeans.fit(Q)
        for i in range(K_cluster):
            C_inds.append(np.where(kmeans.labels_ == i)[0])

        C_center = []
        for i in range(K_cluster):
            C_center.append(kmeans.cluster_centers_[i])
        C_center = np.stack(C_center, axis=0)
        
        ## debug: visualize cluters
        if configs.debug:
            plt.clf()
            color_map = get_cmap('viridis')
            colors = color_map(np.linspace(0, 1, K_cluster))
            for i in range(K_cluster):
                for ind in C_inds[i]:
                    plt.plot(Q_pos[ind, 0], Q_pos[ind, 1], color=colors[i], marker='.', ms=2)
                plt.plot(C_center[i, -2], C_center[i, -1], color=colors[i], marker='+')
            plt.savefig(os.path.join(configs.save_dir, 'origin_cluster.svg'))
        
        # Compute scores for queries
        new_C_inds, scores = [], []
        Q, C_center, C_center_pos = Q[:, :-2], C_center[:, :-2], C_center[:, -2:]
        Q = Q / np.linalg.norm(Q, axis=-1, keepdims=True)
        C_center = C_center / np.linalg.norm(C_center, axis=-1, keepdims=True)
        for i in range(len(C_inds)):
            # get cluster center by average
            scores_i = []
            for j in range(len(C_inds[i])):
                # score 1: similarity to the cluster center (representative)
                score1 = (np.dot(Q[C_inds[i][j]], C_center[i]) + 1) / 2
                # score 2: mean distance to other samples in the cluster (diversity)
                sims = []
                for k in range(len(C_inds[i])):
                    if k == j:
                        continue
                    sim = (np.dot(Q[C_inds[i][j]], Q[C_inds[i][k]]) + 1) / 2
                    sims.append(sim)
                score2 = 1 - np.mean(sims)
                scores_i.append(score1 + score2)
            scores_i = np.array(scores_i)
            if len(scores_i) > NC_thresh:
                top_k = np.argpartition(-scores_i, kth=NC_thresh)[:NC_thresh]
                new_C_inds_i = C_inds[i][top_k]
                new_scores_i = scores_i[top_k]
            else:
                new_C_inds_i = C_inds[i]
                new_scores_i = scores_i
            new_C_inds.append(new_C_inds_i)
            scores.append(new_scores_i)
        
        ## debug: visualize cluters
        if configs.debug:
            plt.clf()
            color_map = get_cmap('viridis')
            colors = color_map(np.linspace(0, 1, K_cluster))
            for i in range(K_cluster):
                for ind in new_C_inds[i]:
                    plt.plot(Q_pos[ind, 0], Q_pos[ind, 1], color=colors[i], marker='.', ms=2)
                plt.plot(C_center_pos[i, -2], C_center_pos[i, -1], color=colors[i], marker='+')
            plt.savefig(os.path.join(configs.save_dir, 'clean_cluster.svg'))
        return new_C_inds, scores
    
    def choose_new_replay_samples(self, Q, Q_DA, C_inds, scores, M, new_tuples, env_idx, alpha=1.0):
        '''
            Q: n x d, n queries in dimension of d (x,y,feature)
            Q_DA: n x d, n augmented queries
            C_inds: indices of anchor clusters
            scores: scores (=score 1 + score 2) of samples within anchor clusters
            M: * x d, history memories, i.e. coreset until now
            env_idx: environment index
            Return: memory_element
        '''
        # Compute scores
        Q_norm = np.linalg.norm(Q, axis=-1)
        Q_DA_norm = np.linalg.norm(Q_DA, axis=-1)
        M_norm = np.linalg.norm(M, axis=-1) if M is not None else None
        max_NC = 0
        for i in range(len(C_inds)):
            for j in range(C_inds[i].shape[0]):
                ind = C_inds[i][j]
                # score 3: uncertainty of model on the sample
                score3 = 1 - (np.dot(Q[ind], Q_DA[ind]) / (Q_norm[ind] * Q_DA_norm[ind]) + 1) / 2
                # score 4: similarity to memory
                if M is not None:
                    max_sim = -1
                    for k in range(M.shape[0]):
                        sim = (np.dot(Q[ind], M[k]) / (Q_norm[ind] * M_norm[k]) + 1) / 2
                        max_sim = sim if sim > max_sim else max_sim
                    score4 = max_sim
                else:
                    score4 = 0.0
                scores[i][j] = scores[i][j] + score3 + alpha * score4
            sort_inds = np.argsort(-scores[i])
            C_inds[i] = C_inds[i][sort_inds]
            scores[i] = scores[i][sort_inds]
            max_NC = len(scores[i]) if len(scores[i]) > max_NC else max_NC
        # Get candidate replay buffer: anchor
        replay_anchor_inds, replay_anchor_scores, replay_anchor_cluster_inds = [], [], []
        for i in range(max_NC):
            # sort by scores
            inds_i, scores_i, cluster_inds_i = [], [], []
            for j in range(len(C_inds)):
                if len(C_inds[j]) > i:
                    inds_i.append(C_inds[j][i])
                    scores_i.append(scores[j][i])
                    cluster_inds_i.append(j)
            inds_i = np.array(inds_i)
            scores_i = np.array(scores_i)
            cluster_inds_i = np.array(cluster_inds_i)
            sort_inds = np.argsort(-scores_i)
            inds_i = inds_i[sort_inds]
            scores_i = scores_i[sort_inds]
            cluster_inds_i = cluster_inds_i[sort_inds]
            for j in range(len(inds_i)):
                replay_anchor_inds.append(inds_i[j])
                replay_anchor_scores.append(scores_i[j])
                replay_anchor_cluster_inds.append(cluster_inds_i[j])
        # Get replay buffer: pair
        num_to_choose = self.K // (env_idx + 1)
        hist_size = 10
        hist, uniform_hist_vec = np.zeros(hist_size), np.ones(hist_size)/hist_size  # [0,0.1), [0.1,0.2),...,[0.9,1.0]
        selected_inds, num_selected = [], 0
        memory_element = MemoryElement()
        memory_element.env_idx = env_idx
        for i in range(len(replay_anchor_inds)):
            anchor_ind, anchor_score, anchor_cluster_ind = replay_anchor_inds[i], replay_anchor_scores[i], replay_anchor_cluster_inds[i]
            if anchor_ind in selected_inds:
                continue
            anchor_tuple = new_tuples[anchor_ind]
            # positive_idxs of anchor_idx
            pair_ind_possibilities = [x for x in anchor_tuple.positives if x not in selected_inds]
            # Check a valid positive pair is possible 
            if len(pair_ind_possibilities) == 0:
                continue
            # find pair sample
            if len(selected_inds) < hist_size:
                pair_ind = random.choice(pair_ind_possibilities)
            else:
                pair_ind, pair_ind_hist_sim = -1, -1.0
                for j in range(len(pair_ind_possibilities)):
                    tmp_hist = hist.copy()
                    for selected_ind in selected_inds:
                        sim = (np.dot(Q[pair_ind_possibilities[j]], Q[selected_ind]) / (Q_norm[pair_ind_possibilities[j]] * Q_norm[selected_ind]) + 1) / 2
                        hist_idx = np.minimum(int(sim * hist_size), hist_size-1)
                        tmp_hist[hist_idx] += 1
                    sim = np.dot(uniform_hist_vec, tmp_hist) / (np.linalg.norm(tmp_hist))
                    if sim > pair_ind_hist_sim:
                        pair_ind, pair_ind_hist_sim, hist = pair_ind_possibilities[j], sim, tmp_hist
            
            pair_tuple = new_tuples[pair_ind]
            selected_inds += [anchor_ind, pair_ind] # Prevent these being picked again
            num_selected += 1
            
            if len(selected_inds) == hist_size:
                for selected_ind1 in selected_inds:
                    for selected_ind2 in selected_inds:
                        sim = (np.dot(Q[selected_ind1], Q[selected_ind2]) / (Q_norm[selected_ind1] * Q_norm[selected_ind2]) + 1) / 2
                        hist_idx = np.minimum(int(sim * hist_size), hist_size-1)
                        hist[hist_idx] += 1

            memory_element.tuple_pairs.append([anchor_tuple, pair_tuple])
            memory_element.scores.append(anchor_score)
            memory_element.cluster_inds.append(anchor_cluster_ind)
            
            if num_selected == num_to_choose:
                break
            
        memory_element.scores /= np.sum(memory_element.scores)
        return memory_element

    def forget_history_replay_samples(self, M, M_DA, M_pos):
        '''
            M: m x d, samples in memory
            M_DA: m x d, augmented samples in memory
            M_pos: m x 2, locations of memory samples
        '''
        if M is None or M_DA is None or M_pos is None:
            return
        M = np.concatenate([M, M_pos], axis=-1)
        M_DA = np.concatenate([M_DA, M_pos], axis=-1)
        start_id, num_keep = 0, self.K // len(self.memories)
        assert len(self.memories[len(self.memories)-1].scores) == num_keep  # size of the current task
        for env_idx in range(len(self.memories)-1):
            memory = self.memories[env_idx]
            new_ids = np.arange(start_id, start_id+len(memory.scores))
            M_i = M[new_ids] / np.linalg.norm(M[new_ids], axis=-1, keepdims=True)
            M_DA_i = M_DA[new_ids] / np.linalg.norm(M_DA[new_ids], axis=-1, keepdims=True)
            # score 2: min distance to other memory samples
            M_sim_i = (np.matmul(M_i, M_i.T) + 1) / 2
            M_sim_i[np.diag_indices_from(M_sim_i)] = 0
            scores2 = 1 - np.max(M_sim_i, axis=-1, keepdims=False)
            # score 3: distance to augmented sample
            scores3 = []
            for j in range(M_i.shape[0]):
                score3 = 1 - (np.dot(M_i[j], M_DA_i[j]) + 1) / 2
                scores3.append(score3)
            scores3 = np.array(scores3)
            scores_i = scores2 + scores3
            scores_i = scores_i / np.sum(scores_i)  # score normalization
            # remove surplus memory samples
            memory.scores = np.multiply(memory.scores, scores_i)
            sort_inds = np.argpartition(-memory.scores, num_keep)
            sort_inds = sort_inds[:num_keep]
            memory.tuple_pairs = list(np.array(memory.tuple_pairs)[sort_inds])
            memory.scores = memory.scores[sort_inds]
            memory.scores /= np.sum(memory.scores)
            self.memories[env_idx] = memory
            ## debug: visualize selected memory samples in the current task
            if configs.debug:
                plt.clf()
                for i in range(len(memory.tuple_pairs)):
                    anchor, pair = memory.tuple_pairs[i][0], memory.tuple_pairs[i][1]
                    plt.plot(anchor.position[0] - memory.global_offset[0], anchor.position[1] - memory.global_offset[1], color='red', marker='.', ms=2)
                    plt.plot(pair.position[0] - memory.global_offset[0], pair.position[1] - memory.global_offset[1], color='green', marker='.', ms=2)
                plt.savefig(os.path.join(configs.save_dir, f'history_memory_samples_{env_idx}.svg'))
    
    def update_positive_non_negative_idx(self):
        # memory tupels
        self.train_tuples = []
        for env_idx in range(len(self.memories)):
            memory = self.memories[env_idx]
            start_idx = len(self.train_tuples)
            env_replaced_idx = np.arange(start_idx, start_idx+len(memory.tuple_pairs))
            # old / new ids
            env_tuples = list(itertools.chain.from_iterable([memory.tuple_pairs[i-start_idx] for i in env_replaced_idx]))
            old_idx = [t.id for t in env_tuples]
            new_idx = list(itertools.chain.from_iterable([2*x, 2*x + 1] for x in env_replaced_idx))
            assert(len(old_idx) == len(new_idx))
            # construct a dict to correspond ids with idxs
            old_to_new_id = {o:n for o,n in zip(old_idx, new_idx)}
            # Replace all the positives and non_negatives with new idx
            for idx, t in enumerate(env_tuples):
                positives = t.positives
                non_negatives = t.non_negatives

                new_id = old_to_new_id[t.id]
                new_positives = [old_to_new_id[p] for p in positives if p in old_to_new_id.keys()]
                new_non_negatives = [old_to_new_id[p] for p in non_negatives if p in old_to_new_id.keys()]

                t.id = new_id
                t.positives = np.sort(new_positives)
                t.non_negatives = np.sort(new_non_negatives)

                env_tuples[idx] = t
            
            env_tuples_paired = [[env_tuples[x], env_tuples[x+1]] for x in list(range(len(env_tuples)))[::2]]
            assert len(env_replaced_idx) == len(env_tuples_paired)
            self.train_tuples += env_tuples_paired

    def update_memory(self, model, env_idx, max_K_cluster=20, NC_thresh=50, alpha=1.0):
        if env_idx == 0:
            new_pickle = configs.train.initial_environment
        else:
            new_pickle = configs.train.incremental_environments[env_idx-1]
        # Load new tuples
        new_tuples = pickle.load(open(new_pickle, 'rb'))
        # get anchor embeddings
        anchor_dataset, Q_pos = {}, []
        for i in range(len(new_tuples)):
            tuple_i = new_tuples[i]
            anchor_dataset[i] = {'query': tuple_i.rel_scan_filepath, 'northing': tuple_i.position[1], 'easting': tuple_i.position[0]}
            Q_pos.append(np.array([tuple_i.position[0], tuple_i.position[1]]))
        Q_pos = np.stack(Q_pos, axis=0)
        Q_pos_center = np.mean(Q_pos, axis=0, keepdims=True)
        Q_pos = Q_pos - Q_pos_center
        Q = get_latent_vectors(model, anchor_dataset)
        Q_DA = get_latent_vectors(model, anchor_dataset, aug_mode=1)
        # get anchor clusters (clustering and compute score 1 and score 2)
        C_inds, scores = self.get_anchor_clusters(Q, Q_pos, max_K_cluster, NC_thresh)
        # get memory embeddings
        M, M_DA, M_pos = None, None, None
        if len(self.memories) > 0:
            memory_dataset, M_pos = {}, []
            memory_tuples = list(itertools.chain.from_iterable([self.train_tuples[i] for i in range(len(self.train_tuples))]))
            for i in range(len(memory_tuples)):
                tuple_i = memory_tuples[i]
                memory_dataset[i] = {'query': tuple_i.rel_scan_filepath, 'northing': tuple_i.position[1], 'easting': tuple_i.position[0]}
                M_pos.append(np.array([tuple_i.position[0], tuple_i.position[1]]))
            M_pos = np.stack(M_pos, axis=0)
            M_pos = M_pos - np.mean(M_pos, axis=0, keepdims=True)
            M = get_latent_vectors(model, memory_dataset)
            M_DA = get_latent_vectors(model, memory_dataset, aug_mode=1)
        # choose new replay samples (compute score 3 and score 4, and choose new replay samples)
        new_memory = self.choose_new_replay_samples(Q, Q_DA, C_inds, scores, M, new_tuples, env_idx, alpha)
        new_memory.global_offset = Q_pos_center[0]
        ## debug: visualize selected memory samples in the current task
        if configs.debug:
            plt.clf()
            for i in range(len(new_memory.tuple_pairs)):
                anchor, pair = new_memory.tuple_pairs[i][0], new_memory.tuple_pairs[i][1]
                plt.plot(anchor.position[0] - Q_pos_center[0,0], anchor.position[1] - Q_pos_center[0,1], color='red', marker='.', ms=2)
                plt.plot(pair.position[0] - Q_pos_center[0,0], pair.position[1] - Q_pos_center[0,1], color='green', marker='.', ms=2)
            plt.savefig(os.path.join(configs.save_dir, 'current_memory_samples.svg'))
        self.memories.append(new_memory)
        # forget some history replay samples (update scores of memory samples and remove some samples with low scores)
        self.forget_history_replay_samples(M, M_DA, M_pos)
        # update training tuples (including positive and non-negative indices)
        self.update_positive_non_negative_idx()
