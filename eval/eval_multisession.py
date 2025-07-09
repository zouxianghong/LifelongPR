import pickle 
import numpy as np 
from tqdm import tqdm 
from eval.eval_utils import get_latent_vectors
from sklearn.neighbors import KDTree

import os
import argparse
import torch
from models.model_factory import model_factory
from torchpack.utils.config import configs
import matplotlib
from matplotlib import pyplot as plt

def eval_multisession(model, database_sets, query_sets, scene_id):
    recall = np.zeros(25)
    count = 0
    similarity = [] 
    all_correct = []
    all_incorrect = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    database_sets = pickle.load(open(database_sets, 'rb'))
    for database_set in database_sets:
        for key in database_set:
            database_set[key]['scene_id'] = scene_id
    
    query_sets = pickle.load(open(query_sets, 'rb'))
    for query_set in query_sets:
        for key in query_set:
            query_set[key]['scene_id'] = scene_id

    for run in tqdm(database_sets, disable=False, desc = 'Getting database embeddings'):
        database_embeddings.append(get_latent_vectors(model, run))

    for run in tqdm(query_sets, disable=False, desc = 'Getting query embeddings'):
        query_embeddings.append(get_latent_vectors(model, run))

    for i in tqdm(range(len(query_sets)), desc = 'Getting Recall'):
        for j in range(len(query_sets)):
            if i == j:
                continue 
            pair_recall, pair_similarity, pair_opr, correct, incorrect = get_recall(
                i, j, database_embeddings, query_embeddings, query_sets, database_sets
            )
            recall += np.array(pair_recall)
            count += 1 
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)
            for x in correct:
                all_correct.append(x)
            for x in incorrect:
                all_incorrect.append(x)
    
    ave_recall = recall / count
    ave_recall_1 = ave_recall[0]
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'Recall@1%': ave_one_percent_recall, 'Recall@1': ave_recall_1, 'Recall@N': ave_recall}
    return stats


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code

    database_output = database_vectors[m]
    queries_output = query_vectors[n]


    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    correct = []
    incorrect = []

    num_evaluated = 0

    for i in range(len(queries_output)): #size 
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                    correct.append(similarity)
                recall[j] += 1
                break
            else:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    incorrect.append(similarity)

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    
    return recall, top1_similarity_score, one_percent_recall, correct, incorrect

# TODO Write code to evaluate an individual run 

def get_res_on_oxford(model, cmp_dir, save_file):
    res = {}
    
    database_files = configs.eval.environments['Oxford']['database_files']
    query_files = configs.eval.environments['Oxford']['query_files']
    database_sets = pickle.load(open(database_files[0], 'rb'))
    query_sets = pickle.load(open(query_files[0], 'rb'))
    
    for database_set in database_sets:
        for key in database_set:
            database_set[key]['scene_id'] = -1
    
    for query_set in query_sets:
        for key in query_set:
            query_set[key]['scene_id'] = -1
    
    database_embeddings, query_embeddings = [], []
    for run in tqdm(database_sets, disable=False, desc = 'Getting database embeddings'):
        database_embeddings.append(get_latent_vectors(model, run))

    for run in tqdm(query_sets, disable=False, desc = 'Getting query embeddings'):
        query_embeddings.append(get_latent_vectors(model, run))

    for i in tqdm(range(len(query_sets)), desc = 'Getting Recall'):
        for j in range(len(query_sets)):
            if i == j:
                continue
            database_output = database_embeddings[i]
            queries_output = query_embeddings[j]
            
            database_nbrs = KDTree(database_output)
            
            for k in range(len(queries_output)): #size 
                # i is query element ndx
                query_details = query_sets[j][k]    # {'query': path, 'northing': , 'easting': }
                true_neighbors = query_details[i]
                if len(true_neighbors) == 0:
                    continue
                _, indices = database_nbrs.query(np.array([queries_output[k]]), k=1)
                top1_idx = indices[0][0]  # top 1
                top1_state = 1 if top1_idx in true_neighbors else 0
                top1_file = database_sets[i][top1_idx]['query']
                res_jik = {'query': query_sets[j][k]['query'],
                           'top1_state': top1_state,
                           'top1_file': top1_file}
                res['{}_{}_{}'.format(j,i,k)] = res_jik
    # save file
    if not os.path.exists(cmp_dir):
        os.makedirs(cmp_dir)
    save_path = os.path.join(cmp_dir, save_file)
    with open(save_path, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return res


def compare_res_on_oxford(res_oxford, res_FT_seq1, res_Our_seq1, res_FT_seq2, res_Our_seq2, cmp_dir):
    res_oxford = pickle.load(open(res_oxford, 'rb'))
    res_FT_seq1 = pickle.load(open(res_FT_seq1, 'rb'))
    res_Our_seq1 = pickle.load(open(res_Our_seq1, 'rb'))
    res_FT_seq2 = pickle.load(open(res_FT_seq2, 'rb'))
    res_Our_seq2 = pickle.load(open(res_Our_seq2, 'rb'))

    for key in res_oxford:
        if res_oxford[key]['top1_state'] == 0:
            continue
        if res_Our_seq1[key]['top1_state'] == 0 and res_Our_seq2[key]['top1_state'] == 0:
            # make dir
            key_dir = os.path.join(cmp_dir, key)
            if not os.path.exists(key_dir):
                os.makedirs(key_dir)
            # draw point cloud in matplot
            def draw_pc(pc_file, save_filepath=None, title_info='', pt_size=3, show_fig=False):
                if not show_fig:
                    matplotlib.use('Agg')
                pc_file = os.path.join(configs.data.dataset_folder, pc_file)
                pc = np.fromfile(pc_file, dtype = np.float64)
                pc = np.reshape(pc, (pc.shape[0] // 3, 3))
                x = pc[:, 0]
                y = pc[:, 1]
                z = pc[:, 2]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, s=pt_size, c=z,  # height data for color
                        cmap='rainbow')
                ax.set_title(title_info, fontsize=30)
                ax.axis()
                # set init view
                ax.view_init(elev=65.0, azim=-45.0)
                if save_filepath:
                    fig.savefig(save_filepath, transparent=False, bbox_inches='tight')
                if show_fig:
                    plt.show()
                else:
                    plt.close('all')
            # draw
            draw_pc(res_oxford[key]['query'], os.path.join(key_dir, 'query.svg'))
            draw_pc(res_oxford[key]['top1_file'], os.path.join(key_dir, 'top1_oxford.svg'))
            # draw_pc(res_FT_seq1[key]['top1_file'], os.path.join(key_dir, 'top1_FT_seq1.svg'))
            draw_pc(res_Our_seq1[key]['top1_file'], os.path.join(key_dir, 'top1_Our_seq1.svg'))
            # draw_pc(res_FT_seq2[key]['top1_file'], os.path.join(key_dir, 'top1_FT_seq2.svg'))
            draw_pc(res_Our_seq2[key]['top1_file'], os.path.join(key_dir, 'top1_Our_seq2.svg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, required = True)
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    print(configs)

    cmp_dir = '/home/ericxhzou/Code/LifelongPR/exp/Ours_submodular/Compare'
    # # result: train on Oxford
    # model = model_factory(ckpt = torch.load('/home/ericxhzou/Code/LifelongPR/exp/Ours_submodular/MinkFPN_GeM/comparison-public/2025-05-29-19-39-00_select-greedy-dist256_forget-greedy_prompt-strategy2_sigma1_temp4/env_0.pth'),
    #                       use_prompt=True)
    # res_oxford = get_res_on_oxford(model, cmp_dir, 'res_oxford.pickle')

    # # result: train on Seq 1 (public) - FT
    # model = model_factory(ckpt = torch.load('/home/ericxhzou/Code/LifelongPR/exp/Ours_submodular/MinkFPN_GeM/comparison-public/2025-05-29-21-31-29_select-random0_forget-random_sigma1_temp0/env_3.pth'),
    #                       use_prompt=True)
    # res_FT_seq1 = get_res_on_oxford(model, cmp_dir, 'res_FT_seq1.pickle')
    
    # # result: train on Seq 1 (public) - Our
    # model = model_factory(ckpt = torch.load('/home/ericxhzou/Code/LifelongPR/exp/Ours_submodular/MinkFPN_GeM/comparison-public/2025-05-29-19-39-00_select-greedy-dist256_forget-greedy_prompt-strategy2_sigma1_temp4/env_3.pth'),
    #                       use_prompt=True)
    # res_Our_seq1 = get_res_on_oxford(model, cmp_dir, 'res_Our_seq1.pickle')
    
    # # result: train on Seq 2 (hete) - FT
    # model = model_factory(ckpt = torch.load('/home/ericxhzou/Code/LifelongPR/exp/Ours_submodular/MinkFPN_GeM/comparison-hete/2025-05-28-17-04-04_select-random0_forget-random_sigma1_temp0/env_3.pth'),
    #                       use_prompt=True)
    # res_FT_seq2 = get_res_on_oxford(model, cmp_dir, 'res_FT_seq2.pickle')
    
    # # result: train on Seq 2 (hete) - Our
    # model = model_factory(ckpt = torch.load('/home/ericxhzou/Code/LifelongPR/exp/Ours_submodular/MinkFPN_GeM/ablation-hete/key_module/run1/2025-05-26-00-02-10_select-greedy-dist256_forget-greedy_prompt-strategy2_sigma1_temp4/env_3.pth'),
    #                       use_prompt=True)
    # res_Our_seq2 = get_res_on_oxford(model, cmp_dir, 'res_Our_seq2.pickle')
    
    # compare
    res_oxford = os.path.join(cmp_dir, 'res_oxford.pickle')
    res_FT_seq1 = os.path.join(cmp_dir, 'res_FT_seq1.pickle')
    res_Our_seq1 = os.path.join(cmp_dir, 'res_Our_seq1.pickle')
    res_FT_seq2 = os.path.join(cmp_dir, 'res_FT_seq2.pickle')
    res_Our_seq2 = os.path.join(cmp_dir, 'res_Our_seq2.pickle')
    compare_res_on_oxford(res_oxford, res_FT_seq1, res_Our_seq1, res_FT_seq2, res_Our_seq2, cmp_dir)
    