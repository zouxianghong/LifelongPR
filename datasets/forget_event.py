import numpy as np
import torch

from misc.utils import load_pickle
from eval.eval_utils import get_latent_vectors, euclidean_distance, cosine_dist

from torchpack.utils.config import configs


@torch.no_grad()
def update_forget_events(model, tuples, forget_events:dict):
    # embeddings
    dataset_dict = {}
    for key in tuples:
        tuple_i = tuples[key]
        dataset_dict[key] = {'query': tuple_i.rel_scan_filepath,
                                'northing': tuple_i.position[1],
                                'easting': tuple_i.position[0],
                                'scene_id': tuple_i.scene_id}
    with torch.no_grad():
        embeddingds = get_latent_vectors(model, dataset_dict)
    # distance function
    if configs.eval.similarity == 'cosine':
        dist_func = cosine_dist
    elif configs.eval.similarity == 'euclidean':
        dist_func = euclidean_distance
    else:
        raise ValueError(f'No supported distance function for {configs.eval.similarity}')
    # forget events
    all_inds = set(np.arange(len(embeddingds)))
    for key in tuples:
        anchor_embedding = embeddingds[key]
        positives = tuples[key].positives
        if len(positives) == 0:
            continue
        dists = dist_func(anchor_embedding, embeddingds)
        max_dist_pos = np.max(dists[np.array(positives)])
        negatives = list(all_inds - set(tuples[key].non_negatives))
        min_dist_neg = np.min(dists[np.array(negatives)])
        delta_dist = min_dist_neg - max_dist_pos
        forget = delta_dist < 0.05
        if key not in forget_events:
            forget_events[key] = []
        forget_events[key].append(forget)


def summarize_forget_events(forget_events: dict):
        forget_inds, forget_ratio = [], {}
        if forget_events is None:
            return forget_inds, forget_ratio
        if len(forget_events) == 0:
            return forget_inds, forget_ratio
        # summarize forget events
        for key in forget_events:
            forget_ratio[key] = np.sum(forget_events[key]) / len(forget_events[key])
            if False not in forget_events[key]:  # haven't been learnt
                continue
            ind_first_learnt = forget_events[key].index(False)
            if ind_first_learnt < len(forget_events[key]) - 1:
                else_events = list(np.array(forget_events[key])[ind_first_learnt+1:])
                if True in else_events:
                    forget_inds.append(key)
            
        print(f'Forget Sample Ratio: {len(forget_inds) / len(forget_events) * 100:<.2f}%')
        return forget_inds, forget_ratio


def print_forget_event_ratios(forget_ratio):
    if not forget_ratio:
        return
    forget_scores = np.array([forget_ratio[key] for key in forget_ratio])
    bin, ratios = 0.1, []
    for i in range(1, int(1/bin+1)):
        ratio = forget_scores[forget_scores <= bin*i]
        ratios.append(len(ratio))
    tmp = ratios[:-1]
    ratios = np.array(ratios)
    tmp = np.array([0] + tmp)
    ratios = (ratios - tmp) / len(forget_scores) * 100
    print(ratios)


if __name__ == '__main__':
    event_filepath = '/home/ericxhzou/Code/InCloud/exp/Ours_submodular/MinkFPN_GeM/events_3.pickle'
    events = load_pickle(event_filepath)
    forget_inds, forget_ratio = summarize_forget_events(events)
    print_forget_event_ratios(forget_ratio)