import os 
import shutil
import csv 
import argparse 
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import multiprocessing as mp


def get_pointcloud_tensor(xyz):
    # normalize
    xyz[:,0] = xyz[:,0] - xyz[:,0].mean()
    xyz[:,1] = xyz[:,1] - xyz[:,1].mean()
    xyz[:,2] = xyz[:,2] - xyz[:,2].mean()
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    return xyz


def process_pointcloud(ARGS):
    pc_path, source_dir, save_dir = ARGS
    if not os.path.isfile(pc_path):
        return None
    xyz = np.load(pc_path).reshape(-1,3)
    if len(xyz) == 0:
        return None  
    xyz = get_pointcloud_tensor(xyz)
    save_path = pc_path.replace(source_dir, save_dir).split('.')[0]
    np.save(save_path, xyz)


def multiprocessing_preprocessing(run, source_dir, save_dir):
    # Prepare inputs 
    clouds_raw = sorted(glob(os.path.join(run, '*')))
    ARGS = [[c, source_dir, save_dir] for c in clouds_raw]

    # Multiprocessing the pre-processing 
    with mp.Pool(32) as p:
        _ = list(tqdm(p.imap(process_pointcloud, ARGS), total = len(ARGS)))


def global_csv_to_northing_easting(csv_path, source_dir, save_dir):
    df = pd.read_csv(csv_path, sep=',')
    df = df.sort_values('timestamp')  # sort by time stamp
    df.reset_index(drop=True)
    save_path = os.path.join(csv_path).replace(source_dir, save_dir)
    df.to_csv(save_path, index=0)


def process_Wuhan(root, save_dir):
    environments = ['wh_hankou_origin','whu_campus_origin']
    for env in environments:
        for run in os.listdir(os.path.join(root, env)):
            if not os.path.isdir(os.path.join(root, env, run, 'pointcloud_30m_2m_clean')):
                continue
            print(os.path.join(save_dir, env, run, 'pointcloud_30m_2m_clean'))
            if not os.path.exists(os.path.join(save_dir, env, run, 'pointcloud_30m_2m_clean')):
                os.makedirs(os.path.join(save_dir, env, run, 'pointcloud_30m_2m_clean'))
            global_csv_to_northing_easting(os.path.join(root, env, run, 'pointcloud_30m_2m_clean.csv'), root, save_dir)
            multiprocessing_preprocessing(os.path.join(root, env, run, 'pointcloud_30m_2m_clean'), root, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, required = True, help = 'Root for Wuhan Dataset')
    parser.add_argument('--save_dir', type = str, required = True, help = 'Directory to save pre-processed data to')
    args = parser.parse_args()

    process_Wuhan(args.root, args.save_dir)