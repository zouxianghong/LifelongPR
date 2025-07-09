# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from glob import glob

# For training and test data splits
X_WIDTH = 50
Y_WIDTH = 50

# For Wuhan: Hankou
P1 = [386273.970, 793668.824]
P2 = [386305.362, 793162.555]
P3 = [385831.885, 792749.506]
P4 = [385362.307, 792877.504]
P5 = [384935.007, 792984.298]
P6 = [384807.070, 793441.784]
P7 = [385304.117, 793992.687]
P8 = [385766.586, 793849.294]
P9 = [385257.490, 794769.447]
P10 = [385789.477, 795208.429]
P11 = [386207.435, 795550.294]
P12 = [386676.948, 795994.441]
P13 = [387111.465, 796413.011]
P14 = [387574.958, 796707.013]
P15 = [387241.220, 797424.534]
P16 = [386738.763, 797150.805]
P17 = [386169.484, 796850.781]
P18 = [385512.475, 796579.367]
P19 = [384986.521, 796359.250]
P20 = [384449.766, 796096.536]
P21 = [383729.387, 795831.933]
P22 = [383190.904, 795463.801]
P23 = [386448.284, 795758.231]
P24 = [385218.739, 796463.208]
# For Wuhan: Campus
P25 = [3380083.109596, 533742.949219]
P26 = [3380165.435562, 533853.722664]
P27 = [3378973.098467, 533977.63572]
P28 = [3378865.227791, 533805.883439]
P29 = [3378748.251467, 534079.430823]
P30 = [3379022.905337, 534138.2076]
P31 = [3378827.995864, 534294.254426]
P = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, 
     P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, 
     P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, 
     P31]


FILENAME = "pointcloud_30m_2m_clean.csv"
POINTCLOUD_FOLS = "pointcloud_30m_2m_clean"
ENVS = ['wh_hankou_origin','whu_campus_origin']


def construct_query_and_database_sets(base_path, folders, save_folder, file_extension, p, output_name):
    database_trees = []
    test_trees = []
    for folder in folders:
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path, folder, FILENAME), sep=',')
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if output_name == "business":
                df_test = df_test._append(row, ignore_index=True)
            elif check_in_test_set(row['northing'], row['easting'], p):
                df_test = df_test._append(row, ignore_index=True)
            df_database = df_database._append(row, ignore_index=True)

        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []
    for folder in folders:
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path, folder, FILENAME), sep=',')
        df_locations['timestamp'] = base_path + '/' + folder + '/' + POINTCLOUD_FOLS + \
                                    '/' + df_locations['timestamp'].astype(str) + file_extension
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if output_name == "business":
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            elif check_in_test_set(row['northing'], row['easting'], p):
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting']}
        database_sets.append(database)
        test_sets.append(test)

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if i == j:
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                index = tree.query_radius(coor, r=25)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, save_folder, f'{output_name}_evaluation_database.pickle')
    output_to_file(test_sets, save_folder, f'{output_name}_evaluation_query.pickle')


def check_in_test_set(northing, easting, points):
    in_test_set = False
    for point in points:
        if point[0] - X_WIDTH < northing < point[0] + X_WIDTH and point[1] - Y_WIDTH < easting < point[1] + Y_WIDTH:
            in_test_set = True
            break
    return in_test_set

def output_to_file(output, save_folder, filename):
    file_path = os.path.join(save_folder, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation datasets')
    parser = argparse.ArgumentParser(description='Generate Inhouse Training Dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--eval_thresh', type = int, default = 30, help = 'Threshold for positive examples')
    parser.add_argument('--file_extension', type = str, default = '.npy', help = 'File extension expected')
    parser.add_argument('--save_folder', type = str, required = True, help = 'Folder to save pickle files to')
    args = parser.parse_args()

    # Check dataset root exists, make save dir if doesn't exist
    print('Dataset root: {}'.format(args.dataset_root))
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    for ENV in ENVS:
        folders = os.listdir(os.path.join(base_path, ENV))
        construct_query_and_database_sets(os.path.join(base_path, ENV), folders, args.save_folder, args.file_extension, P, ENV)

