import os
import sys
import json_numpy
import numpy as np
import pandas as pd
import subprocess


# Function to save data to a JSON file
def save_json(name, info):
    with open(f"{name}.json", 'w') as json_out:
        json_numpy.dump(info, json_out)


# Function to load data from a JSON file
def load_json(name):
    with open(name, 'r') as json_in:
        return json_numpy.load(json_in)


def get_unique_smiles(_smiles):
    unique = list(set([x for y in _smiles for x in y]))
    count_mol = [0 for x in unique]
    n_unique = len(unique)
    for x in range(n_unique):
        for y in _smiles:
            if unique[x] in y:
                count_mol[x] = count_mol[x] + 1
    unique = list(set([unique[x] for x in range(len(unique)) if count_mol[x] > 9]))
    return unique


class GenInfoError(Exception):
    """Custom exception for error handing generate_rooted_info.py"""
    def __init__(self, message="generate_rooted_info.py could not proces input."):
        self.message = message
        super().__init__(self.message)


if __name__ == "__main__":
    py_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    try:
        cmd = ["python", py_dir + "/generate_rooted_info.py"]
        subprocess.Popen(cmd).wait()
    except GenInfoError as e:
        print("Error caught with generate_rooted_info.py processing:", e.message)
        exit()
    # load fragment smiles and connectivity information
    try:
        smiles = load_json("fragment_smiles.json")
        conn_dist = load_json("bond_lengths_away.json")
    except FileNotFoundError:
        print("Sorry, fragment or bond info missing. Failed to generate fingerprints.")
        exit()
    # consider only those molecules in the train + val (not test) set
    try:
        train_index = list(pd.read_csv("../train_indices.csv")['file'])
    except FileNotFoundError:
        print("Sorry, csv of train indices not found.")
        exit()
    all_smiles = [list(smiles[x].values()) for x in smiles if int(x) in train_index]

    # get list of unique smiles from dataset considered
    unique_smiles = get_unique_smiles(all_smiles)
    print("Getting unique fragments . . .")
    all_features = {}
    for value in smiles:
        count = {x: list(smiles[value].values()).count(x) for x in unique_smiles}
        features = []
        for i in unique_smiles:
            tops = [conn_dist[value][x] for x in conn_dist[value] if smiles[value][x] == i]
            if len(tops) > 0:
                max_tops = max(tops)
                features.extend([max_tops])
            else:
                features.extend([0])
        features = np.array(features)
        all_features[int(value)] = features

    feature_value = pd.DataFrame(unique_smiles)
    df_features = pd.DataFrame(all_features).T.sort_index()
    df_features = df_features.reset_index()

    # create csv of CBH features and feature identities (ordering will change)
    feature_value.to_csv("cbh_feature_identity.csv")
    df_features[:2389].to_csv("cbh_features_" + str(len(df_features)) + ".csv")
    print("Done! CBH features found in cbh_features_" + str(len(df_features)) + ".csv")
    print("CBH feature identity/indexing found in cbb_feature_identity")
