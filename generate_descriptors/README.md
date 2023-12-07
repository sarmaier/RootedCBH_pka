# RootedCBH Fingerprints

Scripts to generate RootedCBH features. Generates two csv files . . . 1) descriptors and 2) descriptor labels/identities. Assumes current directory contains all xyz and charge files for conjugate acid/congjugate base, along with "train_indices.csv" a dataframe listing indices of molecules considered in training. Indexing of CBH feature fragment identity not fixed for every run.


<img width="760" alt="image" src="https://github.com/sarmaier/RootedCBH_pka/assets/152440946/12b2d763-89f4-49a7-b009-7c56addb0697"> 

Example of RootedCBH fingerprint, where nitrogen is the site of protonation. Fragments in gray are either not present in the molecule or do not correspond to the fragment with the minimum path length.

## Usage
python src/generate_fingerprints.py

## Requirements
pandas~=1.0.1

numpy~=1.18.1

networkx~=2.5

rdkit~=2020.03.3.0

scipy~=1.5.4


