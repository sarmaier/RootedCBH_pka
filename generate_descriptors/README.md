# RootedCBH Fingerprints

Scripts to generate RootedCBH features. Assumes current directory contains all xyz files for conjugate acid/congjugate base, along with charge files. To generate csv file of descriptors, along with descriptor labels/identities run "python src/generate_fingerprints.py" in directory containing all acid/base xyz and charge files, along with "train_indices.csv" a dataframe listing indicies of molecules considered in training. Indexing of CBH feature fragment identity not fixed for every run.

## Usage
python src/generate_fingerprints.py

## Requirements
pandas~=1.0.1\n
numpy~=1.18.1

networkx~=2.5

rdkit~=2020.03.3.0

scipy~=1.5.4

