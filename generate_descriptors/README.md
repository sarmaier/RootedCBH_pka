Scripts to generate RootedCBH features. Assumes current directory contains all xyz files for conjugate acid/congjugate base, along with charge files.  To generate csv file of descriptors, along with descriptor labels/identities run "python src/generate_fingerprints.py" in directory containing all acid/base xyz and charge files, along with "train_indices.csv" a dataframe listing indicies of molecules considered in training. Indexing of CBH feature fragment identity not fixed for every run. 


