import sys
import subprocess
import numpy as np
from xyz2mol import xyz2mol
from rdkit import Chem


# --------------------------------------------------------------------------------#
# -----> Converts log/xyz file to RDKIT mol object                          <-----#
# --------------------------------------------------------------------------------#

def byte_2_string(byte):
    """convert byte output to string"""
    return byte.decode('UTF-8').rstrip()


def get_labels(string):
    """get atom type as labelled in Gaussian log file. Returns dictionaries"""
    lst = list(filter(None, string.split(
        'Number     Number       Type             X           Y           Z\n '
        '---------------------------------------------------------------------\n      '
    )[1].split('\n ---')[0].split('\n')))
    lst = [int(list(filter(None, x.split(' ')))[1]) for x in lst]
    labels = {i: lst[i] for i in range(len(lst)) if lst[i] != 1}
    return labels


def get_xyz(f_input):
    """get xyz coordinates from .xyz file. Returns dictionaries"""
    atomic_num = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35,
                  'I': 53}
    lines = [line.strip() for line in open(f_input, 'r')][2:]
    _ = [[x for x in list(filter(None, y.split(' ')))] for y in lines]
    labels = {x: atomic_num[_[x][0]] for x in range(len(_))}
    coordinates = {x: [x for x in _[x][1:]] for x in range(len(_))}
    return labels, coordinates


def get_adjacency(mol):
    """get adjacency matrix. Returns nxn numpy array"""
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    np.fill_diagonal(adj, 1)
    return adj


def get_bond_type(mol):
    """get bond order matrix. Returns nxn numpy array"""
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    bond_order = np.zeros([n_atoms, n_atoms])
    Chem.Kekulize(mol)
    for bond in range(n_bonds):
        i, j = mol.GetBondWithIdx(bond).GetBeginAtomIdx(), mol.GetBondWithIdx(bond).GetEndAtomIdx()
        bond_order[i, j] = bond_order[j, i] = mol.GetBondWithIdx(bond).GetBondTypeAsDouble()
    return bond_order


def get_formal_charges(mol):
    formal_charges = {atom.GetIdx(): atom.GetFormalCharge() for atom in mol.GetAtoms()}
    return formal_charges


def xyz_from_log(filename):
    """get xyz coordinates from Gaussian log file. Returns byte output of obabel, dictionary of atom
    labels and dictionary of atomic coordinates"""

    atomic_num = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35,
                  'I': 53}
    byte_output = subprocess.check_output('obabel ' + str(filename) + ' -oxyz', shell=True)
    # list of lists which each element containing atom as 0th element and x,y,z as 1st:3rd element
    _ = [[x for x in list(filter(None, y.split(' ')))] for y in byte_2_string(byte_output).split('\n')[2:]]
    # dictionary of atom identity
    labels = {x: atomic_num[_[x][0]] for x in range(len(_))}
    # dictionary of atom coordinates
    coordinates = {x: [x for x in _[x][1:]] for x in range(len(_))}
    return byte_output, labels, coordinates


class MolCbh:
    """ class to give you mol object as made by RDKIT. This is to be used
    when constructing CBH fragments
    """

    def __init__(self, f_input):
        """initialize class that gets features describing the structural information of a given molecule"""
        self.filename = f_input
        extension = f_input.split('.')[1]
        if extension == 'log':
            self.labels, self.coordinates = xyz_from_log(f_input)[1:]
        elif extension == 'xyz':
            self.labels, self.coordinates = get_xyz(f_input)
        else:
            print('ERROR. Please give file formatted as either xyz or Gaussian log.')
            exit()
        self.mol = self.get_mol(self.labels, self.coordinates)
        self.adjacency = get_adjacency(self.mol)
        self.bond_order = get_bond_type(self.mol)
        self.formal_charges = get_formal_charges(self.mol)

    def get_mol(self, labels, coordinates):
        """get RDKIT mol object using xyz2mol.py from Jensen group. Returns RDKIT mol object"""
        coordinates = [[float(x) for x in y] for y in coordinates.values()]
        if 15 in list(labels.values()):
            mol = xyz2mol(list(labels.values()), coordinates,
                          charge=0, use_graph=True, allow_charged_fragments=True,
                          embed_chiral=False, use_huckel=True)
        else:
            # one tricky instance, this solved it (will need to investigate more)
            if '5016' in self.filename:
                mol = xyz2mol(list(labels.values()), coordinates,
                              charge=1, use_graph=True, allow_charged_fragments=True,
                              embed_chiral=False, use_huckel=False)
            else:
                mol = xyz2mol(list(labels.values()), coordinates,
                              charge=0, use_graph=True, allow_charged_fragments=True,
                              embed_chiral=False, use_huckel=False)

        return mol


if __name__ == '__main__':
    FILENAME = sys.argv[1]
    OBJ = MolCbh(FILENAME)
    print('MOL object created.')
