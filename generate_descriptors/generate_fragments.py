import sys
import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
from rdkit import Chem
from mol_obj_cbh import MolCbh


def point_from_string(string):
    """returns 1x3 numpy array representing coordinates in xyz space"""
    # Split the input string by spaces and convert each substring into a floating-point number
    point = np.array([float(x) for x in string.split(' ')])
    # Return the resulting numpy array containing the coordinates
    return point


def rotate_about_axis(theta, axis, vector):
    """For placement of added hydrogens after fragmenting. Returns 1x3 numpy array representing
    vector rotated by theta"""
    # Convert the rotation angle from degrees to radians
    rotation_radians = np.radians(theta)
    # Calculate the rotation vector by multiplying the angle by the axis
    rotation_vector = rotation_radians * axis
    rotation = R.from_rotvec(rotation_vector)
    # Apply the rotation to the vector and obtain the rotated vector
    rotated_vec = rotation.apply(vector)
    return rotated_vec


def orthogonal_v(vector):
    """returns 1x3 numpy array representing vector orthogonal to input vector"""
    # Check if the input vector is parallel to yz-plane
    if vector[1] == 0 and vector[2] == 0:
        # If the input vector lies on the x-axis, it's impossible to find an orthogonal vector
        if vector[0] == 0:
            raise ValueError('zero v')
        else:
            # Calculate the cross product with a reference vector [0, 1, 0]
            cross = np.cross(vector, [0, 1, 0])
            # Normalize the resulting vector to make it a unit vector
            cross = cross / LA.norm(cross)
            # Return the normalized orthogonal vector
            return cross
    # For a general case, calculate the cross product with [1, 0, 0]
    cross = np.cross(vector, [1, 0, 0])
    cross = cross / LA.norm(cross)
    return cross


def unit_vector(p_0, p_1):
    """returns 1x3 numpy array representing unit vector from p_0 to p_1"""
    # Calculate the vector from p_0 to p_1
    vector = (p_1 - p_0)
    vector = (vector / LA.norm(vector))
    return vector


def get_edge_list(bond_order):
    """Returns list of edges as 'atom1 atoms2 bo_12'"""
    edge_list = [
        f"  {i + 1}  {j + 1}  {int(bond_order[i, j])}  0"
        for i in range(bond_order.shape[0])
        for j in range(i, bond_order.shape[1])
        if bond_order[i, j] != 0
    ]
    return edge_list


def get_sub_block(matrix, index_list):
    """Returns nxn sub block of matrix"""
    return matrix[np.ix_(index_list, index_list)]


def build_terminal(n_atoms, labels_, adj, bond_order, xyz, center, fragment, level):
    """builds terminal part of fragment. Returns updated fragment as string"""
    current = [int(i.split(' ')[0]) for i in fragment]
    for atom in fragment:
        label = int(atom.split(' ')[0])
        for neigh in list(labels_):
            if adj[label, neigh] != 0 and neigh not in current:
                if bond_order[label, neigh] == 1.0 and labels_[label] != 15:
                    #                    if bond_order[label, neigh] == 1.0:
                    p_0 = point_from_string(xyz[label])
                    p_1 = point_from_string(xyz[neigh])
                    shift = unit_vector(p_0, p_1)
                    fragment = fragment + [str(neigh) + ' ' + str(level) + ' ' + ' '.join(
                        [str(round(x, 5)) for x in list(p_0 + shift)])]

                elif labels_[label] == 15:
                    fragment.append(f"{neigh} {level - 1} {xyz[neigh]}")

                elif labels_[label] == 7 and labels_[neigh] == 8:
                    n_bonded = {i: (labels_[i], bond_order[label, i]) for i in range(len(bond_order[label, :]))
                                if bond_order[label, i] != 0}
                    doubly_bonded_o = list(n_bonded.values()).count((8, 2.0))
                    singly_bonded_o = list(n_bonded.values()).count((8, 1.0))
                    if doubly_bonded_o == 1 and singly_bonded_o == 1:
                        neigh_1 = [k for k, v in n_bonded.items() if v == (8, 2.0)][0]
                        neigh_2 = [k for k, v in n_bonded.items() if v == (8, 1.0)][0]
                        fragment.append(f"{neigh_1} {level - 1} {xyz[neigh_1]}")
                        fragment.append(f"{neigh_2} {level - 1} {xyz[neigh_2]}")
                elif bond_order[label, neigh] == 2.0:
                    if labels_[neigh] == 8 and labels_[label] != 6 and labels_[label] != 7:
                        fragment.append(f"{neigh} {level - 1} {xyz[neigh]}")
                elif bond_order[label, neigh] == 3.0 and labels_[neigh] == 7:
                    fragment.append(f"{neigh} {level - 1} {xyz[neigh]}")
    return fragment


def build_fragment(n_atoms, adj, xyz, fragment, level):
    """Method to add neighbors to fragment if not terminal. Returns string"""
    # Extracts current atom labels from the fragment list
    current = [int(i.split(' ')[0]) for i in fragment]
    # Loop through atoms in the fragment
    for atom in fragment:
        label = int(atom.split(' ')[0])
        # Check adjacency with all atoms in the range of n_atoms
        for neigh in range(n_atoms):
            # Check adjacency and if the neighbor is not already in the fragment
            if adj[label, neigh] != 0 and neigh not in current:
                fragment.append(f"{neigh} {level} {xyz[neigh]}")
    return fragment


def build_cbh(cbh, labels_, adj, bond_order, xyz, n_atoms, level, steps):
    """builds dictionary with atom centered fragments. Returns dictionary
    get connectivity of each atom in environment of center atom """
    steps = int(steps)
    if level != steps:
        for i in cbh:
            cbh[i] = build_fragment(n_atoms, adj, xyz, cbh[i], level)
    elif level == steps:  # build the terminal part of the fragment, which depends on valency
        for i in cbh:
            cbh[i] = build_terminal(n_atoms, labels_, adj, bond_order, xyz, i, cbh[i], level)
    return cbh


def cbh_atom(labels_, adj, bond_order, xyz, steps):
    """initiates building of atom-centered fragments.
    Returns finalized dictionary of atom-centered cbh fragments"""
    xyz = {x: ' '.join(xyz[x]) for x in xyz}
    n_atoms = len(labels_.values())
    for i in list(labels_):
        for j in list(labels_):
            if adj[i, j] != 0:
                adj[i, j] = labels_[j]
    # Get immediate connectivity of each atom. Then pass to build_cbh() to keep building
    cbh = {}
    for i in list(labels_):
        if labels_[i] != 1:
            cbh[i] = [str(j) + ' 0 ' + xyz[j] for j in list(labels_) if adj[i, j] != 0]
    # Build fragments of CBH steps
    for level in range(1, int(steps) + 1):
        cbh = build_cbh(cbh, labels_, adj, bond_order, xyz, n_atoms, level, steps)

    return cbh


def make_mol_block(formal_, labels, fragment, fragment_bo, center1=None, center2=None):
    """Returns mol block formatted to be read by RDKIT"""
    elements = {1: 'H', 3: 'Li', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si',
                15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
    charges = {int(x.split(" ")[0]): formal_[int(x.split(" ")[0])] for x in fragment
               if labels[int(x.split(" ")[0])] != 1}
    n_ele = len(list(charges))
    n_charges = np.count_nonzero(np.array(list(charges.values())))
    charges = np.array(list(charges.values()))
    charges = {i: charges[i] for i in range(n_ele) if charges[i] != 0}
    charge_string = ''
    if n_charges == 0:
        charge_string = None
    else:
        n_charges = len(list(charges.values()))
        charge_string = 'M  CHG  ' + str(n_charges) + '  '
        for i in charges:
            charge_string = charge_string + str(i + 1) + '   ' + str(charges[i]) + '   '
    fragment = sorted([x for x in fragment if 'addedH' not in x], key=lambda x: int(x.split()[0]))
    _ = [[format(float(x), '.4f') for x in y.split(' ')[2:]] for y in fragment
         if labels[int(y.split(' ')[0])] != 1]
    _ = ['{:>8} {:>8} {:>8}'.format(*x) for x in _]
    atoms = [labels[int(x.split(' ')[0])] for x in fragment if labels[int(x.split(' ')[0])] != 1]
    atoms = [elements[x] for x in atoms]
    edge_list = '\n'.join(get_edge_list(fragment_bo))
    xyz = '\n'.join('    ' + _[x] + ' ' + atoms[x] + '   0  0  0  0  0  0  0  0  0  0  0  0' \
                    for x in range(len(atoms)))
    n_atoms, n_edges = len(atoms), len(edge_list.split('\n'))
    mol_block = 'Fragment on ' + str(center1) + '-' + str(center2) + '\n     RDKit          2D\n\n' \
                                                                     '  ' + str(n_atoms) + '  ' + str(
        n_edges) + '  0  0  0  0  0  0  0  0999 V2000\n' + \
                xyz + '\n' + edge_list + \
                '\n'
    if charge_string:
        mol_block = mol_block + charge_string + '\nM  END\n$$$$'
    else:
        mol_block = mol_block + 'M  END\n$$$$'
    return mol_block


class CbhError(Exception):
    """Custom exception for specific situations."""

    def __init__(self, message="CBH class could not proces input."):
        self.message = message
        super().__init__(self.message)


class CbhDescriptors:
    """
    generates CBH fragments
    """

    def __init__(self, file_input, steps):
        """initialize class that gets CBH features"""
        self.f_input = file_input
        self.steps = steps
        mol_graph = MolCbh(file_input)
        labels = mol_graph.labels
        adj = mol_graph.adjacency
        self.coordinates = mol_graph.coordinates
        self.topo_distance = Chem.rdmolops.GetDistanceMatrix(mol_graph.mol)

        bond_order = mol_graph.bond_order
        formal_charges = mol_graph.formal_charges
        heavy_neighbors = {atom.GetIdx(): len(atom.GetBonds()) - int(atom.GetTotalNumHs(includeNeighbors=True))
                           for atom in mol_graph.mol.GetAtoms()}

        non_terminal = {x: labels[x] for x in heavy_neighbors if heavy_neighbors[x] > 1}
        cbh_w_index = cbh_atom(labels, adj, bond_order, self.coordinates, steps)

        self.cbh_w_index = {x: cbh_w_index[x] for x in cbh_w_index if x in list(non_terminal)}
        self.cbh_clean, self.cbh_frag_conn, self.cbh_frag_bo, self.frag_smiles, self.center_coords = \
            self.clean_cbh_frags(labels, bond_order, self.cbh_w_index, formal_charges)

    def index_2_labels(self, labels, steps):
        """ Change identity of atom in fragments from indexes to labels. This is needed to print to
        input files for electronic structure calculations"""
        cbh_w_label = {}
        for i in self.cbh_w_index:
            lst = []
            for j in self.cbh_w_index[i]:
                neigh, level, xyz_i = j.split(' ')[0], j.split(' ')[1], ' '.join(j.split(' ')[2:])
                if int(level) != int(steps):
                    lst.append(str(labels[int(neigh)]) + ' ' + xyz_i)
                else:
                    lst.append('1 ' + xyz_i)
            cbh_w_label[i] = lst
        return cbh_w_label

    def clean_cbh_frags(self, labels, bond_order, cbh, formal_):
        """cleans CBH fragments. This changes coordinates of fragment, so may be an issue
        beyond CBH2, but it is useful for convergence of CBH2 fragments
        """
        cbh_frag_connectivity, cbh_frag_bond_order = {}, {}
        mol_xyz = {}
        smi_f = open("log_cons_smiles.txt", "a")
        smi_f.write(self.f_input + "\n")
        smi_f.close()
        smile_dict = {}
        coord_dict = {}
        for i in cbh:
            coord_dict[i] = self.coordinates[i]
            _ = [x for x in cbh[i] if 'addedH' not in x and x.split(' ')[1] != self.steps]
            heavy_index = sorted([int(x.split(' ')[0]) for x in _ if labels[int(x.split(' ')[0])] != 1])
            cbh_frag_connectivity[i] = get_sub_block(bond_order, heavy_index)
            cbh_frag_bond_order[i] = get_sub_block(bond_order, heavy_index)
            mol_block = make_mol_block(formal_, labels, _, cbh_frag_bond_order[i], i)
            mol_graph = Chem.rdmolfiles.MolFromMolBlock(mol_block)
            mol_graph = Chem.rdmolops.AddHs(mol_graph, addCoords=True)
            smi = Chem.MolToSmiles(mol_graph)
            smile_dict[i] = str(smi)
            Chem.AllChem.EmbedMolecule(mol_graph, randomSeed=0xf00d)
            mol_xyz[i] = Chem.rdmolfiles.MolToXYZBlock(mol_graph)

        return mol_xyz, cbh_frag_connectivity, cbh_frag_bond_order, smile_dict, coord_dict


if __name__ == '__main__':
    _f_input = sys.argv[1]
    _steps = sys.argv[2]
    try:
        mol_ = CbhDescriptors(_f_input, _steps)
        print("Processing file . . . " + _f_input)
    except CbhError as e:
        print("CBH descriptor error caught:", e.message)
        exit()
