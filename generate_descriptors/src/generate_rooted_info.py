import re
import glob
import subprocess
import json_numpy
import numpy as np
import copy
import networkx as nx
import networkx.algorithms.isomorphism as iso
from generate_fragments import CbhDescriptors


def make_json(name, data):
    with open(str(name) + ".json", 'w') as json_file:
        json_numpy.dump(data, json_file)


def byte_2_string(byte):
    return byte.decode('UTF-8').rstrip()


def get_sdf(f_):
    atomic_num = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35,
                  'I': 53}
    byte_output = subprocess.check_output('obabel ' + str(f_) + ' -osdf', shell=True)
    return byte_2_string(byte_output)


def sdf_2_graph(string_):
    lines_ = [[x for x in y.split(" ") if x] for y in string_.split("\n") if y]
    lines_ = [[x for x in y] for y in lines_ if y[0].isdigit()]
    g = [tuple([int(x) - 1 for x in y[:2]]) for y in lines_ if len(y) == 7]
    e_att = [y[2] for y in lines_ if len(y) == 7]
    e_att = {i: int(e_att[i]) for i in range(len(e_att))}
    return g, e_att


def net_graph(l_graph: list):
    g = nx.Graph(l_graph)
    return g


def get_labels(string_):
    labels = [[x for x in y.split(" ") if x][3] for y in string_.split("\n")[3:] if len(re.findall("\.", y)) == 3]
    labels = {i: labels[i].lower() for i in range(len(labels))}
    return labels


def get_xyz(string_):
    xyz = [[x for x in y.split(" ") if "." in x] for y in string_.split("\n") if y.count(".") == 3]
    xyz = {i: np.array([float(x) for x in xyz[i]]) for i in range(len(xyz))}
    return xyz


def set_nodes(g, attributes, label):
    nx.set_node_attributes(g, attributes, name=label)
    return g


def compare_graphs(g, h):
    h_coord, heavy_label = "", ""
    h_nodes = [x for x in g.nodes if g.nodes[x]["labels"] == "h"]
    h_nodes = [x for x in h_nodes if g.nodes[[n for n in g.neighbors(x)][0]]["labels"] != "c"]
    for n in h_nodes:
        g_copy = copy.deepcopy(g)
        g_copy.remove_node(n)
        if nx.is_isomorphic(g_copy, h, node_match=iso.categorical_node_match(["labels"], ["labels"]),
                            edge_match=iso.categorical_edge_match(["bo"], ["bo"])):
            g_heavy = list(g.neighbors(n))[0]
            gm = iso.GraphMatcher(g_copy, h, node_match=iso.categorical_node_match(["labels"], ["labels"]))
            gm.is_isomorphic()
            map_ = gm.mapping
            h_heavy = map_[g_heavy]
            break
    return map_, g_heavy, h_heavy


def get_string(filename_):
    f = open(filename_, "r")
    string = f.read()
    f.close()
    return string


def get_molecular_info(_id):
    acid_sdf = get_sdf("id_" + _id + "_ca.xyz")
    base_sdf = get_sdf("id_" + _id + "_cb.xyz")

    acid_cycle, acid_edge = sdf_2_graph(acid_sdf)
    base_cycle, base_edge = sdf_2_graph(base_sdf)

    acid_atypes = get_labels(acid_sdf)
    base_atypes = get_labels(base_sdf)

    acid_graph = net_graph(acid_cycle)
    base_graph = net_graph(base_cycle)

    _acid_att = set_nodes(acid_graph, acid_atypes, "labels")
    _base_att = set_nodes(base_graph, base_atypes, "labels")

    acid_att = set_nodes(_acid_att, acid_edge, "bo")
    base_att = set_nodes(_base_att, base_edge, "bo")

    acid_charge_file = open("id_" + _id + "_ca.charge", "r").read()
    acid_charge = [x for x in acid_charge_file.split(" Charge =")[1].split(" ") if x]

    graph_mapping, acid_heavy, base_heavy = compare_graphs(acid_att, base_att)

    if int(acid_charge[0]) == 0:
        return graph_mapping, acid_heavy, 'ca', base_heavy
    elif int(acid_charge[0]) != 0:
        return graph_mapping, base_heavy, 'cb', base_heavy
#    elif int(acid_charge[0]) == 2 and base_charge == 1:
#        return graph_mapping, base_heavy, 'cb', base_heavy


def distance_function(x):
    return round((1 / int(x + 1)), 4)


if __name__ == "__main__":
    fragment_smiles = {}
    bond_lengths_away = {}
    for filename in glob.glob("*_ca.xyz"):
        mol_id = filename.split("_")[1]
        print("Generating fragments for ID . . . " + mol_id)

        graph_map, heavy_index, acid_or_base, b_heavy_index = get_molecular_info(mol_id)

        cbh_input_file = "id_" + str(mol_id) + "_" + acid_or_base + ".xyz"
        fragments = CbhDescriptors(cbh_input_file, "1")

        center_xyz = {x: np.array(fragments.center_coords[x]).astype(float) for x in fragments.center_coords}

        smiles = fragments.frag_smiles
        n_bond_lengths = {x: distance_function(fragments.topo_distance[heavy_index][x]) for x in center_xyz}
        fragment_smiles[mol_id] = smiles
        bond_lengths_away[mol_id] = n_bond_lengths

    make_json('fragment_smiles', fragment_smiles)
    make_json('bond_lengths_away', bond_lengths_away)
