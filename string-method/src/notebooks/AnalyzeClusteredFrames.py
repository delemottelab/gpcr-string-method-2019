# coding: utf-8

# # Load trajectories and cluster indices etc

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import itertools

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

sys.path.append('MD_common/')
import MD_fun
import collections
from colvars import *

# fun = MD_fun.MD_functions()
logger = logging.getLogger("ancf")


# ## Load simulation and clustering metadata

# In[2]:


def to_absolute_indices(cluster_indices, center_indices):
    """Convert to absolute frame indicies"""
    frames = np.empty(len(center_indices), dtype=int)
    index_count = np.zeros(len(center_indices), dtype=int)
    for traj_idx, cluster in enumerate(cluster_indices):
        cluster_idx = cluster - 1
        count = index_count[cluster_idx]
        if count == center_indices[cluster_idx]:
            frames[cluster_idx] = traj_idx
        index_count[cluster_idx] += 1
    return frames


def load_cluster_representations(simulation):
    return [simulation.traj[i] for i in simulation.cluster_rep_indices]
    # rep_files = sorted([
    #    f
    #    for f in glob.glob(
    #        simulation.clusterpath + 'clustered_frames/reps_cluster_*.dcd')
    # ])
    # return [md.load(f, top=simulation.topology_path) for f in rep_files]


def load_cluster_indices(simulation):
    # Cluster files:
    with open(simulation.clusterpath +
                      "cluster_indices_.txt") as cluster_indices_file:
        simulation.cluster_indices = [int(l[0]) for l in cluster_indices_file]
    with open(simulation.clusterpath +
                      "center_indices.txt") as center_indices_file:
        # Load the center indices. The index represents where the frame occurs in the sequence of frames in that cluster
        simulation.center_indices = [int(l[0]) for l in center_indices_file]
    simulation.cluster_rep_indices = to_absolute_indices(
        simulation.cluster_indices, simulation.center_indices)
    logger.debug(
        "Found center indices at %s which corresponds to traj indices %s",
        simulation.center_indices, simulation.cluster_rep_indices)
    simulation.cluster_representations = load_cluster_representations(simulation)
    return simulation


def load_default(simulation):
    logger.info("Using simulation files in directory " + simulation.path)

    simulation.topology_path = simulation.path + simulation.name + ".pdb"
    simulation.traj = md.load(
        simulation.path + simulation.name + ".dcd",
        top=simulation.topology_path,
        stride=simulation.stride)
    simulation.timestep = simulation.stride * 0.18 / 1000
    return load_cluster_indices(simulation)


# # Get an overview of states

# In[3]:


def plot_cluster_states(simulation):
    plt.figure(figsize=(16, 3))
    plt.plot(times, simulation.cluster_indices)
    plt.ylabel(r'Clustered state')
    plt.xlabel("Time $\mu$s")
    plt.yticks(range(0, max(simulation.cluster_indices)))
    plt.show()
    plot_state_changes(times, simulation.cluster_indices)


# # Identify significant changes between states

# ## Compute distances
# 

# In[4]:


def getAllCalphaDistances(traj, atoms):
    res = np.empty((len(atoms), len(atoms)))
    for i, a in enumerate(atoms):
        res[i, i] = 0
        for j in range(i + 1, len(atoms)):
            dist = md.compute_distances(
                traj, [(a.index, atoms[j].index)], periodic=False)[0]
            res[i, j] = dist
            res[j, i] = dist
    return res


def compute_distance_difference(cluster_representations, traj, atoms):
    if len(cluster_representations) == 0:
        return
    number_reps = len(cluster_representations)
    diffs = np.empty((number_reps, number_reps), dtype=object)
    # distance_matrices = [fun.getAllCalphaDistances(t, query=query)[0] for t in cluster_representations]
    fun = MD_fun.MD_functions()
    distance_matrices = [
        # fun.getAllCalphaDistances(t)[0] for t in cluster_representations
        getAllCalphaDistances(t, atoms) for t in cluster_representations
    ]
    # print(distance_matrices)
    zero_array = np.zeros(distance_matrices[0].shape)
    for i, dist1 in enumerate(distance_matrices):
        diffs[i, i] = zero_array
        for j in range(i + 1, number_reps):
            diff = dist1 - distance_matrices[j]
            diffs[i, j] = diff
            diffs[j, i] = diff
            # TODO square form!
    return diffs, distance_matrices


def compute_transition_matrix(cluster_indices):
    counter = collections.Counter(cluster_indices)
    matrix = np.zeros((len(counter), len(counter)))
    for i, cluster in enumerate(cluster_indices):
        if i == 0:
            continue
        last_cluster = cluster_indices[i - 1]
        # remember to offset indices by -1
        matrix[last_cluster - 1, cluster - 1] += 1
        # TODO double check formula for transition matrix
    # normalize and return
    transition_count = len(cluster_indices) - 1
    return matrix / transition_count


def save_clusterrep_pdb(simulation):
    pdb_name = '(%s_stride_%s)' % (simulation.id, simulation.stride)
    for idx, rep in enumerate(simulation.cluster_representations):
        filename = '%s/clustered_frames/%s_stride_%s_reps_cluster_%s.pdb' % (
            simulation.clusterpath, simulation.id, simulation.stride, idx + 1)
        rep.save(filename)

    # In[5]:


class AtomPair(object):
    def getDist(self):
        return self.dist

    def getScaledDist(self):
        return self.scaled_dist

    def getNormDist(self):
        return self.norm_dist

    def __init__(self, dist, atom1, atom2, params={}):
        self.__dict__.update(params)
        self.__dict__.update({"dist": dist, "atom1": atom1, "atom2": atom2})

    def __str__(self):
        return str(self.__dict__)


class Node(object):
    def __init__(self, params={}):
        self.__dict__.update(params)

    def __str__(self):
        return str(self.__dict__)

    def getColor(self):
        return self.color


class Graph(object):
    def __init__(self, params={}):
        self.__dict__.update(params)

    def __str__(self):
        return str(self.__dict__)

    def vmd_bonds_script(self):
        bonds_script = ""
        for n1 in self.nodes:
            for n2 in n1.neighbours:
                if n2.color > n1.color:
                    bonds_script += vmd_bond(n1.atom, n2.atom)
        return "for {set x 0} {$x <= 5} {incr x} {%s}" % (bonds_script)

    def plot_distances(graph,
                       simulation,
                       max_per_plot=10,
                       histogram=False,
                       bincount=10,
                       separate_clusters=False):
        traj = simulation.traj
        pairs = []
        labels = []

        def _plot_once(dist, label):
            if histogram:
                plt.hist(dist[~np.isnan(dist)], bincount, alpha=(0.3 if separate_clusters else 0.5), label=label)
            else:
                plt.plot(dist, '--', alpha=0.5, label=label)

        def plot():
            dists = md.compute_distances(traj, np.array(pairs), periodic=False)
            for i in range(0, dists.shape[1]):
                dist = dists[:, i]
                label = labels[i]
                if separate_clusters:
                    for cl in range(0, len(simulation.cluster_rep_indices)):
                        cluster = cl + 1
                        values = np.empty(len(dist))
                        for idx in range(0, len(dist)):
                            values[idx] = dist[idx] if simulation.cluster_indices[
                                                           idx] == cluster else np.nan
                        _plot_once(values, label + "(cluster " + str(cl + 1) + ")")
                else:
                    _plot_once(dist, label)
            if histogram:
                plt.xlabel("Dists (nm)")
                plt.ylabel("Count")
            else:
                plt.xlabel("Frame#")
                plt.ylabel("Dist (nm)")
            plt.legend()
            plt.show()

        for n1 in graph.nodes:
            for n2 in n1.neighbours:
                if n2.color > n1.color:
                    pairs.append([n1.atom.index, n2.atom.index])
                    labels.append("%s-%s" % (n1.atom, n2.atom))
                    if len(pairs) % max_per_plot == 0:
                        plot()
                        pairs = []
                        labels = []
        if len(pairs) > 0:
            plot()

    def number_bonds(self):
        count = 0
        for n1 in self.nodes:
            for n2 in n1.neighbours:
                if n2.color > n1.color:
                    count += 1
        return count

    def find_subgraphs(graph):
        def traverse(nodes, visited):
            for n1 in nodes:
                key1 = str(n1.atom)
                if visited.get(key1) is not None:
                    continue
                visited[key1] = n1
                traverse(n1.neighbours, visited)

        all_visited = {}
        subgraph_idx = 0
        subgraphs = []
        for n in graph.nodes:
            key = str(n.atom)
            if all_visited.get(key) is not None:
                continue
            all_visited[key] = n
            visited = {key: n}
            traverse(n.neighbours, visited)
            for k, node in visited.items():
                node.subgraph = subgraph_idx
                all_visited[k] = node
            subgraph_idx += 1
            subgraphs.append([visited[k] for k in sorted(visited.keys())])
            # logger.debug("Visited nodes for subgraph: %s", visited)
        graph.subgraphs = subgraphs

    def color(graph, split_subgraphs=False):
        """
        Simple algorithm to put colors on nodes in a graph so that no neighbours have the same color, 
        i.e. creating a multipartite graph. 
        # based on https://gist.github.com/sramana/583681

        if split_subgraphs is True, then the graph will be split into disconnected subgraphs. 
        All subgraphs will then be partitioned with different colors
        """
        blocked_colors = {}

        def promising(node, color):
            for neighbor in node.neighbours:
                if neighbor.color == color:
                    return False
            return True

        def get_color_for_node(node):
            for color in graph.colors:
                if blocked_colors.get(color, False):
                    continue
                if promising(node, color):
                    return color
            return None

        if split_subgraphs:
            graph.find_subgraphs()
            all_nodes = graph.subgraphs
        else:
            all_nodes = [graph.nodes]
        for nodes in all_nodes:
            used_colors = []
            for n in nodes:
                color = get_color_for_node(n)
                if color is None:
                    return False
                n.color = color
                used_colors.append(color)
            if split_subgraphs:
                for color in used_colors:
                    blocked_colors[color] = True
        return True

    def explain_to_human(graph):
        color_to_nodes = {}
        text = "\nThere were %s atoms, %s bonds and %s colors in the graph (%s subgraphs)\n" % (
            len(graph.nodes), graph.number_bonds(), len(graph.colors),
            len(graph.subgraphs) if hasattr(graph, 'subgraphs') else 1)

        def to_colors(node):
            return sorted(set([n.color for n in node.neighbours]))

        for node in graph.nodes:
            nodes = color_to_nodes.get(node.color, [])
            nodes.append(node)
            color_to_nodes[node.color] = nodes
        for color, nodes in color_to_nodes.items():
            #         logger.info(
            #             "VMD query for color %s:\n%s",
            #             color,
            #             to_vmd_query([n.atom for n in nodes],atom_name="CA"))
            text += "\n - There were " + str(len(nodes)) + " with color " + str(color) + ". Out of these "
            color_to_connections = {}
            for node in nodes:
                connected_colors = to_colors(node)
                key = str(connected_colors)
                value = color_to_connections.get(key, (connected_colors, []))
                value[1].append(node)
                color_to_connections[key] = value
            first = False
            for colors, nodes in color_to_connections.values():
                text += (" and" if first else "") + str(
                    len(nodes)) + " atoms were displaced from atoms with color " + str(colors)
                first = False
        logger.info("Explanation for graph: %s:", text)
        logger.info("VMD query for atoms in graph:\n%s",
                    graph.vmd_atom_selection())
        logger.info("VMD atom coloring script:\n%s",
                    graph.vmd_atom_colors_script())
        logger.info("VMD query for bonds in graph:\n%s",
                    graph.vmd_bonds_script())

    def vmd_atom_selection(self, color=None):
        return to_vmd_query(
            [n.atom for n in self.nodes if color is None or n.color == color],
            atom_name="CA")

    def vmd_atom_colors_script(self):
        text = ""
        for color in self.colors:
            text += "set color%s [atomselect top \"%s\"];" % (
                color, self.vmd_atom_selection(color=color))
            text += "$color%s set beta %s;" % (color, color)
        return text


def partition_as_graph(atompairs,
                       dist_func=AtomPair.getDist,
                       cutoff=1.0,
                       max_partition_count=10,
                       split_subgraphs=False):
    # extract atoms and build a graph
    # TODO a log(n2) midpoint method to find the optimal color count is better
    for color_count in range(1, max_partition_count + 1):
        graph = Graph({"nodes": [], "colors": range(0, color_count)})
        name_to_nodes = {}

        def get_node(a):
            name = str(a)
            node = name_to_nodes.get(name)
            if node is None:
                node = Node({
                    "atom": a,
                    "neighbours": [],
                    "dists": [],
                    "color": None
                })
                name_to_nodes[name] = node
            return node

        for ap in atompairs:
            dist = dist_func(ap)
            if dist < cutoff:
                continue
            node1 = get_node(ap.atom1)
            node2 = get_node(ap.atom2)
            node1.neighbours.append(node2)
            node1.dists.append(dist)
            node2.neighbours.append(node1)
            node2.dists.append(dist)
        graph.nodes = [name_to_nodes[n] for n in sorted(name_to_nodes.keys())]
        if graph.color(split_subgraphs=split_subgraphs):
            # we managed to color the graph. stop here
            return graph
    raise Exception("Could not split/color graph into " +
                    str(max_partition_count) + " partitions")


def get_reference_dist(atom1, atom2, ref_traj):
    ref_atom1 = find_atom(
        atom1.element.symbol,
        str(atom1.residue),
        atom1.name,
        ref_traj,
        query=protein_CA_q)
    ref_atom2 = find_atom(
        atom2.element.symbol,
        str(atom2.residue),
        atom2.name,
        ref_traj,
        query=protein_CA_q)
    if ref_atom1 is None or ref_atom2 is None:
        return None
    return abs(
        md.compute_distances(
            ref_traj, [(ref_atom1.index, ref_atom2.index)], periodic=False)[0])


def map_value_indices(matrix,
                      atoms,
                      reference_dists,
                      symmetric=False,
                      sort=True,
                      sigma=1):
    """Find the max elements of the (2D) matrix and return a tuple with the value and the indices, sorted according to their value"""
    dim = matrix.shape
    res = []
    for i in range(0, dim[0]):
        for j in range(i + 1 if symmetric else 0, dim[1]):
            dist = abs((matrix[i, j]))
            pair = AtomPair(dist, atoms[i], atoms[j])
            ref_dists = reference_dists[i, j] ** sigma  # NOTE EXPONENT
            pair.scaled_dist = np.nan if ref_dists == 0 else dist / ref_dists
            res.append(pair)
    # print(res)
    # return np.array(res, dtype=object)
    return res


def to_atom_pairs(simulation,
                  atoms,
                  distance_differences,
                  distance_matrices,
                  transition_matrix,
                  transition_cutoff=0.00001):
    # print(transition_matrix)
    # print(atoms)
    # Create reference distances which equal the average distance between atoms in all frames
    # TODO ratio off the displacement to the distance in the reference structure?
    reference_dists = np.zeros(distance_matrices[0].shape)
    for i in range(0, len(distance_matrices)):
        reference_dists += distance_matrices[i]
    reference_dists = reference_dists / len(distance_matrices)
    number_to_print = 0
    all_pairs = None  # sum distance differences
    for i in range(0, distance_differences.shape[0]):
        for j in range(i + 1, distance_differences.shape[1]):
            # Skip transition which did not occur
            # TODO should we actually filter away transitions which did not occur? They still might in other circumstances
            if transition_matrix[i,
                                 j] < transition_cutoff and transition_matrix[j,
                                                                              i] < transition_cutoff:
                logger.info(
                    "No direct path between clusters %s and %s. Skipping analysis",
                    i + 1, j + 1)
                continue
            value_indices = map_value_indices(
                distance_differences[i, j],
                atoms,
                reference_dists,
                symmetric=True)
            max_diff = max(value_indices, key=AtomPair.getDist).dist
            for rank, pair in enumerate(
                    sorted(value_indices, key=AtomPair.getDist, reverse=True)):
                if rank < number_to_print:
                    logger.debug("max distance between clusters %s and %s: %s",
                                 i + 1, j + 1, pair)
                pair.rank = rank + 1  # avoid zero rank which leads to division by zero
                pair.norm_dist = pair.dist / max_diff
            if all_pairs is None:
                all_pairs = value_indices
            else:
                for idx, total_pair in enumerate(all_pairs):
                    pair = value_indices[idx]
                    total_pair.dist += pair.dist
                    total_pair.rank += pair.rank
                    total_pair.norm_dist += pair.norm_dist
        return all_pairs


def analyze_pair_distances(simulation,
                           atoms,
                           distance_differences,
                           distance_matrices,
                           transition_matrix,
                           percentile=99.9,
                           dist_func=AtomPair.getNormDist,
                           plot_dist_distribution=True,
                           split_subgraphs=False):
    all_pairs = to_atom_pairs(simulation, atoms, distance_differences,
                              distance_matrices, transition_matrix)
    number_to_print = 0
    # low rank is better
    logger.info("#####Top %s totally most displaced atoms between clusters:",
                number_to_print)
    for pair in sorted(
            all_pairs, key=AtomPair.getDist, reverse=True)[0:number_to_print]:
        logger.debug("max distance: %s", pair)
    logger.info(
        "#####Top %s totally most displaced (scaled) atoms between clusters:",
        number_to_print)
    scaled_sorted = sorted(all_pairs, key=AtomPair.getScaledDist, reverse=True)
    for pair in scaled_sorted[0:number_to_print]:
        logger.debug("max distance: %s", pair)

    #     logger.info(
    #         "#####Top %s ranked displaced atoms between clusters:", number_to_print)
    #     for pair in sorted(all_pairs, key=lambda ap: ap.rank)[0:number_to_print]:
    #         logger.debug(
    #             "max rank: %s", pair)

    logger.info("#####Top %s norm displaced atoms between clusters:",
                number_to_print)
    # rank_scale = math.sqrt(len(simulation.cluster_representations))
    norm_sorted = sorted(all_pairs, key=AtomPair.getNormDist, reverse=True)
    for pair in norm_sorted[0:number_to_print]:
        logger.debug("max distance: %s", pair)

    # Create a multipartite graph
    final_distances = np.array([dist_func(p) for p in all_pairs])
    cutoff = np.percentile(final_distances, percentile)
    logger.info("#####Computing scaled differences with cutoff %s", cutoff)
    graph = partition_as_graph(
        all_pairs,
        dist_func=dist_func,
        cutoff=cutoff,
        split_subgraphs=split_subgraphs)
    if plot_dist_distribution:
        plt.hist(final_distances, 50, label="Count")
        plt.xlabel("Distance")
        plt.plot(
            cutoff,
            30,
            '*',
            label="Cutoff for %s percentile at %s" % (percentile, cutoff))
        plt.legend()
        plt.show()
    # for node in color_to_nodes.get(color):
    # print(node.atom)
    save_csv("rank_cluster_indices", [(p.rank, p.dist,
                                       p.scaled_dist, p.atom1, p.atom2)
                                      for p in all_pairs], simulation)
    return all_pairs, graph


# # Try and visualize cluster and CVs
# ## Create CV object

# In[6]:


def create_cluster_rep_rmsd_cvs(cluster_representations):
    q_CA = "protein and chainid 0 and name CA"
    return [
        RmsdCV("rmsd_cluster_rep_%s" % i, rep, q_CA)
        for i, rep in enumerate(cluster_representations)
    ]


def compute_color_mean_distance(simu_traj, graph, color1, color2):
    dists = None
    for n1 in graph.nodes:
        if n1.color == color1:
            for n2 in graph.nodes:
                if n2.color == color2 and n2 in n1.neighbours:
                    d = compute_distance_CA_atoms(n1.atom.residue,
                                                  n2.atom.residue, simu_traj)
                    if dists is None:
                        dists = d
                    else:
                        dists = np.concatenate((dists, d), axis=1)
    return np.mean(dists, axis=1)


def most_relevant_dist_generator(graph, pairs):
    """
    Returns a generator which just picks to greatest distance between color and uses that a representative
    """
    color_most_relevant = {}
    for n1 in graph.nodes:
        for idx, n2 in enumerate(n1.neighbours):
            if n2.color <= n1.color:
                continue
            color_id = str(n1.color) + "-" + str(n2.color)
            d = n1.dists[idx]
            current = color_most_relevant.get(color_id)
            if current is None or current[0] < d:
                color_most_relevant[color_id] = (d, n1.atom, n2.atom)
    vmd_query = ""
    for (d, atom1, atom2) in color_most_relevant.values():
        vmd_query += vmd_bond(atom1, atom2)
    logger.debug("Most relevant atoms for color pairs: %s. \nBond query:\n%s",
                 color_most_relevant, vmd_query)

    def cv_generator(simu_traj, graph, color1, color2):
        color_id = "%s-%s" % (color1, color2)
        dist, atom1, atom2 = color_most_relevant.get(color_id)
        return compute_distance_CA_atoms(atom1.residue, atom2.residue,
                                         simu_traj)

    return cv_generator


def colors_connected(graph, color1, color2):
    for n1 in graph.nodes:
        if n1.color == color1:
            for n2 in n1.neighbours:
                if n2.color == color2:
                    return True
    return False


def create_cvs(graph, CV_generator=compute_color_mean_distance):
    color_combos = [c for c in itertools.combinations(graph.colors, 2)]
    cvs = []
    for color1, color2 in color_combos:
        if colors_connected(graph, color1, color2):
            cv = ColorCv(graph, color1, color2, CV_generator)
            cvs.append(cv)
    return cvs


# ## Cluster plot using graph colors

# In[7]:


def compute_color_center_distance(simu_traj, graph, color1, color2):
    # Get center of mass vector for each color
    color1_atoms = [n.atom.index for n in graph.nodes if n.color == color1]
    color2_atoms = [n.atom.index for n in graph.nodes if n.color == color2]
    center1 = md.compute_center_of_mass(simu_traj.atom_slice(color1_atoms))
    center2 = md.compute_center_of_mass(simu_traj.atom_slice(color2_atoms))
    # TODO periodic distance
    return np.linalg.norm(center1 - center2, axis=1)


def create_cluster_plots(simulation, pairs, graph, cvs):
    logger.info("Creating cluster plots for %s colors", len(graph.colors))
    for i, cvx in enumerate(cvs):
        for j in range(i + 1, len(cvs)):
            cvy = cvs[j]
            cluster_scatterplot(
                simulation,
                cvx.eval(simulation.traj),
                cvy.eval(simulation.traj),
                xlabel=cvx.id,
                ylabel=cvy.id,
                title="Simulation " + simulation.condition + "-" +
                      simulation.number,
                alpha=0.15)


# ## Fig. 1 A from Dror paper
# expecting to see 3 clusters in active, intermediate and inactive states

# In[8]:


def plot_helix6helix3dist_npxxy_rmsd(simulation, active_traj, inactive_traj):
    q = "chainid 0 and protein and (residue 322 to 327) and name CA"  # TODO name CA?
    active_rmsds, inactive_rmsds = compute_active_inactive_rmsd(simulation.traj, active_traj, inactive_traj, q)
    helix_63_dist = compute_distance_CA_atoms(
        'ARG131', 'LEU272', simulation.traj)
    plt.plot(helix_63_dist, inactive_rmsds)
    plt.xlabel(r'Helix 6-helix 3 distance')
    plt.ylabel(r'NPxxY region rmsd to inactive')
    plt.title("Time parametrized path between states")
    plt.show()
    cluster_scatterplot(simulation, helix_63_dist, inactive_rmsds,
                        xlabel=r'Helix 6-helix 3 distance (nm)',
                        ylabel=r'NPxxY region rmsd to inactive (nm)',
                        title="As in paper, Simulation " + simulation.condition + "-" + simulation.number,
                        alpha=0.3)


# ## Against inactive/active ref structure RMSD

# In[9]:


def plot_active_inactive_rmsds(simulation, active_traj, inactive_traj):
    q = "chainid 0 and protein and name CA"
    active_rmsds, inactive_rmsds = compute_active_inactive_rmsd(
        simulation.traj, active_traj, inactive_traj, q)
    # plt.plot(active_rmsds,inactive_rmsds)
    cluster_scatterplot(
        simulation,
        active_rmsds,
        inactive_rmsds,
        xlabel=r'rmsd to active',
        ylabel=r'rmsd to inactive',
        title="Simulation " + simulation.condition + "-" + simulation.number,
        alpha=0.25)


# ## RMSDs to cluster reps

# In[10]:


def plot_cluster_reps_rmsds(simulation, rep_rmsd_cvs):
    for i, cvi in enumerate(rep_rmsd_cvs):
        rmsdsi = cvi.eval(simulation.traj)
        for j in range(i + 1, len(rep_rmsd_cvs)):
            cvj = rep_rmsd_cvs[j]
            rmsdsj = cvj.eval(simulation.traj)
            cluster_scatterplot(
                simulation,
                rmsdsi,
                rmsdsj,
                xlabel=r'rmsd to cluster rep %s' % (i + 1),
                ylabel=r'rmsd to cluster rep %s' % (j + 1),
                title="Simulation " + simulation.condition + "-" +
                      simulation.number,
                alpha=0.07)
