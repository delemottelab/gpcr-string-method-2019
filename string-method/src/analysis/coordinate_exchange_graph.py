from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

import networkx as nx
import numpy as np
import utils
logger = logging.getLogger("CoordinateExchange")


def create_argparser():
    parser = argparse.ArgumentParser(epilog='By Oliver Fleetwood 2017-2018.')
    parser.add_argument('-wd', '--working_dir', type=str, help='working directory', required=True)
    parser.add_argument('-sp', '--string_filepath', type=str,
                        help='string textfile path with which should be formatted with the current iteration number',
                        default="string%s.txt")
    parser.add_argument('-od', '--out_dir', type=str, help='output directory', required=True)
    parser.add_argument('--min_iteration', type=int, help="first iteration to include", required=False, default=1)
    parser.add_argument('--max_iteration', type=int, help='last iteration to include', required=False, default=200)
    parser.add_argument("--fixed_endpoints", help="Hold string endpoints fixed",
                        type=lambda x: (str(x).lower() == 'true'),
                        default=True)
    return parser


class MappingGraph(nx.DiGraph):
    def __init__(self, args={}, root_id="Targeted MD"):
        nx.DiGraph.__init__(self)
        self.args = args
        self.max_iteration = None  # Not the same as args.max_iteration. This is the last we managed to find
        self.root_id = root_id

    def create_default_mapping(self, previous_stringpath, stringpath, fixed_endpoints):
        mapping = np.zeros((len(stringpath), 3)) - 1
        point_ratio = len(previous_stringpath) / len(stringpath)
        if fixed_endpoints:
            mapping[0, 0] = 0
            mapping[-1, 0] = len(previous_stringpath) - 1
        for point_idx in range(len(mapping)):
            if fixed_endpoints and (point_idx == 0 or point_idx == len(stringpath) - 1):
                continue
            previous_idx = utils.rint(point_idx * point_ratio)
            mapping[point_idx, 0] = previous_idx
        return mapping

    def to_node_id(self, iteration, point_idx):
        return "i{}p{}".format(utils.rint(iteration), utils.rint(point_idx))

    def add_to_graph(self, mapping, iteration):
        for point_idx, point_mapping in enumerate(mapping):
            if self._exclude_point(point_idx, len(mapping)):
                continue
            point_name = self.attach_node(iteration, point_idx)
            previous_point = point_mapping[0]  # TODO maybe add swarm number to label
            self.add_edge(self.to_node_id(iteration - 1, previous_point), point_name)

    def add_root(self, iteration, npoints):
        self.add_node(self.root_id, {"iteration": -1, "point_idx": -1})
        for point_idx in range(npoints):
            if self._exclude_point(point_idx, npoints):
                continue
            point_name = self.attach_node(iteration, point_idx)
            self.add_edge(self.root_id, point_name)

    def map(self):
        filepath = args.working_dir + args.string_filepath
        graph = nx.DiGraph()
        last_stringpath = np.loadtxt(filepath % (args.min_iteration - 1))
        self.add_root(args.min_iteration - 1, len(last_stringpath))
        for i in range(args.min_iteration, args.max_iteration + 1):
            try:
                mappingfile = filepath % (str(i) + "-mapping")
                stringfile = filepath % i
                if os.path.exists(mappingfile):
                    mapping = np.loadtxt(mappingfile)
                    stringpath = np.loadtxt(stringfile)
                elif not os.path.exists(mappingfile) and os.path.exists(stringfile):
                    stringpath = np.loadtxt(stringfile)
                    mapping = self.create_default_mapping(last_stringpath, stringpath, args.fixed_endpoints)
                else:
                    logger.info("Did not find file for iteration %s. Stopping here", i)
                    break
            except Exception as ex:
                logger.exception(ex)
                logger.info("Did not find file for iteration %s. Stopping here", i)
            self.add_to_graph(mapping, i)
            last_stringpath = stringpath
        self.max_iteration = i
        return graph

    def visualize(self, only_bottom_leafs=True, only_exchanges=True):
        graph = self.filter(only_bottom_leafs, only_exchanges)
        A = nx.nx_agraph.to_agraph(graph)
        A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
        A.draw(args.out_dir + 'coordinate_graph_{}to{}_{}_{}.svg'.format(self.args.min_iteration, self.max_iteration,
                                                                         "only_bottom_leafs" if only_bottom_leafs else "",
                                                                         "only_exchanges" if only_exchanges else ""))

    def _exclude_point(self, point_idx, npoints):
        return self.args.fixed_endpoints and (point_idx == 0 or point_idx == npoints - 1)

    def attach_node(self, iteration, point_idx):
        node_id = self.to_node_id(iteration, point_idx)
        node = self.add_node(node_id, {"iteration": iteration, "point_idx": point_idx})
        return node_id

    def filter(self, only_bottom_leafs, only_exchanges):
        if only_bottom_leafs:
            bottom_leafs = [n for n, d in self.nodes_iter(data=True) if d.get('iteration', None) == self.max_iteration]
            nodes = set()
            for l in bottom_leafs:
                nodes.add(l)
                child = l
                while True:
                    predecessors = self.predecessors(child)
                    if len(predecessors) == 0:
                        break
                    else:
                        parent = predecessors[0]
                    if not only_exchanges or self.node[parent]["point_idx"] != self.node[child]["point_idx"]:
                        nodes.add(parent)
                        child = parent
                    else:
                        break
            return self.subgraph(nbunch=nodes)
        elif only_exchanges:
            #Find longest list of sequential exchanges starting from leaves
            leafs = [n for n in self.nodes_iter(data=False) if len(self.successors(n)) == 0]
            # starts = [self.node[self.root_id]]
            nodes = set()
            for l in leafs:
                nodes.add(l)
                child = l
                while True:
                    parents = self.predecessors(child)
                    if len(parents) == 0:
                        break
                    else:
                        parent = parents[0]
                    if self.node[parent]["point_idx"] != self.node[child]["point_idx"]:
                        nodes.add(parent)
                        child = parent
                    else:
                        break
            subgraphs = [subgraph for subgraph in nx.connected_component_subgraphs(self.subgraph(nbunch=nodes).to_undirected())]
            sorted_subgraphs = sorted(subgraphs, cmp=lambda s1, s2: len(s2) - len(s1))
            longest_graph = sorted_subgraphs[0]
            return self.subgraph(nbunch=[n for n in longest_graph.nodes_iter(data=False)])
        else:
            return self


if __name__ == "__main__":
    args = create_argparser().parse_args()
    graph = MappingGraph(args)
    graph.map()
    graph.visualize(only_bottom_leafs=False, only_exchanges=False)
