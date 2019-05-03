from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# if os.environ.get("DISPLAY", None) is not None:
#     matplotlib.use('GTK3Cairo')  #
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colvars
import utils
from utils import ticks_labelsize, label_fontsize, pdb_fontsize, marker_size, linewidth, simuid_to_color, \
    simuid_to_label
import numpy as np

import networkx as nx

logger = logging.getLogger("FEVisualizer")
field_cmap = plt.cm.jet  # hot
string_color = "darkorchid"
field_color_norm = colors.PowerNorm(gamma=1 / 3)
simuid_to_string_events = {
    #based on iter 212
    "holo-optimized": [(52, 'Ionic lock forms'), (12, 'NPxxY twist'), (19, "YY-bond breaks"),
                       (14, "Active/intermediate\ncluster transition"),
                       (43, "Intermediate/inactive\ncluster transition")],
    #based on iter 232
    "apo-optimized": [(55, 'Ionic lock forms'), (11, 'NPxxY twist 1'), (31, 'NPxxY twist 2'), (43, "YY-bond breaks"),
                      (14, "Active/intermediate\ncluster transition"),
                      (45, "Intermediate/inactive\ncluster transition")]
}


def plot_field(runner, fields, cvs, gridmin, gridmax,
               rescale=False,
               stringpath=None,
               cutoff=12,
               stringlabel=None,
               title="Free Energy",
               cv_labels=None,
               field_label=r'$\Delta G$',
               axisdata=None,
               plot_reference_structures=True):
    simu_id = runner.simu_id
    description = simuid_to_label.get(simu_id, simu_id)
    if cv_labels is None:
        cv_labels = [cv.id for cv in cvs]
    if axisdata is None:
        axisdata = get_axis(gridmin, gridmax, fields)
    ref_structs = utils.load_many_reference_structures()
    # Remove undefined values for field
    if len(cvs) == 1:
        i = 0
        cvi = cvs[i]
        y, boxplot_data = _fix_field_and_data(fields, cutoff)
        x = colvars.rescale_evals(axisdata, [cvi]) if rescale else axisdata
        # plot_color = simuid_to_color.get(simu_id)
        # plt.plot(x, field, label=description, color=plot_color, linewidth=linewidth)
        # plt.scatter(x, field, s=marker_size, color=plot_color)
        # plt.ylabel(field_label, fontsize=label_fontsize)
        # plt.xlabel(cv_labels[i], fontsize=label_fontsize)
        # plt.tick_params(labelsize=ticks_labelsize)
        utils.plot_path(y,
                        label=description,
                        scatter=False,
                        legend=True,
                        twoD=False,
                        ncols=1,
                        color=simuid_to_color.get(simu_id),
                        boxplot_data=boxplot_data,
                        ticks=x,
                        xticks=np.arange(x.min(), x.max(), (x.max() - x.min()) / 10),
                        axis_labels=[cv_labels[i], field_label])
        if plot_reference_structures:
            for text, struct in ref_structs:
                try:
                    ci = cvi.eval(struct)
                    if rescale:
                        ci = colvars.rescale_evals(ci, [cvi])[0]
                    yvalue = 0  # 0.5 + np.random.rand()
                    plt.scatter(ci, yvalue, marker="d", color="black", s=marker_size)
                    plt.text(ci, yvalue, text, color="gray", fontsize=pdb_fontsize, rotation=45)
                except Exception as ex:
                    logger.exception(ex)
        # plt.legend(fontsize=label_fontsize)
        # plt.grid()
        plt.title(title)
    elif len(cvs) == 2:
        i, j = 0, 1
        cvi = cvs[i]
        cvj = cvs[j]

        def twoD_plot(z, levels, field_label=field_label, xlabel=cv_labels[i], ylabel=cv_labels[j]):
            im = plt.contourf(x,
                              y,
                              z,
                              levels=levels,
                              norm=field_color_norm,
                              cmap=field_cmap)
            ct = plt.contour(x,
                             y,
                             z,
                             levels=levels,
                             # extent=[xmin, xmax, ymin, ymax],
                             alpha=0.3,
                             colors=('k',))
            plt.grid()
            if xlabel is not None:
                plt.xlabel(xlabel, fontsize=label_fontsize)
            if ylabel is not None:
                plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.tick_params(labelsize=ticks_labelsize)
            # Colorbar
            cbar = plt.colorbar(im, orientation='vertical')
            cbar.set_label(field_label, fontsize=label_fontsize)
            cbar.ax.tick_params(labelsize=ticks_labelsize)
            if title is not None:
                plt.title(title + "," + description)
            # cbar.set_clim(vmin=0, vmax=cutoff)
            # Additional stuff to plot
            if plot_reference_structures:
                for text, struct in ref_structs:
                    try:
                        ci, cj = cvi.eval(struct), cvj.eval(struct)
                        if rescale:
                            ci = colvars.rescale_evals(ci, [cvi])[0]
                            cj = colvars.rescale_evals(cj, [cvj])[0]
                        plt.scatter(ci, cj, marker="d", color="black", s=marker_size)
                        plt.text(ci + 0.01, cj, text, color="gray", fontsize=pdb_fontsize)
                    except Exception as ex:
                        logger.error("Failed to display pdb structure %s", text)
                        logger.exception(ex)

            if stringpath is not None:
                plt.plot(stringpath[:, 0], stringpath[:, 1], '-', label=stringlabel, linewidth=linewidth,
                         color=string_color)
                # plt.scatter(stringpath[:, 0], stringpath[:, 1])
                plt.legend(fontsize=label_fontsize)
                plt.scatter(stringpath[0, 0], stringpath[0, 1], marker="*", color='black', s=marker_size)  # s=200
                plt.scatter(stringpath[-1, 0], stringpath[-1, 1], marker="^", color='black', s=marker_size)
                for point_idx, text in simuid_to_string_events.get(simu_id, []):
                    plt.text(stringpath[point_idx, 0], stringpath[point_idx, 1], text, color="black",
                             fontsize=pdb_fontsize - 6, bbox=dict(facecolor='white', alpha=0.5))

        delta_z = None
        if len(fields.shape) == 3:
            z = np.mean(fields, axis=0)
            if fields.shape[0] > 1:
                delta_z = np.std(fields, axis=0)
                delta_z = delta_z.T
        else:
            z = fields
        z = _fix_range(z, cutoff)
        z = z.T  # Transpose it here so that it plots properly. In matplot lib field[i,:] gives us a row along the x-axis for the i:th y-value
        if rescale:
            x = colvars.rescale_evals(axisdata, [cvi])
            y = colvars.rescale_evals(axisdata, [cvj])
        else:
            x, y = axisdata, axisdata
        # Set the levels parameter of the plots so that we resolve regions of lower FE  (since those are typically statistically more accurate and biologically interesting)
        # levels = np.append(np.arange(0, 2.0, 0.25),np.arange(2.0, 6.0, 0.5))
        field_levels = np.array([0.06 * step ** 2 for step in range(0, 30)])
        field_levels = field_levels[field_levels < 5]
        if delta_z is None:
            plt.subplot(1, 1, 1)
            twoD_plot(z, field_levels)
        else:
            plt.subplot(1, 2, 1)
            twoD_plot(z, field_levels)
            plt.subplot(1, 2, 2)
            twoD_plot(delta_z, None, field_label=r'$\sigma$ [kcal/mol]', ylabel=None)
        # plt.tight_layout()
    else:
        raise Exception("Plotting more than 2 dimensions not supported yet. Size: %s" % (len(cvs)))
        # plt.show()


def plot_networks(stationary_state, free_energies, transition_matrix, cvs, save=True,
                  outputpath="/home/oliverfl/Pictures/network-%s.svg",
                  description="Transition Network", simu_id=None):
    """
    Do imports only when necessarry to avoid errors with GTK symbols backends
    See the error message I get on https://stackoverflow.com/questions/19773190/graph-tool-pyside-gtk-2-x-and-gtk-3-x
    """
    # import gtk
    import graph_tool as gt
    from graph_tool.all import graph_draw, Graph
    # from pylab import *  # for plotting
    if len(stationary_state.shape) == 2:
        stationary_state = np.mean(stationary_state, axis=0)
    free_energy, boxplot_data = _fix_field_and_data(free_energies, 1000)
    # Create graph and labels
    graph = Graph(directed=True)
    vertices = []
    vertex_sizes = graph.new_vertex_property("float")
    vertex_labels = graph.new_vertex_property("string")
    edge_sizes = graph.new_edge_property("float")
    edge_labels = graph.new_edge_property("string")
    cluster_to_label = {
        0: "M",  # "#"Intermediate",
        1: "I",  # "Inactive",
        2: "A"  # "Active"
    }
    #####CREATE VERTICES###########
    for i, rho in enumerate(stationary_state):
        # print(rho)
        v = graph.add_vertex()
        vertices.append(v)
        vsize = (1 + 1 * rho / max(stationary_state)) * 50  # np.log(1 + rho / max(stationary_state)) * 100
        vertex_sizes[v] = vsize
        # Beware that long tables can make the nodes expand to fit the text (and thus override vsize)
        vertex_labels[v] = cluster_to_label.get(i) + " ({0:0.01f})".format(free_energy[i])
    ####CREATE EDGES##########
    max_transition_value = (transition_matrix - np.diag(
        transition_matrix.diagonal())).max()  # Max value of matrix excluding diagonal elements
    for i, row in enumerate(transition_matrix):
        total_traj_count = sum(row)
        for j, rhoij in enumerate(row):
            if rhoij > 0 and i != j:
                e = graph.add_edge(vertices[i], vertices[j])  # , weight=rhoij, label="{}->{}".format(i, j))
                edge_labels[e] = "{}/{}".format(int(rhoij), int(total_traj_count))
                # The edge width is proportional to the relative number of transition from this starting state
                size = (1 + 10 * rhoij / total_traj_count) * 3
                edge_sizes[e] = size
    # Using matplotlib for rendering if save==False
    plt.figure(figsize=(10, 10))
    graph_draw(graph,
               pos=gt.draw.sfdp_layout(graph),
               # output_size=(400, 400),
               output=outputpath % simu_id if save else None,
               # inline=True,
               mplfig=plt.gcf() if not save else None,
               vertex_text=vertex_labels,
               vertex_font_size=10,
               vertex_size=vertex_sizes,
               edge_text=edge_labels,
               edge_pen_width=edge_sizes)
    if not save:
        plt.title(description)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()
        # graphviz_draw(g, layout="neato")#, pos=pos)


def plot_networks_networkx(stationary_state, transition_matrix, cvs, title="Network"):
    graph = nx.MultiDiGraph()

    for i, rho in enumerate(stationary_state):
        # print(rho)
        graph.add_node(i, label=str(i))
    for i, row in enumerate(transition_matrix):
        for j, rhoij in enumerate(row):
            if rhoij > 0:
                graph.add_edge(i, j, weight=rhoij, label="{}->{}".format(i, j))
    nx.draw_networkx(graph, pos=nx.fruchterman_reingold_layout(graph), node_size=np.log(stationary_state * 1e3) * 5e2,
                     node_color="gray", alpha=0.8)
    plt.show()
    # pos = nx.spring_layout(graph)  # positions for all nodes
    # for n, p in pos.iteritems():
    #     graph.node[n]['pos'] = p
    # _plotly_network(graph)


def get_axis(gridmin, gridmax, field):
    nbins = field.shape[0] if len(field.shape) == 1 else field.shape[1]
    gridsize = (gridmax - gridmin) / nbins
    gridoffset = gridsize / 2  # A small ofset since we round to lower values on average
    axis = np.arange(gridmin, gridmax, gridsize) + gridoffset
    return axis


def _fix_range(field, cutoff, minvalue=None):
    if minvalue is None:
        minvalue = field.min()
    field -= minvalue
    field[np.isnan(field)] = sys.float_info.max
    field[field > cutoff] = np.nan
    return field


def _fix_field_and_data(fields, cutoff):
    boxplot_data = None
    if len(fields.shape) == 2:
        y = np.mean(fields, axis=0)
        if fields.shape[0] > 1:
            boxplot_data = _fix_range(fields, cutoff, minvalue=y.min())
    else:
        y = fields
    y = _fix_range(y, cutoff)
    return y, boxplot_data
