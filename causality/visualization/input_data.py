# vim:foldmethod=marker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
import networkx as nx
from umap import UMAP


#  Causal Graph {{{ #

def causal_graph(treatment_variable, outcome_variable, covariate_variables,
                 unobserved_confounders=None, instrument_variables=None,
                 output_filename=None):
    graph = nx.DiGraph()

    graph.add_node(treatment_variable, observed="yes")
    graph.add_node(outcome_variable, observed="yes")
    graph.add_edge(treatment_variable, outcome_variable)

    covariate_variables = covariate_variables or tuple()
    instrument_variables = instrument_variables or tuple()
    unobserved_confounders = unobserved_confounders or tuple()

    for covariate in covariate_variables:
        graph.add_node(covariate, observed="yes")
        graph.add_edge(covariate, treatment_variable)
        graph.add_edge(covariate, outcome_variable)

    for unobserved_confounder in unobserved_confounders:
        graph.add_node(covariate, observed="no")
        graph.add_edge(covariate, treatment_variable)
        graph.add_edge(covariate, outcome_variable)

    for instrument in instrument_variables:
        graph.add_node(instrument, observed="yes")
        graph.add_edge(instrument, treatment_variable)

    if output_filename:
        figure, axis = plt.subplots()
        plt.clf()
        nx.draw_networkx(graph, pos=nx.shell_layout(graph), axis=axis)
        plt.axis("off")
        plt.savefig(output_filename)
        plt.draw()

        return graph, figure
    return graph
#  }}} Causal Graph #


def plot_treatment_assignment(dataset,
                              dimensionality_reducer=UMAP(n_neighbors=5, min_dist=0.1, metric="correlation"),
                              alpha=0.25,
                              marker="o",
                              markersize=20,
                              histogram=True,
                              colors={"treated": "yellow", "control": "purple"},
                              axes=None):

    assert len(colors.keys()) == 2, "Only binary treatment assignment is supported!"
    if histogram and axes is not None:
        assert len(axes) == 2, "Need to pass one axis for the scatter plot and one for the histogram."

    embedding = dimensionality_reducer.fit_transform(dataset.covariates)

    colorbar_axis = None
    if axes is None:
        fig, (ax, colorbar_axis, histogram_axis) = plt.subplots(
            ncols=3,
            gridspec_kw={"width_ratios": [1, 0.03, 1]}
        )
        plt.tight_layout()
    else:
        ax, histogram_axis = axes

    treated = embedding[dataset.treatment_assignment == 1]
    control = embedding[dataset.treatment_assignment == 0]

    cmap = plt.cm.jet
    cmap = cmap.from_list(
        "Treated vs control cmap",
        [colors["control"], colors["treated"]],
        len(colors)
    )

    ax.scatter(
        np.asarray(treated[:, 0].tolist() + control[:, 0].tolist()),
        np.asarray(treated[:, 1].tolist() + control[:, 1].tolist()),
        cmap=cmap,
        c=dataset.treatment_assignment,
        alpha=alpha,
        marker=marker,
        s=markersize,
    )

    ax.set_title('Units: treated vs control')
    plt.setp(ax, xticks=[], yticks=[])

    if histogram:
        histogram_axis.hist(
            ["Treated"] * (dataset.treatment_assignment == 1.).sum() +
            ["Control"] * (dataset.treatment_assignment == 0.).sum()
        )

    if colorbar_axis is not None:
        cb = mpl.colorbar.ColorbarBase(
            colorbar_axis, cmap=cmap,
            norm=BoundaryNorm((0, 1, 2), ncolors=2),
            ticks=(0.5, 1.5), spacing='proportional', orientation="vertical")
        cb.set_ticklabels(("Control", "Treated"))
        colorbar_axis.tick_params(axis=u'both', which=u'both', length=0)

        return (ax, colorbar_axis, histogram_axis)

    return ax, histogram_axis
