import matplotlib.pyplot as plt
import networkx as nx


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
