import numpy as np
import pandas as pd
import networkx as nx

import plotly.express as px
import plotly.graph_objects as go


def create_graph(corr_matrix, threshold):
    corr_matrix = corr_matrix.stack().reset_index()
    corr_matrix.columns = ["Var A", "Var B", "correlation"]

    # Retain Features where correlation is above threshold
    corr_matrix = corr_matrix.loc[abs(corr_matrix["correlation"]) >= threshold]

    # Remove Features that are Same
    corr_matrix = corr_matrix.loc[corr_matrix["Var A"] != corr_matrix["Var B"]]

    # create a new graph from edge list
    Gx = nx.from_pandas_edgelist(
        corr_matrix, "Var A", "Var B", edge_attr=["correlation"])

    # Remove Nodes that are isolated
    Gx.remove_nodes_from(list(nx.isolates(Gx)))

    # Remove Smaller Tree Graph
    cmp_len_arr = list(map(lambda x: len(x), nx.connected_components(Gx)))

    for component in list(nx.connected_components(Gx)):
        if len(component) < max(cmp_len_arr):
            for node in component:
                Gx.remove_node(node)

    return Gx


def assign_colour(correlation):
    return "#ffa09b" if correlation <= 0 else "#9eccb7"


def assign_thickness(correlation, benchmark_thickness=2, scaling_factor=3):
    return benchmark_thickness * abs(correlation) ** scaling_factor


def assign_node_size(degree, scaling_factor=50):
    return degree * scaling_factor


def get_edge_width(Gx):
    edge_width = []

    for key, value in nx.get_edge_attributes(Gx, "correlation").items():
        edge_width.append(assign_thickness(value))

    return edge_width


def get_edge_color(Gx):
    # assign edge colours
    edge_colours = []

    for key, value in nx.get_edge_attributes(Gx, "correlation").items():
        edge_colours.append(assign_colour(value))

    return edge_colours


def get_node_size(Gx, size):
    # assign node size depending on number of connections (degree)
    node_size = []

    for key, value in dict(Gx.degree).items():
        node_size.append(assign_node_size(value, size))

    return node_size


def get_coordinates(Gx, func):
    """Returns the positions of nodes and edges in a format
    for Plotly to draw the network
    """
    # get list of node positions
    pos = func(Gx)

    Xnodes = [pos[n][0] for n in Gx.nodes()]
    Ynodes = [pos[n][1] for n in Gx.nodes()]

    Xedges = []
    Yedges = []
    for e in Gx.edges():
        # x coordinates of the nodes defining the edge e
        Xedges.extend([pos[e[0]][0], pos[e[1]][0], None])
        Yedges.extend([pos[e[0]][1], pos[e[1]][1], None])

    return Xnodes, Ynodes, Xedges, Yedges


def get_top_and_bottom_three(df):
    """
    get a list of the top 3 and bottom 3 most/least correlated assests
    for each node.

    Args:
        df (pd.DataFrame): pandas correlation matrix

    Returns:
        top_3_list (list): list of lists containing the top 3 correlations
            (name and value)
        bottom_3_list (list): list of lists containing the bottom three
            correlations (name and value)
    """

    top_3_list, bottom_3_list = [], []

    for col in df.columns:

        # exclude self correlation #reverse order of the list returned
        top_3 = list(np.argsort(abs(df.loc[:, col]))[-4:-1][::-1])
        # bottom 3 list is returned in correct order
        bottom_3 = list(np.argsort(abs(df.loc[:, col]))[:3])

        # get column index
        col_index = df.columns.get_loc(col)

        # find values based on index locations
        top_3_values = [
            f"{df.index[ind]}: {df.iloc[ind, col_index]:.2f}" for ind in top_3]
        bottom_3_values = [
            f"{df.index[ind]}: {df.iloc[ind, col_index]:.2f}" for ind in bottom_3]

        top_3_list.append("<br>".join(top_3_values)+"<br>")
        bottom_3_list.append("<br>".join(bottom_3_values)+"<br>")

    return top_3_list, bottom_3_list


def network_graph(corr_matrix, title, func, threshold=0.75):
    # Create Basic Graph from Correlation Matrix
    Gx = create_graph(corr_matrix, threshold)

    # Make Graph into Minimum Spanning Tree
    mst = nx.minimum_spanning_tree(Gx)

    # Get Edge Colours
    edge_color = get_edge_color(mst)

    # Get Node Size
    node_size = get_node_size(mst, 8)

    # Get Node Labels
    node_label = list(mst.nodes())

    # get coordinates for nodes and edges
    Xnodes, Ynodes, Xedges, Yedges = get_coordinates(mst, func)

    # Description
    top_3_list, bottom_3_list = get_top_and_bottom_three(corr_matrix)

    description = [
        f"<b>{node}</b><br>Strongest Correlation With: <br>{top_3_list[ind]}<br>Weakest Correlation With: <br>{bottom_3_list[ind]}" for ind, node in enumerate(node_label)
    ]

    # edges
    tracer = go.Scatter(
        x=Xedges,
        y=Yedges,
        mode="lines",
        line=dict(color="#DCDCDC", width = 3),
        hoverinfo="none",
        showlegend=False,
    )

    # nodes
    tracer_marker = go.Scatter(
        x=Xnodes,
        y=Ynodes,
        mode="markers+text",
        textposition="top center",
        marker=dict(size=node_size, line=dict(width=1), color=edge_color),
        text=node_label,
        hoverinfo="text",
        hovertext=description,
        textfont = dict(size=15),
        showlegend=False,
    )

    layout = dict(
        title = title,
        width = 800,
        height = 800,
        hovermode = "closest",
        plot_bgcolor = "#fff",
    )

    fig = go.Figure()
    fig.add_trace(tracer)
    fig.add_trace(tracer_marker)
    fig.update_layout(layout)
    
    # Hide X Axes
    fig.update_xaxes(visible=False)
    
    # Hide Y Axes
    fig.update_yaxes(visible=False)
    return fig
