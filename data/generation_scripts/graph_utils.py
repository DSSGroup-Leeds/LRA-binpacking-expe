import pandas as pd
import networkx as nx
from random import choices, random, randrange, sample
from math import sqrt
from scipy.stats import norm


# Keep affinity distribution of the TClab dataset
pop_affinity = [0, 2, 1, 3, 4]
wei_affinity = [13144, 6556, 3992, 361, 25]

def pick_affinities(n):
    return choices(pop_affinity, weights=wei_affinity, k=n)


def generate_arbitrary_graph(n, density):
    def generate_arb_graph(n, density):
        G = nx.DiGraph()
        G.add_nodes_from(range(1, n+1))
        target_nb_nodes = int(density * n * (n-1))
        nb_nodes = 0

        while nb_nodes < target_nb_nodes:
            # random pick a pair (i,j)
            i = randrange(1, n+1)
            j = randrange(1, n+1)

            if i == j:
                # Avoid self-loop
                continue

            # Add (i,j) as an arc if not already present
            if not G.has_edge(i,j):
                G.add_edge(i,j)
                nb_nodes +=1
        return G

    if density > 0.5:
        # Generate graph with density (1-density)
        # and get its complement
        H = generate_arb_graph(n, 1-density)
        G = nx.complement(H)
        return G

    else:
        G = generate_arb_graph(n, density)
        return G

def generate_arbitrary_dict(G):
    d_aff = {}
    for node in G.nodes():
        list_out = list(G.neighbors(node))
        k = len(list_out)
        d_aff[node] = {
            "inter_degree": int(k),
            "inter_aff": list(zip(list_out, pick_affinities(k)))
        }
    return d_aff

def create_arbitrary_df(n, density, base_df):
    G = generate_arbitrary_graph(n, density)
    d_aff = generate_arbitrary_dict(G)
    df_aff = pd.DataFrame.from_dict(d_aff, orient='index')
    return base_df.join(df_aff)



def generate_normal_neighbors(app_list, app_a, mu, sigma):
    degree = int(round(norm.rvs(loc=mu, scale=sigma)))
    if degree < 0:
        degree = 0
    if degree > (len(app_list)-1):
        degree = (len(app_list)-1)

    tmp_list = app_list.copy()
    tmp_list.remove(app_a) # Avoid self-loop
    neighbors = sample(tmp_list, degree)

    return neighbors

def generate_normal_dict(n, density):
    app_list = list(range(1, n+1))
    mu = n * density
    sigma = mu / 2.0

    d_aff = {}
    for app_a in app_list:
        list_out = generate_normal_neighbors(app_list, app_a, mu, sigma)
        k = len(list_out)
        d_aff[app_a] = {
            "inter_degree": int(k),
            "inter_aff": list(zip(list_out, pick_affinities(k)))
        }
    return d_aff

def create_normal_df(n, density, base_df):
    d_aff = generate_normal_dict(n, density)
    df_aff = pd.DataFrame.from_dict(d_aff, orient='index')
    return base_df.join(df_aff)


def generate_threshold_graph(n, density):
    G = nx.DiGraph()
    G.add_nodes_from(range(1, n+1))

    edge_list = []

    v_in = [random() for x in range(n)]
    v_out = [random() for x in range(n)]

    # Correct parameter d for expected density
    if density <= 0.5:
        real_d = (1.0 + sqrt(1 + 8*n *(n-1)*density)) / (4.0*n)
    else:
        real_d = 1.0 + (1 - sqrt(1 + 8*n *(n-1)*(1-density))) / (4.0*n)

    for u in range(n):
        for v in range(u+1,n):
            if ((v_out[u] + v_in[v])/2.0) <= real_d:
                edge_list.append((u+1,v+1))
            if ((v_out[v] + v_in[u])/2.0) <= real_d:
                edge_list.append((v+1,u+1))

    G.add_edges_from(edge_list)
    return G

def generate_threshold_dict(G):
    d_aff = {}
    for node in G.nodes():
        list_out = list(G.neighbors(node))
        k = len(list_out)
        d_aff[node] = {
            "inter_degree": int(k),
            "inter_aff": list(zip(list_out, pick_affinities(k)))
        }
    return d_aff

def create_threshold_df(n, density, base_df):
    G = generate_threshold_graph(n, density)
    d_aff = generate_threshold_dict(G)
    df_aff = pd.DataFrame.from_dict(d_aff, orient='index')
    return base_df.join(df_aff)
