"""
cluster.py
"""
import pickle as pkl
from collections import defaultdict
import sys
import time
import configparser
from TwitterAPI import TwitterAPI
import networkx as nx
import matplotlib.pyplot as plt
from networkx import edge_betweenness_centrality as betweenness

def robust_request(twitter, resource, params, max_tries=5):

    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_twitter(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    twitter = TwitterAPI(
                   config.get('twitter', 'consumer_key'),
                   config.get('twitter', 'consumer_secret'),
                   config.get('twitter', 'access_token'),
                   config.get('twitter', 'access_token_secret'))
    return twitter

def girvan_newman(G, most_valuable_edge=None):
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return
    if most_valuable_edge is None:
        def most_valuable_edge(G):
            betweenness = nx.edge_betweenness_centrality(G)
            return max(betweenness, key=betweenness.get)
    g = G.copy().to_undirected()
    g.remove_edges_from(g.selfloop_edges())
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)

def _without_most_central_edges(G, most_valuable_edge):
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    return new_components

def most_central_edge(G):
    centrality = betweenness(G, weight='weight')
    return max(centrality, key=centrality.get)

def main():

    graph = nx.read_edgelist('GraphList.txt',delimiter='\t')
    graph_copy = graph.copy()

    components = [c for c in nx.connected_component_subgraphs(graph_copy)]
    #print("The number of components formed are ", len(components))

    m = 0
    communities = defaultdict(list)

    nx.draw(graph)
    plt.savefig('cluster.pdf')
    for i in components:
        comp = girvan_newman(i)
        communities[m].append(tuple(sorted(c) for c in next(comp)))
        m += 1

    pickle_out = open("Cluster_pickle.pickle", "wb")
    pkl._dump(communities, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    main()