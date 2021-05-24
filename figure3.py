import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

n = 10240
degree = 6

SW2DTorusDims = [80, 128]
# Depth, long edge (vertical in diagram), short edge (horizontal).
SW3DHexTorusDims = [20, 16, 32]
CamCubeDims = [16, 20, 32]

def draw_graph(G):
  nx.drawing.nx_pylab.draw(G)
  plt.savefig('G.png')

def flatten_shortest_path_length(lengths):
  l = []
  for k1, sub in lengths.items():
    for k2, v in sub.items():
      if k1 == k2:
        continue
      l.append(v)
  return l

def reverse_pairs(sett):
  return set((entry[1], entry[0]) for entry in sett)

# def add_random_links(G, random_links):
#   for k in np.arange(np.floor(random_links)):
#     print('Adding random link set %d...' % (k + 1))
#     # Repeat until a valid set of new links is generated.
#     while True:
#       perm = np.random.permutation(n)
#       # Build a list of candidate links from a random permutation.
#       candidate_links = set((i, perm[i]) for i in np.arange(n))
#       # Links are bidirectional; edges are undirected.
#       bidirectional_links = candidate_links | reverse_pairs(candidate_links)
#       # Check for duplicate links and self-links.
#       if len(bidirectional_links) < len(candidate_links) * 2:
#         continue
#       # Reject existing links.
#       if len(bidirectional_links & set(G.edges)) > 0:
#         continue
#       # Add the valid set to the graph.
#       G.add_edges_from(candidate_links)
#       break

def add_random_links(G, random_links_per_node):
  print('Adding random links...')
  random_links = random_links_per_node * n / 2
  # Store candidate links in a temporary graph so we don't remove
  # a regular link.
  temp_G = nx.Graph()
  temp_G.add_nodes_from(G)
  for _ in np.arange(random_links):
    # Repeat until a valid link is found.
    while True:
      random_link = np.random.choice(n, size=2, replace=False)
      if not G.has_edge(*random_link) and not temp_G.has_edge(*random_link):
        break
    temp_G.add_edge(*random_link)
  # Swap links around until all nodes have the correct degree.
  # Picks a random neighbour from the highest degree node
  # and swaps the highest degree node with the lowest degree node possible.
  print('Swapping random links around...')
  degrees = list(temp_G.degree())
  degrees.sort(key=lambda x:x[1])
  while degrees[0][1] != degrees[-1][1]:
    node1 = degrees[-1][0]
    node2 = np.random.choice(list(temp_G.neighbors(node1)))
    for i in np.arange(n - 1):
      node3 = degrees[i][0]
      if not temp_G.has_edge(node2, node3) \
          and not G.has_edge(node2, node3) \
          and node2 != node3:
        break
    temp_G.remove_edge(node1, node2)
    temp_G.add_edge(node2, node3)
    # Recalculate the list of degrees every iteration. This is inefficient
    # since we know which edges swapped, but it's convenient.
    degrees = list(temp_G.degree())
    degrees.sort(key=lambda x:x[1])
  # Add candidate links to the original graph.
  G.add_edges_from(temp_G.edges)

def SWRingTopo():
  G = nx.Graph()
  for i in np.arange(n):
    G.add_node(i)
  # Regular links.
  print('Adding regular links...')
  for i in np.arange(n):
    G.add_edge(i, (i + 1) % n)
  # Number of ports assigned to random links.
  random_links = degree - 2
  # Add random_links sets of n/2 random links.
  add_random_links(G, random_links)
  return G

def SW2DTorusTopo():
  G = nx.Graph()
  for i in np.arange(n):
    G.add_node(i)
  # Regular links.
  print('Adding regular links...')
  x, y = SW2DTorusDims
  for i in np.arange(x):
    for j in np.arange(y):
      G.add_edge(i * y + j, (i * y + j + 1) % y)
      G.add_edge(i * y + j, ((i + 1) % x) * y + j)
  # Number of ports assigned to random links.
  random_links = degree - 4
  # Add random_links sets of n/2 random links.
  add_random_links(G, random_links)
  return G

def SW3DHexTorusTopo():
  G = nx.Graph()
  for i in np.arange(n):
    G.add_node(i)
  # Regular links.
  print('Adding regular links...')
  x, y, z = SW3DHexTorusDims
  for i in np.arange(x):
    for j in np.arange(y):
      for k in np.arange(z):
        index = i * y * z + j * z + k
        # Horizontal edge.
        G.add_edge(index, (index + 1) % z)
        # Vertical edge.
        if ((j + k) % 2 == 1):
          G.add_edge(index, (index + z) % (y * z))
        # Z-dimension edge (labelled as x here).
        G.add_edge(index, (index + y * z) % n)
  # Number of ports assigned to random links.
  random_links = degree - 5
  # Add random_links sets of n/2 random links.
  add_random_links(G, random_links)
  return G

def CamCubeTopo():
  G = nx.Graph()
  for i in np.arange(n):
    G.add_node(i)
  # Regular links.
  print('Adding regular links...')
  x, y, z = CamCubeDims
  for i in np.arange(x):
    for j in np.arange(y):
      for k in np.arange(z):
        index = i * y * z + j * z + k
        # Horizontal edge.
        G.add_edge(index, (index + 1) % z)
        # Vertical edge.
        G.add_edge(index, (index + z) % (y * z))
        # Z-dimension edge (labelled as x here).
        G.add_edge(index, (index + y * z) % n)
  return G

def plot_shortest_path_lengths(topo_func, name):
  print('Creating topology: %s...' % name)
  G = topo_func()
  print('Flattening lengths...')
  shortest_path_lengths = flatten_shortest_path_length(
      dict(nx.shortest_path_length(G)))
  print('Plotting...')
  mean = np.mean(shortest_path_lengths)
  err = np.std(shortest_path_lengths)

  plt.bar(name, height=mean, yerr=err,
      color='white', edgecolor='black', width=0.4, capsize=5)

plot_shortest_path_lengths(SWRingTopo, 'SW Ring')
plot_shortest_path_lengths(SW2DTorusTopo, 'SW 2DTor')
plot_shortest_path_lengths(SW3DHexTorusTopo, 'SW 3DHexTor')
plot_shortest_path_lengths(CamCubeTopo, 'CamCube')
plt.savefig('shortest_path_lengths.png')

# draw_graph(SW2DTorusTopo())
