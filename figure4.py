from figure3 import SWRingTopo, SW2DTorusTopo, SW3DHexTorusTopo, CamCubeTopo
from figure3 import SW2DTorusDims, SW3DHexTorusDims, CamCubeDims

from collections import defaultdict
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Number of hops away we should look in a greedy algorithm.
k = 3

SW2DTorusDims = (32, 48)
SW3DHexTorusDims = (16, 8, 12)
CamCubeDims = (8, 12, 16)

###################### Topology Distance Calculations ##########################

def inc_or_dec(n, frm, to):
  """Return whether to go in the +ve direction or the -ve direction.
  """
  c1 = abs(frm - to)
  c2 = n - c1
  if c1 < c2:
    if frm < to:
      return 1
    else:
      return -1
  else:
    if frm < to:
      return -1
    else:
      return 1

def down_or_up(y, z):
  """
  Return whether the vertical edge in 3DHex goes up or down.
  Returns 1 for down, -1 for up.
  """
  return 1 if (y + z) % 2 == 1 else -1

### To create a greedy algorithm for a new topology, implement
### manhattan_distance() and manhattan_next_hop() for that topology.
###
### manhattan_distance() is a function that calculates the shortest distance
### between two nodes using only regular links.
###
### manhattan_next_hop() is a function that finds the next hop in a path
### from one node to another using only regular links.

def SWRing_manhattan_distance(n, frm, to):
  # Path not crossing the (0, n - 1) link.
  c1 = abs(frm - to)
  # Path crossing the (0, n - 1) link.
  c2 = n - c1
  return min(c1, c2)

def SWRing_manhattan_next_hop(n, frm, to):
  assert frm != to
  return (frm + n + inc_or_dec(n, frm, to)) % n


def x_y(node, SW2DTorusDims):
  """Returns the (x, y) coordinates of a given node."""
  h_2d, w_2d = SW2DTorusDims
  x = node // w_2d
  y = node % w_2d
  return (x, y)

def SW2DTorus_manhattan_distance(n, frm, to, SW2DTorusDims=SW2DTorusDims):
  h_2d, w_2d = SW2DTorusDims
  frm_x, frm_y = x_y(frm, SW2DTorusDims)
  to_x, to_y = x_y(to, SW2DTorusDims)
  return (SWRing_manhattan_distance(h_2d, frm_x, to_x) +
      SWRing_manhattan_distance(w_2d, frm_y, to_y))

def SW2DTorus_manhattan_next_hop(n, frm, to, SW2DTorusDims=SW2DTorusDims):
  h_2d, w_2d = SW2DTorusDims
  # This assertion has to be done at some point; we choose to do it here.
  assert h_2d * w_2d == n
  frm_x, frm_y = x_y(frm, SW2DTorusDims)
  to_x, to_y = x_y(to, SW2DTorusDims)
  x_manhattan_distance = SWRing_manhattan_distance(h_2d, frm_x, to_x)
  y_manhattan_distance = SWRing_manhattan_distance(w_2d, frm_y, to_y)
  if x_manhattan_distance > y_manhattan_distance:
    return (SWRing_manhattan_next_hop(h_2d, frm_x, to_x) * w_2d +
        frm_y)
  else:
    return (frm_x * w_2d +
        SWRing_manhattan_next_hop(w_2d, frm_y, to_y))


def x_y_z_sw(node, SW3DHexTorusDims):
  """Returns the (x, y, z) coordinates of a given node in the SW3DHexTorusTopo.
  """
  d_3d, h_3d, w_3d = SW3DHexTorusDims
  x = node // (h_3d * w_3d)
  y = (node % (h_3d * w_3d)) // w_3d
  z = node % w_3d
  return (x, y, z)

def SW3DHexTorus_manhattan_distance(n, frm, to,
                                    SW3DHexTorusDims=SW3DHexTorusDims):
  d_3d, h_3d, w_3d = SW3DHexTorusDims
  frm_x, frm_y, frm_z = x_y_z_sw(frm, SW3DHexTorusDims)
  to_x, to_y, to_z = x_y_z_sw(to, SW3DHexTorusDims)
  x_manhattan_distance = SWRing_manhattan_distance(d_3d, frm_x, to_x)
  y_manhattan_distance = SWRing_manhattan_distance(h_3d, frm_y, to_y)
  z_manhattan_distance = SWRing_manhattan_distance(w_3d, frm_z, to_z)
  ### Check whether the y-axis direction we're traveling in
  ### is the same as the direction of the y-axis regular edge
  ### attached to the frm node. This affects the distance.
  if down_or_up(frm_y, frm_z) == inc_or_dec(h_3d, frm_y, to_y):
    comparison_y_manhattan_distance = y_manhattan_distance - 1
  else:
    comparison_y_manhattan_distance = y_manhattan_distance

  if comparison_y_manhattan_distance <= z_manhattan_distance:
    actual_z_distance = z_manhattan_distance
  else:
    extra = int(math.ceil(
        (comparison_y_manhattan_distance - z_manhattan_distance) / 2))
    actual_z_distance = z_manhattan_distance + 2 * extra
  return x_manhattan_distance + y_manhattan_distance + actual_z_distance

def SW3DHexTorus_manhattan_next_hop(n, frm, to,
                                    SW3DHexTorusDims=SW3DHexTorusDims):
  d_3d, h_3d, w_3d = SW3DHexTorusDims
  # This assertion has to be done at some point; we choose to do it here.
  assert d_3d * h_3d * w_3d == n
  frm_x, frm_y, frm_z = x_y_z_sw(frm, SW3DHexTorusDims)
  ### There are 5 candidate hops; one for each of the regular links
  ### connected to this node. We simply test all of their distances
  ### and pick the least.
  candidate_hops = [
      frm_x * h_3d * w_3d + frm_y * w_3d + (frm_z + 1) % w_3d,
      frm_x * h_3d * w_3d + frm_y * w_3d + (frm_z + w_3d - 1) % w_3d,
      (frm_x * h_3d * w_3d +
          ((frm_y + h_3d + down_or_up(frm_y, frm_z)) % h_3d) * w_3d +
          frm_z),
      ((frm_x + 1) % d_3d) * h_3d * w_3d + frm_y * w_3d + frm_z,
      ((frm_x + d_3d - 1) % d_3d) * h_3d * w_3d + frm_y * w_3d + frm_z,
  ]
  distances = []
  for candidate in candidate_hops:
    distance = SW3DHexTorus_manhattan_distance(n, candidate, to,
                                               SW3DHexTorusDims)
    distances.append((distance, candidate))
  best_next_hop = min(distances, key=lambda x: x[0])
  return best_next_hop[1]


def x_y_z_cc(node, CamCubeDims):
  """Returns the (x, y, z) coordinates of a given node in the CamCubeTopo.
  """
  d_cc, h_cc, w_cc = CamCubeDims
  x = node // (h_cc * w_cc)
  y = (node % (h_cc * w_cc)) // w_cc
  z = node % w_cc
  return (x, y, z)

def CamCube_manhattan_distance(n, frm, to, CamCubeDims=CamCubeDims):
  d_cc, h_cc, w_cc = CamCubeDims
  frm_x, frm_y, frm_z = x_y_z_cc(frm, CamCubeDims)
  to_x, to_y, to_z = x_y_z_cc(to, CamCubeDims)
  return (SWRing_manhattan_distance(d_cc, frm_x, to_x) +
      SWRing_manhattan_distance(h_cc, frm_y, to_y) +
      SWRing_manhattan_distance(w_cc, frm_z, to_z))

def CamCube_manhattan_next_hop(n, frm, to, CamCubeDims=CamCubeDims):
  d_cc, h_cc, w_cc = CamCubeDims
  # This assertion has to be done at some point; we choose to do it here.
  assert d_cc * h_cc * w_cc == n
  frm_x, frm_y, frm_z = x_y_z_cc(frm, CamCubeDims)
  to_x, to_y, to_z = x_y_z_cc(to, CamCubeDims)
  x_manhattan_distance = SWRing_manhattan_distance(d_cc, frm_x, to_x)
  y_manhattan_distance = SWRing_manhattan_distance(h_cc, frm_y, to_y)
  z_manhattan_distance = SWRing_manhattan_distance(w_cc, frm_z, to_z)
  max_distance = max(x_manhattan_distance, y_manhattan_distance,
                     z_manhattan_distance)
  if x_manhattan_distance == max_distance:
    return (SWRing_manhattan_next_hop(d_cc, frm_x, to_x) * h_cc * w_cc +
        frm_y * w_cc +
        frm_z)
  elif y_manhattan_distance == max_distance:
    return (frm_x * h_cc * w_cc +
        SWRing_manhattan_next_hop(h_cc, frm_y, to_y) * w_cc +
        frm_z)
  else:
    return (frm_x * h_cc * w_cc +
        frm_y * w_cc +
        SWRing_manhattan_next_hop(w_cc, frm_z, to_z))

############################# Figure Plotting ##################################

def save_topo(G, name):
  """Saves the topology in an adjacency list text format."""
  nx.readwrite.adjlist.write_adjlist(G, f'{name}.adjlist')

def load_topo(name):
  """
  Loads the topology as a NetworkX graph. Note that node names may be
  inconsistent between loads.
  """
  G = nx.readwrite.adjlist.read_adjlist(f'{name}.adjlist')
  n = len(G)
  mapping = {str(x): x for x in np.arange(n)}
  return nx.relabel.relabel_nodes(G, mapping)

def make_x_hop_lookup(G):
  """
  Create the 3-nearest-neighbour lookup table.

  This is implemented as x_hop_lookup, which consists of zero_hop_lookup,
  one_hop_lookup, two_hop_lookup, and three_hop_lookup tables.

  Parameters:
  G (NetworkX Graph): The topology.

  Returns:
  Dict[Dict[Set]]: An X-hop lookup table, indexed first by number of hops, then
    by source node. Each index contains a set of nodes reachable in exactly X
    hops. X ranges from 0 to 3.
  """
  n = len(G)
  x_hop_lookup = defaultdict(lambda: defaultdict(set))
  # Creating zero_hop_lookup.
  for node in np.arange(n):
    x_hop_lookup[0][node].add(node)
  # Recursively creating the other lookup tables.
  for x in np.arange(1, k+1):
    for node in np.arange(n):
      for neighbor in G.neighbors(node):
        x_hop_lookup[x][node].update(x_hop_lookup[x-1][neighbor])
      # Remove nodes that can be reached in fewer hops.
      for sub in np.arange(1, x+1):
        x_hop_lookup[x][node] -= x_hop_lookup[x-sub][node]
  return x_hop_lookup

def make_routing_table(n, manhattan_distance, manhattan_next_hop,
                       x_hop_lookup=None):
  """
  Make a routing table for a given topology.

  Parameters:
  n (int): The number of nodes in the topology.
  manhattan_distance ([int, int, int] -> int): A topology-specific function.
    Given the endpoints of a path, return the length of the path,
    assuming no random links.
  manhattan_next_hop ([int, int, int] -> int): A topology-specific function.
    Given the endpoints of a path, return the next hop of the path,
    assuming no random links.
  x_hop_lookup (Dict[Dict[Set]]): Optional. The output of make_x_hop_lookup().
    Used to handle the random links added by the topology. If not provided,
    assume the topology only has regular links.

  Returns:
  Dict[Dict[int]]: A greedy routing table, indexed first by source node,
    then by destination node. Contains the next node in the path.
  """
  routing_table = defaultdict(dict)
  for frm in np.arange(n):
    for to in np.arange(n):
      if frm == to:
        routing_table[frm][to] = to
        continue
      candidate_next_hop = manhattan_next_hop(n, frm, to)
      candidate_dist = manhattan_distance(n, candidate_next_hop, to) + 1
      if x_hop_lookup:
        # Look for other candidate hops in the x_hop_lookup table (if present).
        for x in np.arange(1, k+1):
          for x_hop in x_hop_lookup[x][frm]:
            new_dist = manhattan_distance(n, x_hop, to) + x
            if new_dist < candidate_dist:
              ### We now have a node that is x nodes away from frm,
              ###   but we need a node that is one node away
              ###   to form the routing table.
              ### Get the actual next hop using a trick:
              ### Take the intersection of the 1-hop neighbours of frm
              ###   and the (x-1)-hop neighbours of x_hop,
              ###   and pick an arbitrary node.
              intersection = x_hop_lookup[1][frm] & x_hop_lookup[x-1][x_hop]
              candidate_next_hop = intersection.pop()
              candidate_dist = new_dist
      routing_table[frm][to] = candidate_next_hop
  return routing_table

def greedy_routing_table(G, manhattan_distance, manhattan_next_hop):
  """
  Generate a greedy routing table given a topology and its distance functions.

  Curries make_x_hop_lookup and make_routing_table to generate
  a greedy routing table. Can be plugged into greedy_shortest_paths()
  to get the mean and standard deviation path lengths.

  Parameters:
  G (NetworkX Graph): The topology.
  manhattan_distance ([int, int, int] -> int): A topology-specific function.
    Given the endpoints of a path, return the length of the path,
    assuming no random links.
  manhattan_next_hop ([int, int, int] -> int): A topology-specific function.
    Given the endpoints of a path, return the next hop of the path,
    assuming no random links.

  Returns:
  Dict[Dict[int]]: A greedy routing table, indexed first by source node,
    then by destination node. Contains the next node in the path.
  """
  n = len(G)
  print('Creating x_hop_lookup...')
  x_hop_lookup = make_x_hop_lookup(G)
  print('Creating the routing table...')
  routing_table = make_routing_table(n, manhattan_distance, manhattan_next_hop,
                                     x_hop_lookup)
  return routing_table

def regular_routing_table(G, manhattan_distance, manhattan_next_hop):
  """
  Generate a 'greedy' routing table for a topology with only regular links.

  Differs from greedy_routing_table() in that it does not need to discover
  potentially unknown links, as it has no random links. Useful in particular
  for the CamCube topology.

  Parameters:
  G (NetworkX Graph): The topology.
  manhattan_distance ([int, int, int] -> int): A topology-specific function.
    Given the endpoints of a path, return the length of the path,
    assuming no random links.
  manhattan_next_hop ([int, int, int] -> int): A topology-specific function.
    Given the endpoints of a path, return the next hop of the path,
    assuming no random links.

  Returns:
  Dict[Dict[int]]: A greedy routing table, indexed first by source node,
    then by destination node. Contains the next node in the path.
  """
  n = len(G)
  print('Creating the routing table...')
  routing_table = make_routing_table(n, manhattan_distance, manhattan_next_hop)
  return routing_table

def greedy_shortest_paths(G, routing_table):
  """
  Calculate shortest paths using greedy geographical routing.

  Obtains the mean and standard deviation of all the paths in G using a
  greedy routing table generated by make_routing_table().

  Parameters:
  G (NetworkX Graph): The topology.
  routing_table (Dict[Dict[int]]): The greedy routing table generated using
    knowledge of the regular nature of the topology, with some exploration
    performed by each node to take advantage of random links (if present).

  Returns:
  (float, float): The mean and standard deviation of all path lengths generated
    by the routing table on the topology.
  """
  n = len(G)
  paths_seen = 0
  mean = 0
  std = 0
  for frm in np.arange(n):
    for to in np.arange(n):
      # Ignore self-paths.
      if frm == to:
        continue
      # Iteratively follow the routing table to the destination.
      dist = 0
      cur_node = frm
      while cur_node != to:
        assert G.has_edge(cur_node, routing_table[cur_node][to])
        cur_node = routing_table[cur_node][to]
        dist += 1
      # Once the destination has been reached, update the running mean/std.
      # Borrowed from: https://www.kite.com/python/answers/how-to-find-a-running-standard-deviation-in-python
      paths_seen += 1
      new_mean = mean + (dist - mean) / paths_seen
      std += (dist - mean) * (dist - new_mean)
      mean = new_mean
  return (mean, math.sqrt(std / (paths_seen - 1)))

def plot_greedy_shortest_path_lengths(name, manhattan_distance,
                                      manhattan_next_hop,
                                      routing_table_func):
  """Plots one bar of figure 4 of the SWDCs paper."""
  print(f'Loading topology {name}...')
  G = load_topo(name)
  routing_table = routing_table_func(G, manhattan_distance, manhattan_next_hop)
  print('Calculating greedy shortest paths...')
  mean, std = greedy_shortest_paths(G, routing_table)

  plt.bar(name, height=mean, yerr=std,
      color='white', edgecolor='black', width=0.4, capsize=5)

################################### Main #######################################

if __name__ == '__main__':
  # plot_greedy_shortest_path_lengths('SWRingTopo_1024')
  # plt.savefig('shortest_path_lengths_greedy.png')

  plot_greedy_shortest_path_lengths('SWRingTopo_1536',
                                    SWRing_manhattan_distance,
                                    SWRing_manhattan_next_hop,
                                    greedy_routing_table)
  plot_greedy_shortest_path_lengths('SW2DTorusTopo_1536_32_48',
                                    SW2DTorus_manhattan_distance,
                                    SW2DTorus_manhattan_next_hop,
                                    greedy_routing_table)
  plot_greedy_shortest_path_lengths('SW3DHexTorusTopo_1536_16_8_12',
                                    SW3DHexTorus_manhattan_distance,
                                    SW3DHexTorus_manhattan_next_hop,
                                    greedy_routing_table)
  plot_greedy_shortest_path_lengths('CamCubeTopo_1536_8_12_16',
                                    CamCube_manhattan_distance,
                                    CamCube_manhattan_next_hop,
                                    regular_routing_table)
  plt.savefig('shortest_path_lengths_greedy.png')

  # save_topo(SWRingTopo(n=1536), 'SWRingTopo_1536')
  # save_topo(SW2DTorusTopo(dims=(32, 48)), 'SW2DTorusTopo_1536_32_48')
  # save_topo(SW3DHexTorusTopo(dims=(16, 8, 12)), 'SW3DHexTorusTopo_1536_16_8_12')
  # save_topo(CamCubeTopo(dims=(8, 12, 16)), 'CamCubeTopo_1536_8_12_16')
