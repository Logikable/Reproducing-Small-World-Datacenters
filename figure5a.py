from mininet.log import lg 

import ipmininet
from ipmininet.cli import IPCLI
from ipmininet.ipnet import IPNet 
from ipmininet.iptopo import IPTopo 
from ipmininet.router.config.base import RouterConfig
from ipmininet.router.config import StaticRoute
from ipmininet.router.config.zebra import Zebra 

import numpy as np
import matplotlib.pyplot as plt

from figure4 import load_routing_table

class CustomTopo(IPTopo):
  def __init__(self, adj_name, routing_table_name):
    self.adj = {}
    self.ip_addresses = {}
    self.routing_table = {}

    # parse adjacency list
    print('Parsing adjacency list...')
    with open(f'{adj_name}.adjlist', 'r') as f:
      for line in f.readlines():
        # ignore commented and blank lines
        if line == '' or line.startswith('#'):
          continue

        tokens = line.split(' ')

        # update adjacency list
        v = int(tokens[0])
        neighbors = [int(tokens[i]) for i in range(1, len(tokens))]
        self.adj[v] = neighbors

        # generate ip address
        self.ip_addresses[v] = self.node_to_ip(v)

    # parse routing table
    print('Parsing routing table...')
    self.routing_table = load_routing_table(routing_table_name)

    # calculate number of nodes
    self.n = max(self.adj.keys())

    IPTopo.__init__(self)

  def node_to_ip(self, v):
    """
    Converts a node id to an IPv6 address.
    Assumes v <= 9999.
    """
    return f'2001:2345:{str(v)}::'

  def build(self, *args, **kwargs):
    routers = {}

    # build routers
    print('Adding routers...')
    for u, tmp in self.routing_table.items():
      routes = []
      for v, next_hop in tmp.items():
        routes.append(StaticRoute(self.ip_addresses[v], self.ip_addresses[next_hop]))
      routers[u] = self.addRouter_v6(f'r{u}', routes)

    # build links
    print('Adding links...')
    for u, neighbors in self.adj.items():
      for v in neighbors:

        self.addLink(routers[u], routers[v],
        # params1={'ip': TODO},
        # params2={'ip': TODO}
        )

    print('Building...')
    super(CustomTopo, self).build(*args, **kwargs)

  def addRouter_v6(self, name, routes):
    return self.addRouter(name, use_v4=False, use_v6=True, 
                          config=(RouterConfig, {'daemons': [(Zebra, {'static_routes': routes})]}))

def start_ping(net, n, i, rate='0.1', time='20s'):
  src = net.get(f'r{i}')
  dst = net.get(f'r{np.random.choice(n)}')

  outfile = f'out/ping_{i}.txt'
  
  src.popen(f'ping -i 0.1 -w {time} {dst.IP()} > {outfile}', shell=True)

def uniform_random_benchmark(adj_file, routing_table_file):
  topo = CustomTopo(adj_file, routing_table_file)
  net = IPNet(topo=topo, use_v4=False, allocate_IPs=False)

  net.start()
  net.pingAll()

  IPCLI(net)

  # for i in range(topo.n):
  #   start_ping(net, topo.n, i)

  net.stop()

if __name__ == '__main__':
  uniform_random_benchmark('TestCustomTopo_test_simple', 'TestCustomTopo_test_simple')