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

class CustomTopo(IPTopo):
  def __init__(self, adj_file, routing_table_file):
    IPTopo.__init__(self)
    
    self.adj = {}
    self.ip_addresses = {}
    self.routing_table = {}

    # parse adjacency list
    print('Parsing adjacency list...')
    with open(adj_file, 'r') as f:
      for line in f.readlines():
        tokens = line.split(' ')

        # ignore commented and blank lines
        if tokens[0] == '' or tokens[0] == '#':
          continue

        # update adjacency list
        v = int(tokens[0])
        neighbors = [int(tokens[i]) for i in range(1, len(tokens))]
        self.adj[v] = neighbors

        # generate ip address
        self.ip_addresses[v] = self.node_to_ip(v)

    # parse routing table
    print('Parsing routing table...')
    with open(routing_table_file, 'r') as f:
      # TODO
      pass

    # calculate number of nodes
    self.n = max(self.adj.keys())

  def node_to_ip(self, v):
    """Converts a node id to an IPv6 address."""
    return ''

  def build(self, *args, **kwargs):
    routers = {}

    # build routers
    print('Adding routers...')
    for u, tmp in self.routing_table.items():
      routes = []
      for v, next_hop in tmp:
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

  for i in range(topo.n):
    start_ping(net, topo.n, i)

  net.stop()

if __name__ == '__main__':
  uniform_random_benchmark('todo', 'todo')