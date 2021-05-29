from figure3 import SWRingTopo, SW2DTorusTopo, SW3DHexTorusTopo, CamCubeTopo
import unittest

class TestSWRingTopo(unittest.TestCase):
  def test_small(self):
    print('======== SWRingTopo ========')
    n = 100
    G = SWRingTopo(n=n)

    self.assertEqual(G.number_of_nodes(), n, 'wrong number of nodes')
    self.assertEqual(G.size(), n * 6 // 2, 'wrong number of edges')

    for i in range(n):
      neighbors = list(G.neighbors(i))
      self.assertEqual(len(neighbors), 6, f'node {i} does not have 6 edges')
      
      if (i + 1) % n not in neighbors:
        self.fail(f'no edge between nodes {i} and {(i + 1) % n}')

class TestSW2DTorusTopo(unittest.TestCase):
  def test_small(self):
    print('======= SW2DTorusTopo ======')
    x, y = 10, 10
    n = x * y
    G = SW2DTorusTopo(dims=[x, y])

    self.assertEqual(G.number_of_nodes(), n, 'wrong number of nodes')
    self.assertEqual(G.size(), n * 6 // 2, 'wrong number of edges')

    for i in range(x):
      for j in range(y):
        index = i * y + j
        neighbors = list(G.neighbors(index))
        self.assertEqual(len(neighbors), 6, 
                         f'node ({i},{j}) does not have 6 edges')

        if i * y + (j + 1) % y not in neighbors:
          self.fail(f'no edge between nodes ({i},{j}) and ({i},{j+1})')
        if ((i + 1) % x) * y + j not in neighbors:
          self.fail(f'no edge between nodes ({i},{j}) and ({i+1},{j})')

class TestSW3DHexTorusTopo(unittest.TestCase):
  def test_small(self):
    print('===== SW3DHexTorusTopo =====')
    x, y, z = 6, 6, 6
    n = x * y * z
    G = SW3DHexTorusTopo(dims=[x, y, z])

    self.assertEqual(G.number_of_nodes(), n, 'wrong number of nodes')
    self.assertEqual(G.size(), n * 6 // 2, 'wrong number of edges')

    for i in range(x):
      for j in range(y):
        for k in range(z):
          index = i * y * z + j * z + k
          neighbors = list(G.neighbors(index))
          self.assertEqual(len(neighbors), 6,
                           f'node ({i},{j},{k}) does not have 6 edges')

          if i * y * z + j * z + (k + 1) % z not in neighbors:
            self.fail(f'no edge between nodes ({i},{j},{k}) and ({i},{j},{k+1})')
          if ((j + k) % 2 == 1) \
             and i * y * z + ((j + 1) % z) * z + k not in neighbors:
            self.fail(f'no edge between nodes ({i},{j},{k}) and ({i},{j+1},{k})')
          if ((i + 1) % y) % z * y * z + j * z + k not in neighbors:
            self.fail(f'no edge between nodes ({i},{j},{k}) and ({i+1},{j},{k})')

class TestCamCubeTopo(unittest.TestCase):
  def test_small(self):
    print('======== CamCubeTopo =======')
    x, y, z = 5, 5, 5
    n = x * y * z
    G = CamCubeTopo(dims=[x, y, z])

    self.assertEqual(G.number_of_nodes(), n, 'wrong number of nodes')
    self.assertEqual(G.size(), n * 6 // 2, 'wrong number of edges')

    for i in range(x):
      for j in range(y):
        for k in range(z):
          index = i * y * z + j * z + k
          neighbors = list(G.neighbors(index))
          self.assertEqual(len(neighbors), 6,
                           f'node ({i},{j},{k}) does not have 6 edges')

          if i * y * z + j * z + (k + 1) % z not in neighbors:
            self.fail(f'no edge between nodes ({i},{j},{k}) and ({i},{j},{k+1})')
          if i * y * z + ((j + 1) % z) * z + k not in neighbors:
            self.fail(f'no edge between nodes ({i},{j},{k}) and ({i},{j+1},{k})')
          if ((i + 1) % y) % z * y * z + j * z + k not in neighbors:
            self.fail(f'no edge between nodes ({i},{j},{k}) and ({i+1},{j},{k})')

if __name__ == '__main__':
  unittest.main()
