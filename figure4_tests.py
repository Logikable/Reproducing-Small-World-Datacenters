from figure4 import *
import unittest

class TestSaveLoadTopo(unittest.TestCase):
  def test_simple(self):
    print('======== SaveLoadTopo ========')
    name = 'TestSaveLoadTopo_test_simple'
    save_G = SWRingTopo(n=100, degree=3)
    save_topo(save_G, name)
    load_G = load_topo(name)

    self.maxDiff = None

    save_G_edges = list(save_G.edges())
    load_G_edges = list(load_G.edges())

    # Order all of the edges the same way (smaller_node, bigger_node).
    save_G_edges = [(min(n1, n2), max(n1, n2)) for n1, n2 in save_G_edges]
    load_G_edges = [(min(n1, n2), max(n1, n2)) for n1, n2 in load_G_edges]

    self.assertCountEqual(list(save_G.nodes()), list(load_G.nodes()))
    self.assertCountEqual(save_G_edges, load_G_edges)

class TestSaveLoadRoutingTable(unittest.TestCase):
  def test_simple(self):
    print('======== SaveLoadRoutingTable ========')
    name = 'TestSaveLoadRoutingTable_test_simple'
    G = SWRingTopo(n=100, degree=3)
    save_rt = greedy_routing_table(G, 
                                   SWRing_manhattan_distance, 
                                   SWRing_manhattan_next_hop)
    save_routing_table(save_rt, name)
    load_rt = load_routing_table(name)

    self.assertEqual(save_rt, load_rt)

class TestSW3DHexTorus(unittest.TestCase):
    def x_y_z_to_node(self, x, y, z, dims):
        dim_x, dim_y, dim_z = dims
        return x * dim_y * dim_z + y * dim_z + z

    def test_manhattan_distance_y_z_only(self):
        print('======== SW3DHexTorus_manhattan_distance_y_z_only ========')
        dims = [100, 200, 300]
        n = dims[0] * dims[1] * dims[2]

        frm_x, frm_y, frm_z = 0, 0, 2
        to_x, to_y, to_z = 0, 3, 5
        frm = self.x_y_z_to_node(frm_x, frm_y, frm_z, dims)
        to = self.x_y_z_to_node(to_x, to_y, to_z, dims)
        dist = SW3DHexTorus_manhattan_distance(n, frm, to, dims)
        self.assertEqual(dist, 6)

        frm_x, frm_y, frm_z = 0, 0, 2
        to_x, to_y, to_z = 0, 3, 4
        frm = self.x_y_z_to_node(frm_x, frm_y, frm_z, dims)
        to = self.x_y_z_to_node(to_x, to_y, to_z, dims)
        dist = SW3DHexTorus_manhattan_distance(n, frm, to, dims)
        self.assertEqual(dist, 7)

        frm_x, frm_y, frm_z = 0, 0, 2
        to_x, to_y, to_z = 0, 3, 3
        frm = self.x_y_z_to_node(frm_x, frm_y, frm_z, dims)
        to = self.x_y_z_to_node(to_x, to_y, to_z, dims)
        dist = SW3DHexTorus_manhattan_distance(n, frm, to, dims)
        self.assertEqual(dist, 6)

        frm_x, frm_y, frm_z = 0, 0, 2
        to_x, to_y, to_z = 0, 3, 2
        frm = self.x_y_z_to_node(frm_x, frm_y, frm_z, dims)
        to = self.x_y_z_to_node(to_x, to_y, to_z, dims)
        dist = SW3DHexTorus_manhattan_distance(n, frm, to, dims)
        self.assertEqual(dist, 7)

        frm_x, frm_y, frm_z = 0, 0, 2
        to_x, to_y, to_z = 0, 3, 1
        frm = self.x_y_z_to_node(frm_x, frm_y, frm_z, dims)
        to = self.x_y_z_to_node(to_x, to_y, to_z, dims)
        dist = SW3DHexTorus_manhattan_distance(n, frm, to, dims)
        self.assertEqual(dist, 6)

        frm_x, frm_y, frm_z = 0, 0, 3
        to_x, to_y, to_z = 0, 3, 0
        frm = self.x_y_z_to_node(frm_x, frm_y, frm_z, dims)
        to = self.x_y_z_to_node(to_x, to_y, to_z, dims)
        dist = SW3DHexTorus_manhattan_distance(n, frm, to, dims)
        self.assertEqual(dist, 6)

        frm_x, frm_y, frm_z = 0, 0, 3
        to_x, to_y, to_z = 0, 3, 3
        frm = self.x_y_z_to_node(frm_x, frm_y, frm_z, dims)
        to = self.x_y_z_to_node(to_x, to_y, to_z, dims)
        dist = SW3DHexTorus_manhattan_distance(n, frm, to, dims)
        self.assertEqual(dist, 5)

    def test_manhattan_next_hop_y_z_only(self):
        print('======== SW3DHexTorus_manhattan_next_hop_y_z_only ========')
        dims = [100, 200, 300]
        n = dims[0] * dims[1] * dims[2]

        frm_x, frm_y, frm_z = 0, 0, 2
        to_x, to_y, to_z = 0, 3, 5
        correct_x, correct_y, correct_z = 0, 0, 3
        frm = self.x_y_z_to_node(frm_x, frm_y, frm_z, dims)
        to = self.x_y_z_to_node(to_x, to_y, to_z, dims)
        correct = self.x_y_z_to_node(correct_x, correct_y, correct_z, dims)
        next_hop = SW3DHexTorus_manhattan_next_hop(n, frm, to, dims)
        self.assertEqual(next_hop, correct)

        frm_x, frm_y, frm_z = 0, 0, 4
        to_x, to_y, to_z = 0, 3, 1
        correct_x, correct_y, correct_z = 0, 0, 3
        frm = self.x_y_z_to_node(frm_x, frm_y, frm_z, dims)
        to = self.x_y_z_to_node(to_x, to_y, to_z, dims)
        correct = self.x_y_z_to_node(correct_x, correct_y, correct_z, dims)
        next_hop = SW3DHexTorus_manhattan_next_hop(n, frm, to, dims)
        self.assertEqual(next_hop, correct)

        frm_x, frm_y, frm_z = 0, 3, 3
        to_x, to_y, to_z = 0, 0, 0
        correct_x, correct_y, correct_z = 0, 2, 3
        frm = self.x_y_z_to_node(frm_x, frm_y, frm_z, dims)
        to = self.x_y_z_to_node(to_x, to_y, to_z, dims)
        correct = self.x_y_z_to_node(correct_x, correct_y, correct_z, dims)
        next_hop = SW3DHexTorus_manhattan_next_hop(n, frm, to, dims)
        self.assertEqual(next_hop, correct)

if __name__ == '__main__':
  unittest.main()
