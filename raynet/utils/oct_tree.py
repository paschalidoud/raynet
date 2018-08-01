
import numpy as np

from geometry import point_in_aabbox, ray_aabbox_intersection
from fast_utils import fast_ray_triangles_intersection

class OctTree(object):
    box_factors = np.array([
        [0, 0, 0, -0.5, -0.5, -0.5],
        [0.5, 0, 0, 0, -0.5, -0.5],
        [0, 0.5, 0, -0.5, 0, -0.5],
        [0.5, 0.5, 0, 0, 0, -0.5],
        [0, 0, 0.5, -0.5, -0.5, 0],
        [0.5, 0, 0.5, 0, -0.5, 0],
        [0, 0.5, 0.5, -0.5, 0, 0],
        [0.5, 0.5, 0.5, 0, 0, 0],
    ], dtype=np.float32)

    def __init__(self, triangles, depth=5):
        self.triangles = triangles
        self.depth = depth
        self._n_inner = sum(8**i for i in range(depth))
        self._n_all = sum(8**i for i in range(depth+1))
        self._tree = np.zeros((
            self._n_all,
            6
        ))
        self._contents = [[] for _ in range(self._n_all)]
        self._tree[0, :3] = triangles.min(axis=0).reshape(-1, 3).min(axis=0)
        self._tree[0, 3:] = triangles.max(axis=0).reshape(-1, 3).max(axis=0)
        self._build_tree(0)
        self._fill(triangles)

    def _children(self, i):
        return [8*i+1+j for j in range(8)]

    def _parent(self, i):
        return (i-1) // 8

    def _is_leaf(self, i):
        return self._leaf_index(i) >= 0

    def _leaf_index(self, i):
        return i - self._n_inner

    def _build_tree(self, cur):
        d = self._tree[cur, 3:] - self._tree[cur, :3]
        d = np.array([d, d]).ravel()
        for i, ch in enumerate(self._children(cur)):
            if ch < len(self._tree):
                self._tree[ch] = self._tree[cur] + d * self.box_factors[i]
                self._build_tree(ch)

    def _insert_triangle(self, triangle, node):
        if self._is_leaf(node):
            return node

        for i, ch in enumerate(self._children(node)):
            if all(
                point_in_aabbox(
                    triangle[3*j:3*j+3],
                    self._tree[ch, :3],
                    self._tree[ch, 3:]
                ) for j in range(3)
            ):
                return self._insert_triangle(triangle, ch)

        return node

    def _fill(self, triangles):
        for i in range(len(triangles)):
            node = self._insert_triangle(triangles[i], 0)
            self._contents[node].append(i)

    def ray_intersections(self, origin, destination):
        idxs = np.zeros((len(self.triangles),), dtype=np.bool)
        self._add_triangles(origin, destination, 0, idxs)

        return fast_ray_triangles_intersection(
            origin[:3, 0], destination[:3, 0],
            self.triangles[idxs, :3],
            self.triangles[idxs, 3:6],
            self.triangles[idxs, 6:]
        )

    def _add_triangles(self, origin, destination, node, idxs):
        for i in self._contents[node]:
            idxs[i] = True
        if self._is_leaf(node):
            return
        for ch in self._children(node):
            if ray_aabbox_intersection(
                origin, destination,
                self._tree[node, :3], self._tree[node, 3:]
            ) != (None, None):
                self._add_triangles(origin, destination, ch, idxs)
