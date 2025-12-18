from norm_calculator import reduce_one_triangle_in_graph
import networkx as nx
import copy



class SpinNetwork:
    def __init__(self, graph):
        """
        graph: a networkx Graph or MultiGraph
        """
        self.graph = copy.deepcopy(graph)
        self.coeffs = []  # list of dicts {"type":"6j", "args2": (...)} or similar

    def reduce_one_triangle(self):
        """
        Reduces one eligible triangle in self.graph.
        Returns True if a triangle was reduced, False otherwise.
        Appends one 6j-symbol to self.coeffs.
        """
        R, coeff = reduce_one_triangle_in_graph(self.graph)
        if R is None:
            return False
        self.graph = R
        self.coeffs.append(coeff)
        return True

    def reduce_all_triangles(self):
        """
        Greedily reduce all eligible triangles until none remain.
        Returns the total number of reductions applied.
        """
        count = 0
        while self.reduce_one_triangle():
            count += 1
        return count

    # -----------------------------
    # Collect f indices
    # -----------------------------
    def collect_f_indices(self):
        """
        Returns sorted list of all f indices (doubled int) in 6j-symbols,
        and a mapping f -> list of positions in self.coeffs
        """
        f_map = {}
        for idx, c in enumerate(self.coeffs):
            if c.get("type") == "6j":
                args2 = c["args2"]
                f2 = args2[5]  # f is last argument
                f_map.setdefault(f2, []).append(idx)
        f_values_sorted = sorted(f_map.keys())
        return f_values_sorted, f_map