#     SPDX-License-Identifier: GPL-3.0-or-later
#     Copyright (C) 2026, N. Cohen, University of Vienna & IQOQI Vienna

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Tree visualizer for the SpinNetwork genealogy.

Provides TreeVisualizer.display_tree(), which renders a simple 2-D plot of
the ancestry/descendancy relationships starting from a given root (or any
node in the tree).  Useful for debugging and tracking the evolution of spin
network states across multiple transitions.

Usage
-----
    from src.api import load_network, TreeVisualizer

    n1 = load_network("drawn_graph.graphml")
    n2 = n1.transition_to()
    n3 = n1.transition_to()   # second child of n1
    n4 = n2.transition_to()

    TreeVisualizer.display_tree(n1)   # shows n1 → {n2, n3}, n2 → n4
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from src.api import SpinNetwork


class TreeVisualizer:
    """
    Lightweight utility for visualising the SpinNetwork genealogy tree.

    All methods are class methods; no instance needs to be created.
    """

    @classmethod
    def display_tree(
        cls,
        root: "SpinNetwork",
        figsize: tuple = (8, 5),
        title: str = "SpinNetwork Genealogy",
    ) -> None:
        """
        Render the genealogy tree rooted at the given SpinNetwork.

        Performs a BFS traversal from root downward, builds a directed
        NetworkX graph of parent–child relationships, and draws it with
        matplotlib.  Node labels show the SpinNetwork id; edge labels show
        how many open ends each transition produced.  Transitions have no
        cheap cached probability (see calculate_probability() in
        src/probability.py) — compute it explicitly if you need the number.

        Parameters
        ----------
        root : SpinNetwork
            The node from which to start the traversal.  Typically the
            oldest ancestor (no parent_transition), but any node works.
        figsize : tuple, optional
            Matplotlib figure size (width, height) in inches.
        title : str, optional
            Figure window/title string.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        # --- BFS to collect all nodes and edges --------------------------
        tree_graph: nx.DiGraph = nx.DiGraph()
        queue: deque["SpinNetwork"] = deque([root])
        visited: set = set()

        node_labels: Dict[int, str] = {}  # id(sn) → display label
        edge_labels: Dict[tuple, str] = {}  # (id_parent, id_child) → P=…

        while queue:
            sn = queue.popleft()
            sn_key = id(sn)
            if sn_key in visited:
                continue
            visited.add(sn_key)

            n_nodes = sn._graph._nx_graph.number_of_nodes()
            n_edges = sn._graph._nx_graph.number_of_edges()
            node_labels[sn_key] = f"{sn._id}\n({n_nodes}v, {n_edges}e)"
            tree_graph.add_node(sn_key)

            for t in sn.children:
                child = t.child
                if child is None:
                    continue
                child_key = id(child)
                tree_graph.add_edge(sn_key, child_key)
                edge_labels[(sn_key, child_key)] = (
                    f"+{len(t.produced_open_ends)} open ends"
                )
                queue.append(child)

        if tree_graph.number_of_nodes() == 0:
            print("TreeVisualizer: tree contains no nodes.")
            return

        # --- Layout ------------------------------------------------------
        # Use a top-down hierarchical layout if available (graphviz), else
        # fall back to spring layout.
        try:
            pos = nx.nx_agraph.graphviz_layout(tree_graph, prog="dot")
        except Exception:
            pos = nx.spring_layout(tree_graph, seed=42)

        # --- Draw --------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)

        nx.draw(
            tree_graph,
            pos=pos,
            ax=ax,
            labels=node_labels,
            with_labels=True,
            node_size=1800,
            node_color="#aec6cf",
            font_size=7,
            arrows=True,
            arrowsize=15,
        )
        nx.draw_networkx_edge_labels(
            tree_graph,
            pos=pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=7,
        )

        ax.axis("off")
        plt.tight_layout()
        plt.show()

    @classmethod
    def ascii_tree(cls, root: "SpinNetwork", indent: int = 0) -> str:
        """
        Return a simple ASCII representation of the genealogy tree.

        Useful for quick inspection in non-graphical environments.

        Parameters
        ----------
        root : SpinNetwork
            The starting node.
        indent : int
            Internal recursion counter; leave at 0 when calling externally.

        Returns
        -------
        str
            Multi-line string with tree structure.

        Example
        -------
            print(TreeVisualizer.ascii_tree(n1))
            # SpinNetwork [id=a1b2c3d4, depth=0]
            # └── Transition (+1 open ends)
            #     └── SpinNetwork [id=e5f6a7b8, depth=1]

        Note: transitions have no cheap cached probability; use
        calculate_probability(parent, child) (src/probability.py) to compute
        it explicitly.
        """
        prefix = "    " * indent
        connector = "└── " if indent > 0 else ""
        lines: List[str] = [
            f"{prefix}{connector}SpinNetwork "
            f"[id={root._id}, depth={root._genealogy_depth()}]"
        ]
        for t in root.children:
            lines.append(
                f"{prefix}    └── Transition (+{len(t.produced_open_ends)} open ends)"
            )
            if t.child is not None:
                lines.append(cls.ascii_tree(t.child, indent + 2))
        return "\n".join(lines)
