import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import networkx as nx
import numpy as np

# -----------------------------------------------------------------------
# Graph rendering
# -----------------------------------------------------------------------
#
# The main job here is to save snapshots of the graph at intermediate
# reduction steps so the animator can later compile them into a GIF/PDF.
# Plain networkx drawing is not used because parallel edges (MultiGraph)
# need to be curved to remain distinguishable. We draw each edge as a
# quadratic Bézier curve whose control point is offset perpendicular to
# the straight line between the two nodes.

# Renders the graph to a PNG file inside graph_snapshots/.
# Each edge gets a curvature that spreads parallel edges apart (computed
# by np.linspace(-1, 1, n_edges) so a single edge is straight).
# The 'step' argument is an integer counter used to build the filename
# (step_0001.png, step_0002.png …); None falls back to graph.png.
# 'note' is unused at runtime but kept for call-site clarity.
def draw_graph(graph, EdgeLabels=None, step=None, note=""):
    pos = nx.get_node_attributes(graph, "pos")
    if not pos or not all(isinstance(p, tuple) and len(p) == 2 for p in pos.values()):
        pos = nx.kamada_kawai_layout(graph)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.axis('off')

    nx.draw_networkx_nodes(graph, pos, node_size=0, ax=ax)

    for u, v in graph.edges():
        n_edges = graph.number_of_edges(u, v)
        keys = list(graph[u][v].keys()) if isinstance(graph, nx.MultiGraph) else [0]
        # Spread curvatures symmetrically around 0 so a single edge is straight.
        curvatures = np.linspace(-1, 1, n_edges)

        for i, key in enumerate(keys):
            rad = curvatures[i] if n_edges > 1 else 0.0
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            # Perpendicular direction for the control point offset.
            dx, dy = y2 - y1, x1 - x2
            cx = xm + rad * dx
            cy = ym + rad * dy

            # Build a quadratic Bézier path and render it as a patch.
            Path = mpath.Path
            path = Path([(x1, y1), (cx, cy), (x2, y2)],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            ax.add_patch(
                mpatches.PathPatch(path, facecolor='none', lw=1.5)
            )

            # Place the edge label at the Bézier midpoint (t=0.5).
            if EdgeLabels:
                t = 0.5
                bx = (1 - t)**2 * x1 + 2*(1 - t)*t*cx + t**2 * x2
                by = (1 - t)**2 * y1 + 2*(1 - t)*t*cy + t**2 * y2
                label = EdgeLabels.get((u, v, key), EdgeLabels.get((u, v)))
                if label is not None:
                    ax.text(bx, by, str(label),
                            fontsize=8, ha='center', va='center')

    filename = f"graph_snapshots/step_{step:04d}.png" if step is not None else "graph_snapshots/graph.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Planarity diagnostics
# -----------------------------------------------------------------------

# Draws the Kuratowski obstruction subgraph (K5 or K3,3 subdivision) returned
# by nx.check_planarity(G, counterexample=True) when the graph is non-planar.
# Saves to `out_path` (PNG) and calls plt.show() so the window pops up.
# Called from any site that detects non-planarity.
def plot_kuratowski(subgraph, out_path="non_planar_obstruction.png"):
    """Visualise and save the Kuratowski obstruction subgraph.

    Parameters
    ----------
    subgraph : nx.Graph
        The K₅ / K₃,₃ subdivision returned as a counterexample by
        ``nx.check_planarity(G, counterexample=True)``.
    out_path : str
        Destination PNG file path.  Defaults to ``non_planar_obstruction.png``
        in the current working directory.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(
        "Kuratowski obstruction subgraph\n"
        "(K5 or K3,3 subdivision - graph is non-planar)",
        fontsize=12,
    )

    pos = nx.spring_layout(subgraph, seed=0)
    nx.draw_networkx(
        subgraph,
        pos=pos,
        ax=ax,
        node_color="#e74c3c",
        node_size=500,
        font_color="white",
        font_weight="bold",
        edge_color="#2c3e50",
        width=2,
    )
    ax.text(
        0.01, 0.01,
        f"Obstruction: {subgraph.number_of_nodes()} nodes, "
        f"{subgraph.number_of_edges()} edges",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  Kuratowski subgraph saved to: {out_path}")


# -----------------------------------------------------------------------
# Layout computation
# -----------------------------------------------------------------------

# Computes node positions and stores them back into the graph's node
# attributes so that subsequent draws are consistent. Called once after
# loading/gluing a graph, before reduction begins.
#
# layout options:
#   "spring"   – force-directed (good for arbitrary graphs)
#   "kamada"   – force-directed with edge-length optimisation (usually best)
#   "circular" – nodes equally spaced on a circle
#   "planar"   – straight-line planar embedding (only for planar graphs)
#   "auto"     – tries planar first, falls back to kamada-kawai
def compute_layout(graph, layout="auto"):
    if layout == "spring":
        pos = nx.spring_layout(graph, seed=42)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "planar":
        try:
            pos = nx.planar_layout(graph)
        except nx.NetworkXException:
            pos = nx.spring_layout(graph, seed=42)
    elif layout == "auto":
        try:
            pos = nx.planar_layout(graph)
        except nx.NetworkXException:
            pos = nx.kamada_kawai_layout(graph)
    else:
        raise ValueError(f"Unknown layout type '{layout}'")

    nx.set_node_attributes(graph, pos, "pos")
    return pos
