import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import networkx as nx
import numpy as np

# -------------------------------
# --- Draw graph (curved edges)
# -------------------------------

# def draw_graph(graph, EdgeLabels=None):
#     pos = nx.get_node_attributes(graph, "pos")
#     if not pos or not all(isinstance(p, tuple) and len(p) == 2 for p in pos.values()):
#         # print("Invalid or missing positions; using spring layout.")
#         # pos = nx.spring_layout(graph)
#         print("Invalid or missing positions; using kamada kawai layout.")
#         pos = nx.kamada_kawai_layout(graph)

#     fig, ax = plt.subplots()
#     ax.set_aspect('equal')
#     ax.axis('off')

#     # Draw nodes and labels
#     nx.draw_networkx_nodes(graph, pos, node_size=0, ax=ax)
#     # nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=200, ax=ax)
#     # nx.draw_networkx_labels(graph, pos, font_size=5, ax=ax)

#     # Draw edges manually (so we control curvature)
#     for u, v in graph.edges():
#         n_edges = graph.number_of_edges(u, v)
#         keys = list(graph[u][v].keys()) if isinstance(graph, nx.MultiGraph) else [0]

#         curvatures = np.linspace(-1, 1, n_edges)
#         for i, key in enumerate(keys):
#             rad = curvatures[i] if n_edges > 1 else 0.0

#             # Compute positions
#             x1, y1 = pos[u]
#             x2, y2 = pos[v]
#             xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
#             dx, dy = y2 - y1, x1 - x2
#             control_x = xm + rad * dx
#             control_y = ym + rad * dy

#             # Define a quadratic Bézier curve path
#             Path = mpath.Path
#             verts = [(x1, y1), (control_x, control_y), (x2, y2)]
#             codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
#             path = mpath.Path(verts, codes)
#             patch = mpatches.PathPatch(path, facecolor='none', lw=1.5, edgecolor='black')
#             ax.add_patch(patch)

#             # Compute midpoint on the curve for label
#             t = 0.5
#             bx = (1 - t)**2 * x1 + 2 * (1 - t) * t * control_x + t**2 * x2
#             by = (1 - t)**2 * y1 + 2 * (1 - t) * t * control_y + t**2 * y2

#             # Add label if provided
#             if EdgeLabels:
#                 label_key = (u, v, key) if (u, v, key) in EdgeLabels else (u, v)
#                 label = EdgeLabels.get(label_key)
#                 if label is not None:
#                     ax.text(bx, by, str(label), color='black', fontsize=8,
#                             ha='center', va='center', backgroundcolor='white')

#     plt.show()

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
        curvatures = np.linspace(-1, 1, n_edges)

        for i, key in enumerate(keys):
            rad = curvatures[i] if n_edges > 1 else 0.0
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = y2 - y1, x1 - x2
            cx = xm + rad * dx
            cy = ym + rad * dy

            Path = mpath.Path
            path = Path([(x1, y1), (cx, cy), (x2, y2)],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            ax.add_patch(
                mpatches.PathPatch(path, facecolor='none', lw=1.5)
            )

            if EdgeLabels:
                t = 0.5
                bx = (1 - t)**2 * x1 + 2*(1 - t)*t*cx + t**2 * x2
                by = (1 - t)**2 * y1 + 2*(1 - t)*t*cy + t**2 * y2
                label = EdgeLabels.get((u, v, key), EdgeLabels.get((u, v)))
                if label is not None:
                    ax.text(bx, by, str(label),
                            fontsize=8, ha='center', va='center')

    # ---- SAVE TO DISK ----
    filename = f"graph_snapshots/step_{step:04d}.png" if step is not None else "graph_snapshots/graph.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)



# -------------------------------
# --- Compute graph layout
# -------------------------------

def compute_layout(graph, layout="auto"):
    """
    Compute or update the node positions for the graph.
    If layout='auto', it will use the most balanced layout.
    """
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
        # Try planar; fallback to spring
        try:
            pos = nx.planar_layout(graph)
        except nx.NetworkXException:
            pos = nx.kamada_kawai_layout(graph)
    else:
        raise ValueError(f"Unknown layout type '{layout}'")

    # Save positions in graph for consistency
    nx.set_node_attributes(graph, pos, "pos")
    return pos
