#!/usr/bin/env python3
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
Test the reconnection and probability calculation workflow programmatically.

The test builds its own small spin network fixture (numeric labels, four
open edges) instead of reading drawn_graph.graphml, so it does not depend
on whatever graph the user last drew in the editor.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import networkx as nx
import tempfile
import shutil


def make_fixture_graph_file():
    """
    Write a small test network to a temporary GraphML file and return its path.

    Structure: two internal trivalent nodes (3, 4) joined by one internal
    edge; each also carries two open edges ending in degree-1 leaf nodes.
    All labels are spin 1, so every vertex satisfies the triangle inequality.

        1 ---- 3 ---- 4 ---- 2      (open, internal, open)
               |      |
               5      6             (open, open)
    """
    G = nx.MultiGraph()
    G.add_edge("1", "3", label=1.0)  # open edge at leaf 1
    G.add_edge("2", "4", label=1.0)  # open edge at leaf 2
    G.add_edge("3", "4", label=1.0)  # internal edge
    G.add_edge("3", "5", label=1.0)  # open edge at leaf 5
    G.add_edge("4", "6", label=1.0)  # open edge at leaf 6

    with tempfile.NamedTemporaryFile(mode="w", suffix=".graphml", delete=False) as f:
        path = f.name
    nx.write_graphml(G, path)
    return path


def load_graph(file_path):
    """Load graph from GraphML."""
    graph = nx.read_graphml(file_path, force_multigraph=True)

    for u, v, data in graph.edges(data=True):
        try:
            data["label"] = float(data["label"])
        except (ValueError, KeyError):
            pass

    return graph


def find_open_edges(graph):
    """Find all open edges (edges with at least one endpoint of degree < 3)."""
    open_edges = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        if graph.degree(u) < 3 or graph.degree(v) < 3:
            open_edges.append((u, v, key, data))
    return open_edges


def reconnect_edges(graph, edge1, edge2):
    """
    Reconnect two open edges by creating a new node and new edge.

    Returns:
        (new_graph, reconnection_data)
    """
    u1, v1, key1, data1 = edge1
    u2, v2, key2, data2 = edge2

    # Determine which endpoints to connect (degree < 3)
    endpoint1 = u1 if graph.degree(u1) < 3 else v1
    endpoint2 = u2 if graph.degree(u2) < 3 else v2

    # Create new node at midpoint (for visualization purposes)
    new_node_id = max(int(n) for n in graph.nodes()) + 1
    graph.add_node(str(new_node_id))

    # Remove old edges
    label1 = data1.get("label", 1.0)
    label2 = data2.get("label", 1.0)

    graph.remove_edge(u1, v1, key1)
    graph.remove_edge(u2, v2, key2)

    # Add edges from old endpoints to new node
    graph.add_edge(endpoint1, str(new_node_id), label=label1)
    graph.add_edge(endpoint2, str(new_node_id), label=label2)

    # Determine other endpoints
    other1 = v1 if endpoint1 == u1 else u1
    other2 = v2 if endpoint2 == u2 else u2

    # Create new open edge with combined label (using triangle inequality rule)
    new_label = abs(label1 - label2)  # Minimum possible value
    graph.add_edge(other1, other2, label=new_label)

    # Record reconnection data
    reconnection = {
        "old_edges": [
            {"nodes": (u1, v1), "label": label1},
            {"nodes": (u2, v2), "label": label2},
        ],
        "new_edge": {"nodes": (other1, other2), "label": new_label},
        "reconnection_node": str(new_node_id),
    }

    return graph, reconnection


def test_workflow():
    """Test the complete reconnection and probability workflow."""
    print("\n" + "=" * 70)
    print("TESTING RECONNECTION & PROBABILITY WORKFLOW")
    print("=" * 70)

    # Build the self-contained fixture graph (see make_fixture_graph_file)
    original_file = make_fixture_graph_file()

    print(f"\n[1] Loading original graph: {original_file}")
    graph = load_graph(original_file)
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")

    # Find open edges
    print("\n[2] Finding open edges...")
    open_edges = find_open_edges(graph)
    print(f"  Found {len(open_edges)} open edges:")
    for i, (u, v, key, data) in enumerate(open_edges[:5], 1):
        label = data.get("label", "?")
        print(f"    {i}. ({u}, {v}) with label={label}")

    assert len(open_edges) >= 2, "Fixture must have at least 2 open edges"

    # Perform reconnection.  Pick two open edges attached to DIFFERENT
    # internal nodes, so the reconnection creates a proper new edge between
    # them (pairing two open edges of the same node would give a self-loop).
    # GraphML loading does not preserve edge insertion order, so we cannot
    # simply take open_edges[0] and open_edges[1].
    def internal_endpoint(g, edge):
        u, v, key, data = edge
        return u if g.degree(u) >= 3 else v

    print("\n[3] Reconnecting two open edges at different nodes...")
    edge1 = open_edges[0]
    edge2 = next(
        e
        for e in open_edges[1:]
        if internal_endpoint(graph, e) != internal_endpoint(graph, edge1)
    )

    reconnected_graph, reconnection_data = reconnect_edges(graph.copy(), edge1, edge2)

    print(f"  Created reconnection node: {reconnection_data['reconnection_node']}")
    print(
        f"  Old edge 1: {reconnection_data['old_edges'][0]['nodes']} "
        f"(label={reconnection_data['old_edges'][0]['label']})"
    )
    print(
        f"  Old edge 2: {reconnection_data['old_edges'][1]['nodes']} "
        f"(label={reconnection_data['old_edges'][1]['label']})"
    )
    print(
        f"  New edge: {reconnection_data['new_edge']['nodes']} "
        f"(label={reconnection_data['new_edge']['label']})"
    )

    # Save reconnected graph
    print("\n[4] Saving reconnected graph...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".graphml", delete=False) as f:
        reconnected_file = f.name

    nx.write_graphml(reconnected_graph, reconnected_file)
    print(f"  Saved to: {reconnected_file}")

    # Save reconnection data
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_reconnections.json", delete=False
    ) as f:
        recon_data_file = f.name

    with open(recon_data_file, "w") as f:
        json.dump([reconnection_data], f, indent=2)
    print(f"  Reconnection data: {recon_data_file}")

    # Compute probability
    print("\n[5] Computing probability...")
    try:
        from scripts.compute_probability import (
            compute_norm,
            compute_delta_product,
            compute_theta_product,
        )

        # Compute norms
        print("  Computing original norm...")
        norm1 = compute_norm(original_file, quiet=True)
        print(f"    ||G₁|| = {norm1}")

        print("  Computing reconnected norm...")
        norm2 = compute_norm(reconnected_file, quiet=True)
        print(f"    ||G₂|| = {norm2}")

        assert norm1 != 0, "Original norm must be non-zero"
        assert norm2 != 0, "Reconnected norm must be non-zero"

        norm_ratio = norm2 / norm1 if norm1 != 0 else 0
        print(f"    ||G₂||/||G₁|| = {norm_ratio}")

        # Extract labels
        new_labels = [reconnection_data["new_edge"]["label"]]
        theta_triplets = [
            (
                reconnection_data["old_edges"][0]["label"],
                reconnection_data["old_edges"][1]["label"],
                reconnection_data["new_edge"]["label"],
            )
        ]

        # Compute Delta and Theta
        print("\n  Computing Δ product...")
        delta_product = compute_delta_product(new_labels)
        print(f"    Δ({new_labels[0]}) = {delta_product}")

        print("\n  Computing Θ product...")
        theta_product = compute_theta_product(theta_triplets)
        a, b, c = theta_triplets[0]
        print(f"    Θ({a}, {b}, {c}) = {theta_product}")

        # Compute probability
        if theta_product == 0:
            print("\n  ⚠ Warning: Theta product is zero!")
            probability = 0
        else:
            probability = abs((delta_product / theta_product) * norm_ratio)

        print(f"\n  {'★'*70}")
        print(f"  PROBABILITY: p = {probability:.15e}")
        print(f"  {'★'*70}")

        # A physical probability must be finite and non-negative
        import math

        assert probability >= 0
        assert not math.isnan(probability)
        assert not math.isinf(probability)

    finally:
        # Cleanup temp files
        print("\n[6] Cleaning up temporary files...")
        try:
            os.unlink(original_file)
            os.unlink(reconnected_file)
            os.unlink(recon_data_file)
            print("  ✓ Cleanup complete")
        except OSError:
            pass

    print("\n" + "=" * 70)
    print("TEST COMPLETE: Workflow successful!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        test_workflow()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
