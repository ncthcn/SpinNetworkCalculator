#!/usr/bin/env python3
"""
Test the reconnection and probability calculation workflow programmatically.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import networkx as nx
import tempfile
import shutil

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
    label1 = data1.get('label', 1.0)
    label2 = data2.get('label', 1.0)

    graph.remove_edge(u1, v1, key1)
    graph.remove_edge(u2, v2, key2)

    # Add edges from old endpoints to new node
    graph.add_edge(endpoint1, str(new_node_id), label=label1)
    graph.add_edge(endpoint2, str(new_node_id), label=label2)

    # Determine other endpoints
    other1 = v1 if endpoint1 == u1 else u1
    other2 = v2 if endpoint2 == u2 else v2

    # Create new open edge with combined label (using triangle inequality rule)
    new_label = abs(label1 - label2)  # Minimum possible value
    graph.add_edge(other1, other2, label=new_label)

    # Record reconnection data
    reconnection = {
        'old_edges': [
            {'nodes': (u1, v1), 'label': label1},
            {'nodes': (u2, v2), 'label': label2}
        ],
        'new_edge': {
            'nodes': (other1, other2),
            'label': new_label
        },
        'reconnection_node': str(new_node_id)
    }

    return graph, reconnection


def test_workflow():
    """Test the complete reconnection and probability workflow."""
    print("\n" + "="*70)
    print("TESTING RECONNECTION & PROBABILITY WORKFLOW")
    print("="*70)

    # Load original graph
    original_file = "drawn_graph.graphml"
    if not os.path.exists(original_file):
        print(f"Error: {original_file} not found")
        return False

    print(f"\n[1] Loading original graph: {original_file}")
    graph = load_graph(original_file)
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")

    # Find open edges
    print("\n[2] Finding open edges...")
    open_edges = find_open_edges(graph)
    print(f"  Found {len(open_edges)} open edges:")
    for i, (u, v, key, data) in enumerate(open_edges[:5], 1):
        label = data.get('label', '?')
        print(f"    {i}. ({u}, {v}) with label={label}")

    if len(open_edges) < 2:
        print("  Error: Need at least 2 open edges for reconnection")
        return False

    # Perform reconnection
    print("\n[3] Reconnecting first two open edges...")
    edge1 = open_edges[0]
    edge2 = open_edges[1]

    reconnected_graph, reconnection_data = reconnect_edges(
        graph.copy(), edge1, edge2
    )

    print(f"  Created reconnection node: {reconnection_data['reconnection_node']}")
    print(f"  Old edge 1: {reconnection_data['old_edges'][0]['nodes']} "
          f"(label={reconnection_data['old_edges'][0]['label']})")
    print(f"  Old edge 2: {reconnection_data['old_edges'][1]['nodes']} "
          f"(label={reconnection_data['old_edges'][1]['label']})")
    print(f"  New edge: {reconnection_data['new_edge']['nodes']} "
          f"(label={reconnection_data['new_edge']['label']})")

    # Save reconnected graph
    print("\n[4] Saving reconnected graph...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
        reconnected_file = f.name

    nx.write_graphml(reconnected_graph, reconnected_file)
    print(f"  Saved to: {reconnected_file}")

    # Save reconnection data
    with tempfile.NamedTemporaryFile(mode='w', suffix='_reconnections.json', delete=False) as f:
        recon_data_file = f.name

    with open(recon_data_file, 'w') as f:
        json.dump([reconnection_data], f, indent=2)
    print(f"  Reconnection data: {recon_data_file}")

    # Compute probability
    print("\n[5] Computing probability...")
    try:
        from scripts.compute_probability import compute_norm, compute_delta_product, compute_theta_product

        # Compute norms
        print("  Computing original norm...")
        norm1 = compute_norm(original_file, quiet=True)
        print(f"    ||G₁|| = {norm1}")

        print("  Computing reconnected norm...")
        norm2 = compute_norm(reconnected_file, quiet=True)
        print(f"    ||G₂|| = {norm2}")

        norm_ratio = norm2 / norm1 if norm1 != 0 else 0
        print(f"    ||G₂||/||G₁|| = {norm_ratio}")

        # Extract labels
        new_labels = [reconnection_data['new_edge']['label']]
        theta_triplets = [
            (
                reconnection_data['old_edges'][0]['label'],
                reconnection_data['old_edges'][1]['label'],
                reconnection_data['new_edge']['label']
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

    except Exception as e:
        print(f"  Error computing probability: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup temp files
        print("\n[6] Cleaning up temporary files...")
        try:
            os.unlink(reconnected_file)
            os.unlink(recon_data_file)
            print("  ✓ Cleanup complete")
        except:
            pass

    print("\n" + "="*70)
    print("TEST COMPLETE: Workflow successful!")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    try:
        success = test_workflow()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
