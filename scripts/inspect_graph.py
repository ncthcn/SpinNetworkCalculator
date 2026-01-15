#!/usr/bin/env python3
"""
Inspect Graph - Show open edges and help identify flagging information

This tool helps you identify which edges are "open" (connected to vertices with degree < 3)
and shows the information you need for the CLI comparison workflow.

Usage:
    python inspect_graph.py graph.graphml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import networkx as nx


def load_graph(filepath):
    """Load a graph from a GraphML file."""
    graph = nx.read_graphml(filepath, force_multigraph=True)

    # Convert edge labels
    for u, v, data in graph.edges(data=True):
        try:
            data["label"] = float(data["label"])
        except ValueError:
            pass

    return graph


def get_open_edges(graph):
    """Get all open edges (edges where at least one endpoint has degree < 3)."""
    open_edges = []
    for n1, n2, key, data in graph.edges(keys=True, data=True):
        if graph.degree(n1) < 3 or graph.degree(n2) < 3:
            label = data.get('label', '?')
            open_edges.append((n1, n2, key, label))
    return open_edges


def get_vertex_info(graph, vertex):
    """Get information about edges connected to a vertex."""
    incident_edges = []
    for neighbor, edge_dict in graph[vertex].items():
        for key, edge_data in edge_dict.items():
            label = edge_data.get('label', '?')
            incident_edges.append({
                'neighbor': neighbor,
                'key': key,
                'label': label,
                'nodes': (vertex, neighbor)
            })
    return incident_edges


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a spin network graph to find open edges",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("graph_file", help="Input .graphml file")
    parser.add_argument(
        "--vertex",
        type=str,
        help="Show detailed info for a specific vertex"
    )

    args = parser.parse_args()

    if not os.path.exists(args.graph_file):
        print(f"Error: File '{args.graph_file}' not found.")
        sys.exit(1)

    print("\n" + "="*70)
    print(f"GRAPH INSPECTION: {args.graph_file}")
    print("="*70)

    # Load graph
    graph = load_graph(args.graph_file)

    # Basic statistics
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {len(graph.nodes())}")
    print(f"  Edges: {len(graph.edges())}")

    # Degree distribution
    degrees = dict(graph.degree())
    print(f"\nDegree Distribution:")
    for d in [0, 1, 2, 3]:
        nodes_with_deg_d = [n for n, deg in degrees.items() if deg == d]
        if nodes_with_deg_d:
            print(f"  Degree {d}: {len(nodes_with_deg_d)} nodes {nodes_with_deg_d}")

    # Open edges
    open_edges = get_open_edges(graph)
    print(f"\nOpen Edges (connected to vertices with degree < 3):")
    print(f"  Count: {len(open_edges)}")

    if open_edges:
        print(f"\n  {'Node1':<8} {'Node2':<8} {'Key':<6} {'Label':<10} {'Deg1':<6} {'Deg2':<6}")
        print(f"  {'-'*54}")
        for n1, n2, key, label in open_edges:
            deg1 = graph.degree(n1)
            deg2 = graph.degree(n2)
            print(f"  {str(n1):<8} {str(n2):<8} {str(key):<6} {str(label):<10} {deg1:<6} {deg2:<6}")

    # Vertex detail
    if args.vertex:
        print(f"\n" + "="*70)
        print(f"VERTEX DETAIL: {args.vertex}")
        print("="*70)

        if args.vertex not in graph.nodes():
            print(f"  ⚠ Vertex '{args.vertex}' not found in graph")
        else:
            degree = graph.degree(args.vertex)
            print(f"  Degree: {degree}")

            edges = get_vertex_info(graph, args.vertex)
            print(f"  Connected Edges: {len(edges)}")

            if edges:
                print(f"\n  {'Neighbor':<10} {'Key':<6} {'Label':<10}")
                print(f"  {'-'*26}")
                for e in edges:
                    print(f"  {str(e['neighbor']):<10} {str(e['key']):<6} {str(e['label']):<10}")

                # CLI command hint
                if len(edges) == 3:
                    labels = [e['label'] for e in edges]
                    print(f"\n  For CLI comparison, if you flag one of these edges:")
                    for i, e in enumerate(edges):
                        other_labels = [edges[j]['label'] for j in range(len(edges)) if j != i]
                        print(f"\n  Edge {i+1}: {args.vertex} -- [{e['label']}] -- {e['neighbor']}")
                        print(f"    --flagged-edge {args.vertex} {e['neighbor']} \\")
                        print(f"    --flagged-label {e['label']} \\")
                        print(f"    --flagged-vertex {args.vertex} \\")
                        print(f"    --other-labels {' '.join(str(l) for l in other_labels)}")

    # Usage hints
    print(f"\n" + "="*70)
    print("USAGE HINTS")
    print("="*70)
    print("\nTo inspect a specific vertex:")
    print(f"  python inspect_graph.py {args.graph_file} --vertex <vertex_id>")

    if open_edges:
        example_edge = open_edges[0]
        n1, n2, key, label = example_edge
        deg1 = graph.degree(n1)
        deg2 = graph.degree(n2)

        # Pick the vertex with degree < 3
        if deg1 < 3:
            vertex = n1
        else:
            vertex = n2

        print(f"\nExample: To inspect vertex {vertex}:")
        print(f"  python inspect_graph.py {args.graph_file} --vertex {vertex}")

    print(f"\nTo compare graphs with CLI (no GUI):")
    print(f"  python compare_graphs_cli.py original.graphml modified.graphml \\")
    print(f"      --flagged-edge <node1> <node2> --flagged-label <label> \\")
    print(f"      --flagged-vertex <vertex> --other-labels <label1> <label2>")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
