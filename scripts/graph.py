import tkinter as tk
import tkinter.simpledialog
import networkx as nx
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import vertex_satisfies_triangular_conditions

class GraphEditor:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master, width=800, height=600, bg="white")
        self.canvas.pack()

        self.graph = nx.MultiGraph()  # Use MultiGraph to allow parallel edges
        self.nodes = {}  # Stores node IDs and positions
        self.selected_node = None  # Keeps track of the first node selected for edge creation

        # Bind left click for both adding nodes and selecting nodes for edges
        self.canvas.bind("<Button-1>", self.handle_click)

        # Add a slider to control edge curvature
        self.curvature_slider = tk.Scale(
            self.master, from_=0, to=300, orient=tk.HORIZONTAL, label="Edge Curvature"
        )
        self.curvature_slider.set(150)  # Default curvature multiplier
        self.curvature_slider.pack(side=tk.BOTTOM, pady=10)

    def handle_click(self, event):
        """
        Handles left-click events on the canvas.
        If a node is clicked, it's used to add an edge. If the click is outside a node, a new node is added.
        """
        x, y = event.x, event.y

        # Check if the click is near an existing node
        for node_id, (node_x, node_y) in self.nodes.items():
            if (x - node_x)**2 + (y - node_y)**2 <= 1000:  # Click within proximity of 100 pixels
                if self.selected_node is None:
                    # Select the first node for the edge
                    self.selected_node = node_id
                    self.highlight_node(node_id)
                    print(f"Node {node_id} selected as the starting node for an edge.")
                else:
                    # Attempt to add an edge between the selected node and the current node
                    self.attempt_add_edge(self.selected_node, node_id)
                    self.selected_node = None  # Reset the selected node after attempting to add the edge
                return

        # If click is not near any existing node, add a new node
        self.add_node(x, y)

    def add_node(self, x, y):
        """Add a new node at the specified position."""
        node_id = len(self.nodes) + 1  # Create a unique node ID
        self.nodes[node_id] = (x, y)
        self.graph.add_node(node_id, pos=(x, y))

        # Draw the node
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="lightblue", outline="black")
        self.canvas.create_text(x, y, text=str(node_id), font=("Arial", 10))

        print(f"Node {node_id} added at position ({x}, {y}).")

    def attempt_add_edge(self, node1, node2):
        """Attempt to add an edge between two nodes while enforcing constraints."""
        if node1 == node2:
            print("Cannot create an edge between a node and itself.")
            return

        # Check if either node already has 3 edges
        if self.graph.degree(node1) >= 3:
            print(f"Node {node1} already has 3 edges and cannot accept more.")
            return

        if self.graph.degree(node1) >= 3:
            print(f"Node {node2} already has 3 edges and cannot accept more.")
            return

        # Prompt for edge label
        label = self.get_edge_label()
        if label is not None:
            # Temporarily add the edge to the graph
            self.graph.add_edge(node1, node2, label=label)

            # # Check the conditions
            # if not self.check_conditions(node1) or not self.check_conditions(node2):
            #     print(f"Edge between Node {node1} and Node {node2} violates the triangular or integer condition.")
            #     # Remove the temporary edge
            #     self.graph.remove_edge(node1, node2)
            #     return
            # Check the conditions only if the label is numeric
            if isinstance(label, (int, float)):
                if not self.check_conditions(node1) or not self.check_conditions(node2):
                    print(f"Edge between Node {node1} and Node {node2} violates the triangular or integer condition.")
                    # Remove the edge if conditions are not satisfied
                    self.graph.remove_edge(node1, node2)
                    return

            # Finalize the edge addition if conditions are satisfied
            self.add_edge(node1, node2, label)

    def add_edge(self, node1, node2, label):
        """Add an edge between two nodes with a specified label."""
        # Draw the edge on the canvas
        self.draw_edge(node1, node2, label)
        print(f"Edge added between Node {node1} and Node {node2} with label {label}.")

    def draw_edge(self, node1, node2, label):
        """Draw an edge between two nodes, with curvature if necessary."""
        x1, y1 = self.nodes[node1]
        x2, y2 = self.nodes[node2]

        # Check if there are multiple edges between the same nodes
        parallel_edges = self.graph.number_of_edges(node1, node2)
        if parallel_edges > 1:
            # Calculate curvature for parallel edges
            edge_index = parallel_edges - 1
            curvature_multiplier = self.curvature_slider.get()  # Get the slider value
            curvature = (edge_index - (parallel_edges - 1) / 2) * curvature_multiplier
            self.draw_curved_line(x1, y1, x2, y2, curvature, label)
        else:
            # Draw a straight line for a single edge
            self.canvas.create_line(x1, y1, x2, y2, fill="black")

            # Display the label on the edge (midpoint)
            label_x = (x1 + x2) / 2
            label_y = (y1 + y2) / 2
            self.canvas.create_text(label_x, label_y, text=str(label), font=("Arial", 9), fill="red")

    def draw_curved_line(self, x1, y1, x2, y2, curvature, label):
        """Draw a curved line between two points using a Bézier curve."""
        # Calculate control point for the curve
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dx = y2 - y1
        dy = x1 - x2
        control_x = mid_x + curvature * dx / math.sqrt(dx**2 + dy**2)
        control_y = mid_y + curvature * dy / math.sqrt(dx**2 + dy**2)

        # Draw the Bézier curve
        self.canvas.create_line(x1, y1, control_x, control_y, x2, y2, smooth=True, fill="black")

        # Display the label on the curve (near the midpoint)
        label_x = (x1 + x2 + 2 * control_x) / 4
        label_y = (y1 + y2 + 2 * control_y) / 4
        self.canvas.create_text(label_x, label_y, text=str(label), font=("Arial", 9), fill="red")

    def highlight_node(self, node_id):
        """Visually highlight the selected node."""
        x, y = self.nodes[node_id]
        self.canvas.create_oval(x-12, y-12, x+12, y+12, outline="red", width=2)

    def get_edge_label(self):
        """Prompt the user to input a label for the edge."""
        label = tk.simpledialog.askstring(
            "Input Edge Label",
            "Enter a label for the edge (must be an integer or half-integer):"
        )
        if label is None:  # If the user cancels the input dialog
            return None

        try:
            label = float(label)
            if label * 2 != int(label * 2):  # Check if the value is an integer or half-integer
                print("Label must be an integer or a half-integer.")
                return None
            return label
        except ValueError:
            # print("Invalid input: Label must be numeric.")
            # return None
            return label  # Allow non-numeric labels for flexibility
        
    
    def check_conditions(self, node):
        """Check the triangular and integer conditions at the node."""
        # Collect all labels for edges incident to this node
        labels = []
        for neighbor, edges in self.graph[node].items():
            for key, edge_data in edges.items():
                if "label" in edge_data:
                    labels.append(edge_data["label"])

        # Only check if the node has exactly 3 edges
        if len(labels) == 3:
            a, b, c = labels
            # Triangular condition
            if vertex_satisfies_triangular_conditions(labels) is False: 
                return False
            return True

        return True

    def save_graph(self):
        """Save the graph to a GraphML file."""
        # Create a copy of the graph to avoid modifying the original
        graph_copy = self.graph.copy()

        # Convert node attributes to GraphML-compatible types
        for node, attrs in graph_copy.nodes(data=True):
            for key, value in attrs.items():
                if isinstance(value, tuple):  # Convert tuples (e.g., (x, y)) to strings
                    graph_copy.nodes[node][key] = str(value)

        # Convert edge attributes to GraphML-compatible types
        for u, v, attrs in graph_copy.edges(data=True):
            for key, value in attrs.items():
                if isinstance(value, tuple):  # Convert tuples to strings
                    graph_copy.edges[u, v][key] = str(value)

        # Save the graph to a GraphML file
        try:
            nx.write_graphml(graph_copy, "drawn_graph_with_labels.graphml")
            print("Graph saved successfully to 'drawn_graph_with_labels.graphml'.")
            print("Exiting program.")
            sys.exit(0)
        except Exception as e:
            print(f"Error saving graph: {e}")


# Create the GUI
root = tk.Tk()
root.title("Graph Editor")
editor = GraphEditor(root)

# Add a save button
save_button = tk.Button(root, text="Save Graph", command=editor.save_graph)
save_button.pack(side=tk.BOTTOM, pady=10)

root.mainloop()