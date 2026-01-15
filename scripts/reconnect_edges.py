#!/usr/bin/env python3
"""
Reconnect Open Edges - Interactive tool for connecting open edges

This script loads a graph and allows you to connect two open edges together,
creating a new open edge. Multiple reconnections can be performed.

The reconnected edges are automatically flagged for probability calculations.
"""

import tkinter as tk
import tkinter.simpledialog
import tkinter.messagebox
import networkx as nx
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class EdgeReconnector:
    def __init__(self, master, input_file=None):
        self.master = master
        self.master.title("Spin Network Edge Reconnector")

        # Store input file path
        self.input_file = input_file

        # Reconnection state
        self.reconnections = []  # List of reconnection records
        self.selected_nodes = []  # Currently selected open nodes for reconnection

        # Create main frame
        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create toolbar
        self.create_toolbar(main_frame)

        # Create canvas
        self.canvas = tk.Canvas(main_frame, width=900, height=650, bg="#f5f5f5",
                                highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create info panel
        self.create_info_panel(main_frame)

        # Graph data
        self.graph = nx.MultiGraph()
        self.original_graph = None  # Store original for comparison
        self.nodes = {}
        self.edge_graphics = {}

        # Editor state
        self.mode = "select"
        self.hover_edge = None
        self.curvature = 50

        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_hover)

        # Keyboard shortcuts
        self.master.bind("<Key>", self.on_key_press)

        # Load input file if provided
        if input_file and os.path.exists(input_file):
            self.load_graph(input_file)

        self.update_display()
        self.redraw_all()

    def create_toolbar(self, parent):
        """Create toolbar."""
        toolbar = tk.Frame(parent, bg="#2c3e50", pady=8, padx=10)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        title = tk.Label(toolbar, text="Edge Reconnector", font=("Arial", 16, "bold"),
                        bg="#2c3e50", fg="white")
        title.pack(side=tk.LEFT, padx=10)

        util_frame = tk.Frame(toolbar, bg="#2c3e50")
        util_frame.pack(side=tk.RIGHT, padx=10)

        load_btn = tk.Button(util_frame, text="Load Graph", width=12, height=1,
                            command=self.load_graph_dialog, fg="black",
                            font=("Arial", 10, "bold"), cursor="hand2")
        load_btn.grid(row=0, column=0, padx=3)

        connect_btn = tk.Button(util_frame, text="Connect (C)", width=12, height=1,
                               command=self.connect_selected_edges, fg="black",
                               font=("Arial", 10, "bold"), cursor="hand2")
        connect_btn.grid(row=0, column=1, padx=3)

        undo_btn = tk.Button(util_frame, text="Undo (Z)", width=12, height=1,
                           command=self.undo_reconnection, fg="black",
                           font=("Arial", 10, "bold"), cursor="hand2")
        undo_btn.grid(row=0, column=2, padx=3)

        save_btn = tk.Button(util_frame, text="Save & Compute", width=14, height=1,
                           command=self.save_and_compute, fg="black",
                           font=("Arial", 10, "bold"), cursor="hand2")
        save_btn.grid(row=0, column=3, padx=3)

    def create_info_panel(self, parent):
        """Create info panel."""
        info_frame = tk.Frame(parent, bg="white", width=280, relief=tk.SUNKEN, bd=1)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.pack_propagate(False)

        tk.Label(info_frame, text="Instructions", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(10, 5))

        self.instructions = tk.Text(info_frame, height=10, width=30, wrap=tk.WORD,
                                   bg="#ecf0f1", fg="#2c3e50", font=("Arial", 9),
                                   relief=tk.FLAT, padx=10, pady=10, state=tk.DISABLED)
        self.instructions.pack(padx=10, pady=5)

        tk.Label(info_frame, text="Graph Statistics", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(15, 5))

        self.stats_label = tk.Label(info_frame, text="", font=("Arial", 10),
                                   bg="#ecf0f1", fg="#2c3e50", relief=tk.RAISED,
                                   bd=1, width=30, height=4, justify=tk.LEFT, padx=10)
        self.stats_label.pack(padx=10, pady=5)

        tk.Label(info_frame, text="Selected Edges", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(15, 5))

        self.selected_label = tk.Label(info_frame, text="None selected",
                                       font=("Arial", 9), bg="#fff3cd", fg="#856404",
                                       relief=tk.RAISED, bd=1, width=30, height=3,
                                       justify=tk.LEFT, padx=5)
        self.selected_label.pack(padx=10, pady=5)

        tk.Label(info_frame, text="Reconnections", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(15, 5))

        self.reconnect_label = tk.Label(info_frame, text="No reconnections yet",
                                        font=("Arial", 9), bg="#d1ecf1", fg="#0c5460",
                                        relief=tk.RAISED, bd=1, width=30, height=6,
                                        justify=tk.LEFT, padx=5, anchor="nw")
        self.reconnect_label.pack(padx=10, pady=5)

    def load_graph_dialog(self):
        """Open file dialog to load graph."""
        filename = tk.filedialog.askopenfilename(
            title="Select GraphML file",
            filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        if filename:
            self.load_graph(filename)

    def load_graph(self, filepath):
        """Load a graph from .graphml file."""
        try:
            # Store input file path
            self.input_file = filepath

            loaded_graph = nx.read_graphml(filepath, force_multigraph=True)

            self.graph.clear()
            self.nodes.clear()
            self.reconnections = []
            self.selected_edges = []

            # Copy nodes
            for node in loaded_graph.nodes():
                node_id = int(node) if node.isdigit() else node

                if 'x' in loaded_graph.nodes[node] and 'y' in loaded_graph.nodes[node]:
                    x = float(loaded_graph.nodes[node]['x'])
                    y = float(loaded_graph.nodes[node]['y'])
                else:
                    pos = nx.spring_layout(loaded_graph, seed=42)
                    x, y = pos[node]
                    x = (x + 1) * 400 + 50
                    y = (y + 1) * 300 + 25

                self.nodes[node_id] = (x, y)
                self.graph.add_node(node_id, pos=(x, y))

            # Copy edges
            for u, v, key, data in loaded_graph.edges(keys=True, data=True):
                u_id = int(u) if (isinstance(u, str) and u.isdigit()) else u
                v_id = int(v) if (isinstance(v, str) and v.isdigit()) else v

                label = data.get('label', '?')
                try:
                    label = float(label)
                except (ValueError, TypeError):
                    pass

                self.graph.add_edge(u_id, v_id, label=label, key=key)

            # Store original for comparison
            self.original_graph = self.graph.copy()

            self.update_display()
            self.redraw_all()

            print(f"✓ Loaded graph from {filepath}")
            print(f"  Nodes: {len(self.graph.nodes())}, Edges: {len(self.graph.edges())}")

        except Exception as e:
            tk.messagebox.showerror("Load Error", f"Failed to load graph:\n{e}")

    def get_open_nodes(self):
        """Get all open nodes (degree < 3)."""
        return [node for node in self.graph.nodes() if self.graph.degree(node) < 3]

    def get_open_edges(self):
        """Get all open edges (connected to vertices with degree < 3)."""
        open_edges = []
        for n1, n2, key, data in self.graph.edges(keys=True, data=True):
            if self.graph.degree(n1) < 3 or self.graph.degree(n2) < 3:
                label = data.get('label', '?')
                open_edges.append((n1, n2, key, label))
        return open_edges

    def is_open_edge(self, n1, n2, key):
        """Check if edge is open."""
        return self.graph.degree(n1) < 3 or self.graph.degree(n2) < 3

    def on_canvas_click(self, event):
        """Handle canvas click - select open nodes."""
        x, y = event.x, event.y
        node = self.find_node_at(x, y)

        if node and self.is_open_node(node):
            if node in self.selected_nodes:
                self.selected_nodes.remove(node)
            else:
                self.selected_nodes.append(node)

            self.update_display()
            self.redraw_all()

    def find_node_at(self, x, y):
        """Find node near coordinates."""
        threshold = 15  # Click radius
        for node, (nx, ny) in self.nodes.items():
            dist = math.sqrt((x - nx)**2 + (y - ny)**2)
            if dist < threshold:
                return node
        return None

    def is_open_node(self, node):
        """Check if a node is an open node (degree < 3)."""
        return self.graph.degree(node) < 3

    def on_canvas_hover(self, event):
        """Handle mouse hover."""
        x, y = event.x, event.y
        hover_edge = self.find_edge_at(x, y)

        if hover_edge != self.hover_edge:
            self.hover_edge = hover_edge
            self.redraw_all()

    def find_edge_at(self, x, y):
        """Find edge near coordinates."""
        threshold = 10
        for n1, n2, key in self.graph.edges(keys=True):
            if n1 not in self.nodes or n2 not in self.nodes:
                continue
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]

            dist = self.point_to_segment_distance(x, y, x1, y1, x2, y2)
            if dist < threshold:
                return (n1, n2, key)
        return None

    def point_to_segment_distance(self, px, py, x1, y1, x2, y2):
        """Distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def connect_selected_edges(self):
        """Connect the two selected open nodes."""
        if len(self.selected_nodes) != 2:
            tk.messagebox.showwarning(
                "Selection Error",
                "Please select exactly 2 open nodes to connect.\n" +
                f"Currently selected: {len(self.selected_nodes)}"
            )
            return

        open_node1, open_node2 = self.selected_nodes

        # Find the edges connected to these open nodes
        edges1 = list(self.graph.edges(open_node1, keys=True, data=True))
        edges2 = list(self.graph.edges(open_node2, keys=True, data=True))

        if not edges1 or not edges2:
            tk.messagebox.showerror(
                "Connection Error",
                "Selected nodes have no edges!"
            )
            return

        # Get the edge labels (should be only one edge per open node)
        # For each node, find its connecting edge
        edge1 = edges1[0] if edges1 else None
        edge2 = edges2[0] if edges2 else None

        if not edge1 or not edge2:
            tk.messagebox.showerror(
                "Connection Error",
                "Could not find edges for selected nodes!"
            )
            return

        # Extract labels
        label1 = edge1[3].get('label', 1.0) if len(edge1) > 3 else 1.0
        label2 = edge2[3].get('label', 1.0) if len(edge2) > 3 else 1.0

        # Find the "other" nodes (non-open endpoints)
        other_node1 = edge1[1] if edge1[0] == open_node1 else edge1[0]
        other_node2 = edge2[1] if edge2[0] == open_node2 else edge2[0]

        # Calculate all possible values based on triangle inequality
        possible_values = self.calculate_possible_values(label1, label2)

        # Ask user if they want all values or just one
        choice = tk.messagebox.askyesnocancel(
            "Reconnection Mode",
            f"Connecting edges with labels {label1} and {label2}.\n\n" +
            f"Possible new edge values: {possible_values}\n\n" +
            "Compute probabilities for:\n" +
            "  YES - All possible values (recommended for normalization test)\n" +
            "  NO - Single specific value\n" +
            "  CANCEL - Abort reconnection",
            parent=self.master
        )

        if choice is None:  # Cancel
            return

        if choice:  # Yes - compute all
            self.compute_all_mode = True
            # Store edge info for compute_all_reconnection
            edge_info1 = (open_node1, other_node1, edge1[2], label1)
            edge_info2 = (open_node2, other_node2, edge2[2], label2)
            self.compute_all_reconnection(edge_info1, edge_info2, label1, label2, possible_values,
                                          open_node1, open_node2, other_node1, other_node2)
            return
        else:  # No - single value
            self.compute_all_mode = False
            # Ask for new edge label
            new_label = tk.simpledialog.askstring(
                "New Edge Label",
                f"Connecting edges with labels {label1} and {label2}.\n" +
                f"Possible values: {possible_values}\n" +
                "Enter label for new open edge:",
                parent=self.master
            )

            if new_label is None:
                return

            try:
                new_label = float(new_label)
            except ValueError:
                pass

        # Merge the two open nodes into one reconnection node
        # Position it between the two open nodes
        x1, y1 = self.nodes[open_node1]
        x2, y2 = self.nodes[open_node2]
        new_x = (x1 + x2) / 2
        new_y = (y1 + y2) / 2

        # Remove old edges and nodes
        self.graph.remove_edge(open_node1, other_node1, edge1[2])
        self.graph.remove_edge(open_node2, other_node2, edge2[2])

        # Remove the open nodes
        self.graph.remove_node(open_node1)
        self.graph.remove_node(open_node2)
        del self.nodes[open_node1]
        del self.nodes[open_node2]

        # Create new reconnection node at the merge point
        new_node = max(self.nodes.keys(), default=0) + 1
        self.nodes[new_node] = (new_x, new_y)
        self.graph.add_node(new_node, pos=(new_x, new_y))

        # Connect the two "other" nodes to the new reconnection node
        self.graph.add_edge(other_node1, new_node, label=label1)
        self.graph.add_edge(other_node2, new_node, label=label2)

        # Add new open edge from the reconnection node
        # Create a new external node for the open end
        external_node = max(self.nodes.keys(), default=0) + 1
        ext_x = new_x + 50  # Offset for visibility
        ext_y = new_y - 50

        self.nodes[external_node] = (ext_x, ext_y)
        self.graph.add_node(external_node, pos=(ext_x, ext_y))
        self.graph.add_edge(new_node, external_node, label=new_label)

        # Record reconnection
        reconnection = {
            'old_edges': [
                {'nodes': (open_node1, other_node1), 'label': label1},
                {'nodes': (open_node2, other_node2), 'label': label2}
            ],
            'new_edge': {
                'nodes': (new_node, external_node),
                'label': new_label,
                'reconnection_node': new_node
            }
        }
        self.reconnections.append(reconnection)

        self.selected_nodes = []
        self.update_display()
        self.redraw_all()

        print(f"\n✓ Merged open nodes → created new open edge with label {new_label}")

    def calculate_possible_values(self, j1, j2):
        """Calculate all possible values for new edge based on triangle inequality."""
        possible = []
        j_min = abs(j1 - j2)
        j_max = j1 + j2

        # Determine step size (1.0 for integer, 0.5 for half-integer)
        if (j1 % 1 == 0) and (j2 % 1 == 0):
            step = 1.0
        else:
            step = 0.5

        current = j_min
        while current <= j_max:
            possible.append(current)
            current += step

        return possible

    def compute_all_reconnection(self, edge_info1, edge_info2, label1, label2, possible_values,
                                   open_node1, open_node2, other_node1, other_node2):
        """Create multiple reconnected graphs for all possible new edge values."""
        # Edge info format: (open_node, other_node, key, label)

        # Store original state for creating multiple variants
        original_graph = self.graph.copy()

        # We'll create a separate graph file for each possible value
        reconnection_data = {
            'old_edges': [
                {'nodes': (open_node1, other_node1), 'label': label1},
                {'nodes': (open_node2, other_node2), 'label': label2}
            ],
            'possible_values': possible_values,
            'compute_all': True
        }

        self.reconnections.append(reconnection_data)

        # Save the state before reconnection
        self.pre_reconnection_graph = original_graph

        self.selected_nodes = []

        print(f"\n✓ Marked nodes for reconnection with edges {label1} and {label2}")
        print(f"  Possible values: {possible_values}")
        print("  When you save, all probability computations will be performed automatically.")

        # Update display
        self.update_display()
        self.redraw_all()

    def undo_reconnection(self):
        """Undo the last reconnection."""
        if not self.reconnections:
            print("No reconnections to undo")
            return

        # For simplicity, reload from original and reapply all but last
        reconnection = self.reconnections.pop()

        # Reset to original
        self.graph = self.original_graph.copy()
        self.nodes = {n: self.graph.nodes[n].get('pos', (0, 0)) for n in self.graph.nodes()}

        # Reapply remaining reconnections
        # (This is simplified - full implementation would replay)

        self.update_display()
        self.redraw_all()
        print("↶ Undid last reconnection")

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        key = event.char.lower()
        if key == 'c':
            self.connect_selected_edges()
        elif key == 'z':
            self.undo_reconnection()
        elif key == 's':
            self.save_and_compute()

    def update_display(self):
        """Update all display elements."""
        # Instructions
        self.instructions.config(state=tk.NORMAL)
        self.instructions.delete(1.0, tk.END)
        self.instructions.insert(1.0,
            "1. Click open NODES (orange) to select\n" +
            "2. Select exactly 2 open nodes\n" +
            "3. Press 'Connect' or C key\n" +
            "4. Choose mode:\n" +
            "   YES - Compute all probabilities\n" +
            "   NO - Single value\n" +
            "5. Nodes merge automatically!\n" +
            "6. Press Save & Compute\n\n" +
            "Shortcuts:\n" +
            "C - Connect selected\n" +
            "Z - Undo\n" +
            "S - Save & Compute"
        )
        self.instructions.config(state=tk.DISABLED)

        # Stats
        open_nodes = self.get_open_nodes()
        stats_text = (
            f"Nodes: {len(self.graph.nodes())}\n" +
            f"Edges: {len(self.graph.edges())}\n" +
            f"Open Nodes: {len(open_nodes)}\n" +
            f"Reconnections: {len(self.reconnections)}"
        )
        self.stats_label.config(text=stats_text)

        # Selected nodes
        if not self.selected_nodes:
            self.selected_label.config(text="None selected\n(Click open nodes)")
        else:
            sel_text = f"Selected {len(self.selected_nodes)}/2:\n"
            for node in self.selected_nodes:
                degree = self.graph.degree(node)
                sel_text += f"  Node {node} (degree {degree})\n"
            self.selected_label.config(text=sel_text)

        # Reconnections
        if not self.reconnections:
            self.reconnect_label.config(text="No reconnections yet")
        else:
            recon_text = f"{len(self.reconnections)} reconnection(s):\n\n"
            for i, r in enumerate(self.reconnections, 1):
                old1 = r['old_edges'][0]
                old2 = r['old_edges'][1]
                new = r['new_edge']
                recon_text += f"{i}. {old1['label']}+{old2['label']} → {new['label']}\n"
            self.reconnect_label.config(text=recon_text)

    def redraw_all(self):
        """Redraw the entire graph."""
        self.canvas.delete("all")

        # Grid
        for i in range(0, 900, 50):
            self.canvas.create_line(i, 0, i, 650, fill="#e0e0e0", width=1)
        for i in range(0, 650, 50):
            self.canvas.create_line(0, i, 900, i, fill="#e0e0e0", width=1)

        # Draw edges
        try:
            for n1, n2, key in self.graph.edges(keys=True):
                if n1 in self.nodes and n2 in self.nodes:
                    edge_data = self.graph.edges[n1, n2, key]
                    label = edge_data.get('label', '?')

                    is_hover = (self.hover_edge == (n1, n2, key))
                    is_open = self.is_open_edge(n1, n2, key)

                    self.draw_edge(n1, n2, key, label, is_hover, False, is_open)
        except Exception as e:
            print(f"Warning: Error drawing edges: {e}")

        # Draw nodes
        for node_id in list(self.nodes.keys()):
            if node_id in self.nodes:
                x, y = self.nodes[node_id]
                is_open = self.is_open_node(node_id)
                is_selected = node_id in self.selected_nodes
                self.draw_node(node_id, x, y, is_open, is_selected)

    def draw_node(self, node_id, x, y, is_open=False, is_selected=False):
        """Draw a node."""
        radius = 12

        # Determine colors based on state
        if is_selected:
            fill = "#e74c3c"  # Red for selected
            outline = "#c0392b"
            width = 4
        elif is_open:
            fill = "#f39c12"  # Orange for open
            outline = "#d68910"
            width = 3
        else:
            fill = "#ecf0f1"  # Gray for regular
            outline = "#34495e"
            width = 2

        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                               fill=fill, outline=outline, width=width)
        self.canvas.create_text(x, y, text=str(node_id), font=("Arial", 9, "bold"),
                               fill="white" if (is_selected or is_open) else "#2c3e50")

    def draw_edge(self, node1, node2, key, label, is_hover, is_selected, is_open):
        """Draw an edge."""
        x1, y1 = self.nodes[node1]
        x2, y2 = self.nodes[node2]

        # Determine color
        if is_selected:
            color = "#9b59b6"  # Purple for selected
            width = 4
        elif is_hover and is_open:
            color = "#e74c3c"  # Red for hover on open
            width = 3
        elif is_open:
            color = "#f39c12"  # Orange for open
            width = 2
        else:
            color = "#34495e"  # Default
            width = 2

        # Draw line
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

        # Label
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
        bbox = self.canvas.bbox(self.canvas.create_text(lx, ly, text=str(label), font=("Arial", 10, "bold")))
        if bbox:
            self.canvas.create_rectangle(bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2, fill="#f5f5f5", outline="")
        self.canvas.create_text(lx, ly, text=str(label), font=("Arial", 10, "bold"), fill="#c0392b")

    def save_and_compute(self):
        """Save modified graph and compute probability."""
        if not self.reconnections:
            tk.messagebox.showwarning(
                "No Reconnections",
                "You haven't made any reconnections yet!\n" +
                "Select 2 open nodes and press Connect (C) first."
            )
            return

        # Check if user chose "compute all" mode
        compute_all = any(r.get('compute_all', False) for r in self.reconnections)

        if compute_all:
            # Compute all probabilities mode
            self.compute_all_probabilities_workflow()
        else:
            # Single probability mode - save graph
            self.save_single_reconnection()

    def save_single_reconnection(self):
        """Save single reconnection graph."""
        filename = tk.filedialog.asksaveasfilename(
            title="Save Modified Graph",
            defaultextension=".graphml",
            filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
            initialfile="reconnected_graph.graphml"
        )

        if not filename:
            return

        # Save graph
        graph_copy = self.graph.copy()
        for node, attrs in graph_copy.nodes(data=True):
            for key, value in attrs.items():
                if isinstance(value, tuple):
                    graph_copy.nodes[node][key] = str(value)

        for u, v, attrs in graph_copy.edges(data=True):
            for key, value in attrs.items():
                if isinstance(value, tuple):
                    graph_copy.edges[u, v][key] = str(value)

        try:
            nx.write_graphml(graph_copy, filename)

            # Save reconnection data
            recon_file = filename.replace('.graphml', '_reconnections.json')
            import json
            with open(recon_file, 'w') as f:
                json.dump(self.reconnections, f, indent=2)

            print(f"✓ Saved modified graph to {filename}")
            print(f"✓ Saved reconnection data to {recon_file}")

            tk.messagebox.showinfo(
                "Success",
                f"Graph and reconnection data saved!\n\n" +
                f"Graph: {filename}\n" +
                f"Data: {recon_file}\n\n" +
                f"Now run compute_probability.py to calculate probabilities."
            )

            self.master.destroy()

        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save:\n{e}")

    def compute_all_probabilities_workflow(self):
        """Compute probabilities for all possible values."""
        import subprocess
        import sys
        import tempfile

        # Get the reconnection data
        recon = self.reconnections[0]  # Assuming single reconnection for now
        edge1_data = recon['old_edges'][0]
        edge2_data = recon['old_edges'][1]

        # Format edge specifications
        edge1_spec = f"{edge1_data['nodes'][0]}-{edge1_data['nodes'][1]}"
        edge2_spec = f"{edge2_data['nodes'][0]}-{edge2_data['nodes'][1]}"

        # Save original graph to temp file if needed
        if not hasattr(self, 'input_file') or not self.input_file:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
                original_file = f.name
            nx.write_graphml(self.original_graph, original_file)
        else:
            original_file = self.input_file

        print(f"\n{'='*70}")
        print("COMPUTING ALL PROBABILITIES...")
        print(f"{'='*70}")
        print(f"Original graph: {original_file}")
        print(f"Reconnecting edges: {edge1_spec} and {edge2_spec}")
        print(f"This may take a moment...\n")

        # Run compute_all_probabilities.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        compute_script = os.path.join(script_dir, 'compute_all_probabilities.py')

        try:
            result = subprocess.run(
                [sys.executable, compute_script, original_file, edge1_spec, edge2_spec],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print(result.stdout)

                # Parse results to show in message box
                if "PASSED" in result.stdout:
                    status = "✓ PASSED"
                    status_msg = "Probabilities sum to 1!\nPhysical consistency verified."
                else:
                    status = "✗ FAILED"
                    status_msg = "Probabilities do not sum to 1.\nCheck the output for details."

                # Extract probability summary
                lines = result.stdout.split('\n')
                summary = []
                in_table = False
                for line in lines:
                    if 'New Edge' in line and 'Probability' in line:
                        in_table = True
                    if in_table and ('--' in line or line.strip().startswith(('0', '1', '2', 'TOTAL'))):
                        summary.append(line)
                    if in_table and 'PASSED' in line:
                        break

                summary_text = '\n'.join(summary[:10])  # First 10 lines of summary

                tk.messagebox.showinfo(
                    "Probability Computation Complete",
                    f"{status}\n\n" +
                    f"Reconnected: {edge1_spec} + {edge2_spec}\n\n" +
                    f"{status_msg}\n\n" +
                    "Full results saved to JSON file.\n" +
                    "See console output for details."
                )

                self.master.destroy()

            else:
                print(result.stderr)
                tk.messagebox.showerror(
                    "Computation Error",
                    f"Failed to compute probabilities:\n\n{result.stderr[:500]}"
                )

        except subprocess.TimeoutExpired:
            tk.messagebox.showerror(
                "Timeout",
                "Probability computation took too long (>5 minutes).\n" +
                "Try with a simpler graph or use the command line."
            )
        except Exception as e:
            tk.messagebox.showerror(
                "Error",
                f"Failed to run probability computation:\n\n{str(e)}"
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconnect open edges in a spin network")
    parser.add_argument("input_file", nargs="?", help="Input .graphml file")
    args = parser.parse_args()

    root = tk.Tk()
    reconnector = EdgeReconnector(root, args.input_file)
    root.mainloop()
