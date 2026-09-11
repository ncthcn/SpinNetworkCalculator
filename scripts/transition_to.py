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
Transition Tool - Interactive tool for spin network state transitions.

Loads a spin network graph and lets the user modify it by:
1. Adding extra edges (shown in green)
2. Reconnecting open ends (select two open nodes, press C)

Press S to save the modified graph and exit.  No computation is performed here;
call calculate_probability(n1, n2) afterwards to evaluate transition probabilities.
"""

import tkinter as tk
import tkinter.simpledialog
import tkinter.messagebox
import tkinter.filedialog
import networkx as nx
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import vertex_satisfies_triangular_conditions, parse_spin_label, is_numeric_label


# -----------------------------------------------------------------------
# Transition tool (tkinter GUI)
# -----------------------------------------------------------------------
# Open nodes/edges are highlighted in orange; added edges in green.
# Keyboard shortcuts: N add node | E add edge | C reconnect selected |
#                     Z undo | S save | R reset view | Esc → select mode
class TransitionTool:
    def __init__(self, master, input_file=None):
        self.master = master
        self.master.title("Spin Network Transition Tool")

        # Store input file path
        self.input_file = input_file

        # State tracking
        self.added_edges = []  # List of added edge records {nodes, label, key}
        self.reconnections = []  # List of reconnection records
        self.selected_nodes = []  # Currently selected nodes

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
        self.mode = "select"  # Modes: select, add_node, add_edge
        self.hover_edge = None
        self.hover_node = None
        self.curvature = 50
        self.history = []

        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.panning = False
        self.pan_start = None

        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_hover)

        # Zoom bindings
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)

        # Pan bindings
        self.canvas.bind("<Button-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_motion)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.canvas.bind("<Shift-Button-1>", self.on_pan_start)
        self.canvas.bind("<Shift-B1-Motion>", self.on_pan_motion)
        self.canvas.bind("<Shift-ButtonRelease-1>", self.on_pan_end)

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

        title = tk.Label(toolbar, text="Transition Tool", font=("Arial", 16, "bold"),
                        bg="#2c3e50", fg="white")
        title.pack(side=tk.LEFT, padx=10)

        util_frame = tk.Frame(toolbar, bg="#2c3e50")
        util_frame.pack(side=tk.RIGHT, padx=10)

        load_btn = tk.Button(util_frame, text="Load Graph", width=12, height=1,
                            command=self.load_graph_dialog, fg="black",
                            font=("Arial", 10, "bold"), cursor="hand2")
        load_btn.grid(row=0, column=0, padx=3)

        add_node_btn = tk.Button(util_frame, text="Add Node (N)", width=12, height=1,
                                command=lambda: self.set_mode("add_node"), fg="black",
                                font=("Arial", 10, "bold"), cursor="hand2")
        add_node_btn.grid(row=0, column=1, padx=3)

        add_edge_btn = tk.Button(util_frame, text="Add Edge (E)", width=12, height=1,
                                command=lambda: self.set_mode("add_edge"), fg="black",
                                font=("Arial", 10, "bold"), cursor="hand2")
        add_edge_btn.grid(row=0, column=2, padx=3)

        connect_btn = tk.Button(util_frame, text="Reconnect (C)", width=12, height=1,
                               command=self.connect_selected_nodes, fg="black",
                               font=("Arial", 10, "bold"), cursor="hand2")
        connect_btn.grid(row=0, column=3, padx=3)

        undo_btn = tk.Button(util_frame, text="Undo (Z)", width=10, height=1,
                           command=self.undo, fg="black",
                           font=("Arial", 10, "bold"), cursor="hand2")
        undo_btn.grid(row=0, column=4, padx=3)

        save_btn = tk.Button(util_frame, text="Save (S)", width=12, height=1,
                           command=self.save_and_exit, fg="black",
                           font=("Arial", 10, "bold"), cursor="hand2")
        save_btn.grid(row=0, column=5, padx=3)

    def create_info_panel(self, parent):
        """Create info panel."""
        info_frame = tk.Frame(parent, bg="white", width=300, relief=tk.SUNKEN, bd=1)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.pack_propagate(False)

        tk.Label(info_frame, text="Current Mode", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(10, 5))

        self.mode_label = tk.Label(info_frame, text="Select", font=("Arial", 14),
                                   bg="#ecf0f1", fg="#2c3e50", relief=tk.RAISED, bd=2,
                                   width=25, height=2)
        self.mode_label.pack(padx=10, pady=5)

        tk.Label(info_frame, text="Instructions", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(10, 5))

        self.instructions = tk.Text(info_frame, height=8, width=32, wrap=tk.WORD,
                                   bg="#ecf0f1", fg="#2c3e50", font=("Arial", 9),
                                   relief=tk.FLAT, padx=10, pady=10, state=tk.DISABLED)
        self.instructions.pack(padx=10, pady=5)

        tk.Label(info_frame, text="Graph Statistics", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(10, 5))

        self.stats_label = tk.Label(info_frame, text="", font=("Arial", 10),
                                   bg="#ecf0f1", fg="#2c3e50", relief=tk.RAISED,
                                   bd=1, width=32, height=5, justify=tk.LEFT, padx=10)
        self.stats_label.pack(padx=10, pady=5)

        tk.Label(info_frame, text="Added Edges (Green)", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(10, 5))

        self.added_label = tk.Label(info_frame, text="None",
                                    font=("Arial", 9), bg="#d4edda", fg="#155724",
                                    relief=tk.RAISED, bd=1, width=32, height=4,
                                    justify=tk.LEFT, padx=5, anchor="nw")
        self.added_label.pack(padx=10, pady=5)

        tk.Label(info_frame, text="Reconnections", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(10, 5))

        self.reconnect_label = tk.Label(info_frame, text="None",
                                        font=("Arial", 9), bg="#d1ecf1", fg="#0c5460",
                                        relief=tk.RAISED, bd=1, width=32, height=4,
                                        justify=tk.LEFT, padx=5, anchor="nw")
        self.reconnect_label.pack(padx=10, pady=5)

    def set_mode(self, mode):
        """Change the current editing mode."""
        self.mode = mode
        self.selected_nodes = []
        self.update_display()
        self.redraw_all()

    def load_graph_dialog(self):
        """Open file dialog to load graph."""
        filename = tk.filedialog.askopenfilename(
            title="Select GraphML file",
            filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        if filename:
            self.load_graph(filename)

    # Clears the current state and loads the graph into self.graph and
    # self.nodes. Stores a copy in self.original_graph for later ||G1|| computation.
    def load_graph(self, filepath):
        """Load a graph from .graphml file."""
        try:
            self.input_file = filepath
            loaded_graph = nx.read_graphml(filepath, force_multigraph=True)

            self.graph.clear()
            self.nodes.clear()
            self.added_edges = []
            self.reconnections = []
            self.selected_nodes = []
            self.history = []

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

                raw_label = data.get('label', '?')
                label = parse_spin_label(str(raw_label)) if raw_label != '?' else raw_label

                self.graph.add_edge(u_id, v_id, label=label, key=key, added=False)

            # Store original for comparison
            self.original_graph = self.graph.copy()

            self.master.title(f"Transition Tool - {os.path.basename(filepath)}")
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

    def is_open_node(self, node):
        """Check if a node is open (degree < 3)."""
        return self.graph.degree(node) < 3

    def is_open_edge(self, n1, n2, key):
        """Check if edge is open."""
        return self.graph.degree(n1) < 3 or self.graph.degree(n2) < 3

    def is_added_edge(self, n1, n2, key):
        """Check if edge was added (flagged)."""
        try:
            return self.graph.edges[n1, n2, key].get('added', False)
        except KeyError:
            return False

    def on_canvas_click(self, event):
        """Handle canvas click based on current mode."""
        x, y = event.x, event.y
        clicked_node = self.find_node_at(x, y)

        if self.mode == "select":
            # Select open nodes for reconnection
            if clicked_node and self.is_open_node(clicked_node):
                if clicked_node in self.selected_nodes:
                    self.selected_nodes.remove(clicked_node)
                else:
                    self.selected_nodes.append(clicked_node)
                self.update_display()
                self.redraw_all()

        elif self.mode == "add_node":
            # Add a new node at clicked position
            if clicked_node is None:
                self.add_node(x, y)
                self.update_display()
                self.redraw_all()

        elif self.mode == "add_edge":
            # Add edge between two existing nodes
            if clicked_node is not None:
                if len(self.selected_nodes) == 0:
                    self.selected_nodes.append(clicked_node)
                elif len(self.selected_nodes) == 1:
                    if clicked_node != self.selected_nodes[0]:
                        self.add_edge_between_nodes(self.selected_nodes[0], clicked_node)
                    self.selected_nodes = []
                self.update_display()
                self.redraw_all()

    def add_node(self, x, y):
        """Add a new node at the specified screen position."""
        # Convert screen to world coordinates
        wx, wy = self.screen_to_world(x, y)

        # Save state for undo
        self.save_state("Add node")

        # Create new node
        node_id = max(self.nodes.keys(), default=0) + 1
        self.nodes[node_id] = (wx, wy)
        self.graph.add_node(node_id, pos=(wx, wy))

        print(f"✓ Added node {node_id} at ({wx:.0f}, {wy:.0f})")

    def add_edge_between_nodes(self, node1, node2):
        """Add a new edge between two existing nodes."""
        # Check degree constraints
        if self.graph.degree(node1) >= 3:
            tk.messagebox.showwarning("Invalid Edge", f"Node {node1} already has 3 edges!")
            return

        if self.graph.degree(node2) >= 3:
            tk.messagebox.showwarning("Invalid Edge", f"Node {node2} already has 3 edges!")
            return

        # Get edge label
        label = self.get_edge_label()
        if label is None:
            return

        # Save state for undo
        self.save_state("Add edge")

        # Add edge
        key = self.graph.add_edge(node1, node2, label=label, added=True)

        # Check triangular conditions
        if isinstance(label, (int, float)):
            if not self.check_conditions(node1) or not self.check_conditions(node2):
                tk.messagebox.showerror(
                    "Triangular Condition Violated",
                    f"Edge with label {label} violates triangular inequality!\n" +
                    f"For edges j₁, j₂, j₃ at a node: |j₁-j₂| ≤ j₃ ≤ j₁+j₂"
                )
                self.undo()
                return

        # Record as added edge
        self.added_edges.append({
            'nodes': (node1, node2),
            'label': label,
            'key': key
        })

        print(f"✓ Added edge: {node1} --[{label}]-- {node2} (flagged)")

    def get_edge_label(self):
        """Prompt for edge label."""
        raw = tk.simpledialog.askstring(
            "Edge Label",
            "Enter spin value, symbol, or expression:\n"
            "  • Numeric:     1,  1/2,  1.5\n"
            "  • Symbol:      F_1,  a\n"
            "  • Expression:  a+2b,  (a+b)/2",
            parent=self.master
        )

        if raw is None:
            return None

        label = parse_spin_label(raw)

        if isinstance(label, (int, float)):
            if label * 2 != int(label * 2):
                tk.messagebox.showwarning("Invalid Label", "Numeric spin must be integer or half-integer!")
                return None

        return label

    def check_conditions(self, node):
        """Check triangular conditions and integer-sum admissibility at a node."""
        # Only numeric labels can be checked; symbolic labels are skipped.
        labels = []
        for neighbor, edges in self.graph[node].items():
            for key, edge_data in edges.items():
                label = edge_data.get("label")
                if isinstance(label, (int, float)):
                    labels.append(label)

        if len(labels) == 3:
            if not vertex_satisfies_triangular_conditions(labels):
                return False
            if sum(labels) != int(sum(labels)):
                return False
        return True

    # Reconnects two selected open nodes: finds their incident edges (label_a,
    # label_b), enumerates all admissible new values, then either records a
    # "compute all" marker or physically rewires the graph for a single value.
    def connect_selected_nodes(self):
        """Connect the two selected open nodes (reconnection)."""
        if len(self.selected_nodes) != 2:
            tk.messagebox.showwarning(
                "Selection Error",
                "Please select exactly 2 open nodes to reconnect.\n" +
                f"Currently selected: {len(self.selected_nodes)}\n\n" +
                "Use Select mode and click on orange (open) nodes."
            )
            return

        open_node1, open_node2 = self.selected_nodes

        # Find the edges connected to these open nodes
        edges1 = list(self.graph.edges(open_node1, keys=True, data=True))
        edges2 = list(self.graph.edges(open_node2, keys=True, data=True))

        if not edges1 or not edges2:
            tk.messagebox.showerror("Connection Error", "Selected nodes have no edges!")
            return

        edge1 = edges1[0]
        edge2 = edges2[0]

        # Extract labels
        label1 = edge1[3].get('label', 1.0)
        label2 = edge2[3].get('label', 1.0)

        # Find the "other" nodes (non-open endpoints)
        other_node1 = edge1[1] if edge1[0] == open_node1 else edge1[0]
        other_node2 = edge2[1] if edge2[0] == open_node2 else edge2[0]

        # Calculate admissible c values (only meaningful for numeric labels)
        labels_are_symbolic = not (isinstance(label1, (int, float)) and
                                   isinstance(label2, (int, float)))
        possible_values = (
            self.calculate_possible_values(label1, label2)
            if not labels_are_symbolic else []
        )

        prompt_suffix = (
            "Enter label for new open edge (numeric or symbolic):"
            if labels_are_symbolic else
            f"Admissible values: {possible_values}\nEnter label for new open edge:"
        )
        new_label = tk.simpledialog.askstring(
            "New Edge Label",
            f"Reconnecting edges with labels {label1} and {label2}.\n\n{prompt_suffix}",
            parent=self.master
        )

        if new_label is None:
            return

        try:
            new_label = float(new_label)
        except ValueError:
            pass

        # Validate numeric labels against triangle inequality
        if (not labels_are_symbolic and
                isinstance(new_label, (int, float)) and
                new_label not in possible_values):
            tk.messagebox.showerror(
                "Invalid Label",
                f"Label {new_label} is not admissible.\n"
                f"Valid values: {possible_values}\n\n"
                f"j₁+j₂+j₃ must be integer."
            )
            return

        self.save_state("Reconnection")
        self.perform_reconnection(open_node1, open_node2, other_node1, other_node2,
                                  edge1, edge2, label1, label2, new_label)

        self.update_display()
        self.redraw_all()

    # Physically rewires the graph: removes both open edges and their stub nodes,
    # creates a new trivalent reconnection node connected to other_node1/2 with
    # label1/2, and adds a new open edge with new_label.
    def perform_reconnection(self, open_node1, open_node2, other_node1, other_node2,
                            edge1, edge2, label1, label2, new_label):
        """Perform the actual reconnection operation."""
        # Position for new reconnection node
        x1, y1 = self.nodes[open_node1]
        x2, y2 = self.nodes[open_node2]
        new_x = (x1 + x2) / 2
        new_y = (y1 + y2) / 2

        # Remove old edges and nodes
        self.graph.remove_edge(open_node1, other_node1, edge1[2])
        self.graph.remove_edge(open_node2, other_node2, edge2[2])

        self.graph.remove_node(open_node1)
        self.graph.remove_node(open_node2)
        del self.nodes[open_node1]
        del self.nodes[open_node2]

        # Create new reconnection node
        new_node = max(self.nodes.keys(), default=0) + 1
        self.nodes[new_node] = (new_x, new_y)
        self.graph.add_node(new_node, pos=(new_x, new_y))

        # Connect to the other nodes
        self.graph.add_edge(other_node1, new_node, label=label1)
        self.graph.add_edge(other_node2, new_node, label=label2)

        # Create new external node for open end
        external_node = max(self.nodes.keys(), default=0) + 1
        ext_x = new_x + 50
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
            },
            'compute_all': False
        }
        self.reconnections.append(reconnection)
        self.selected_nodes = []

        print(f"\n✓ Reconnected: {label1} + {label2} → {new_label}")

    # Returns [|j1-j2|, |j1-j2|+1, ..., j1+j2]. Integer step enforces the
    # admissibility rule j1+j2+j3 ∈ Z at the reconnection vertex.
    def calculate_possible_values(self, j1, j2):
        possible = []
        j_min = abs(j1 - j2)
        j_max = j1 + j2

        current = j_min
        while current <= j_max:
            possible.append(current)
            current += 1.0

        return possible

    def find_node_at(self, x, y):
        """Find node near screen coordinates."""
        wx, wy = self.screen_to_world(x, y)
        threshold = 15 / self.zoom_level
        for node, (nx, ny) in self.nodes.items():
            dist = math.sqrt((wx - nx)**2 + (wy - ny)**2)
            if dist < threshold:
                return node
        return None

    def find_edge_at(self, x, y):
        """Find edge near screen coordinates."""
        wx, wy = self.screen_to_world(x, y)
        threshold = 10 / self.zoom_level
        for n1, n2, key in self.graph.edges(keys=True):
            if n1 not in self.nodes or n2 not in self.nodes:
                continue
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]

            dist = self.point_to_segment_distance(wx, wy, x1, y1, x2, y2)
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

    def on_canvas_hover(self, event):
        """Handle mouse hover."""
        x, y = event.x, event.y
        hover_node = self.find_node_at(x, y)
        hover_edge = self.find_edge_at(x, y)

        if hover_node != self.hover_node or hover_edge != self.hover_edge:
            self.hover_node = hover_node
            self.hover_edge = hover_edge
            self.redraw_all()

    def save_state(self, action):
        """Save current state for undo."""
        state = {
            'graph': self.graph.copy(),
            'nodes': self.nodes.copy(),
            'added_edges': self.added_edges.copy(),
            'reconnections': self.reconnections.copy(),
            'action': action
        }
        self.history.append(state)
        if len(self.history) > 50:
            self.history.pop(0)

    def undo(self):
        """Undo last action."""
        if not self.history:
            print("Nothing to undo")
            return

        state = self.history.pop()
        self.graph = state['graph']
        self.nodes = state['nodes']
        self.added_edges = state['added_edges']
        self.reconnections = state['reconnections']
        self.selected_nodes = []
        self.update_display()
        self.redraw_all()
        print(f"↶ Undid: {state['action']}")

    # ========== Zoom and Pan Methods ==========

    def screen_to_world(self, sx, sy):
        """Convert screen coordinates to world coordinates."""
        wx = (sx - self.pan_offset[0]) / self.zoom_level
        wy = (sy - self.pan_offset[1]) / self.zoom_level
        return wx, wy

    def world_to_screen(self, wx, wy):
        """Convert world coordinates to screen coordinates."""
        sx = wx * self.zoom_level + self.pan_offset[0]
        sy = wy * self.zoom_level + self.pan_offset[1]
        return sx, sy

    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming."""
        mx, my = event.x, event.y
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = 1.1
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = 0.9
        else:
            return

        new_zoom = self.zoom_level * factor
        if new_zoom < 0.2 or new_zoom > 5.0:
            return

        wx, wy = self.screen_to_world(mx, my)
        self.zoom_level = new_zoom
        self.pan_offset[0] = mx - wx * self.zoom_level
        self.pan_offset[1] = my - wy * self.zoom_level
        self.redraw_all()

    def on_pan_start(self, event):
        """Start panning."""
        self.panning = True
        self.pan_start = (event.x, event.y)

    def on_pan_motion(self, event):
        """Handle panning motion."""
        if self.panning and self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            self.pan_start = (event.x, event.y)
            self.redraw_all()

    def on_pan_end(self, event):
        """End panning."""
        self.panning = False
        self.pan_start = None

    def reset_view(self):
        """Reset zoom and pan to default."""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.redraw_all()

    # ========== End Zoom and Pan Methods ==========

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        # Check keysym for special keys like Escape
        if event.keysym == 'Escape':
            self.set_mode("select")
            return

        key = event.char.lower() if event.char else ''
        if key == 'n':
            self.set_mode("add_node")
        elif key == 'e':
            self.set_mode("add_edge")
        elif key == 'c':
            self.connect_selected_nodes()
        elif key == 'z':
            self.undo()
        elif key == 's':
            self.save_and_exit()
        elif key == 'r':
            self.reset_view()

    def update_display(self):
        """Update all display elements."""
        # Mode display
        mode_info = {
            "select": ("Select Nodes", "#3498db"),
            "add_node": ("Add Node", "#9b59b6"),
            "add_edge": ("Add Edge", "#27ae60"),
        }
        title, color = mode_info.get(self.mode, ("Unknown", "#95a5a6"))
        self.mode_label.config(text=title, bg=color, fg="white")

        # Instructions
        self.instructions.config(state=tk.NORMAL)
        self.instructions.delete(1.0, tk.END)
        if self.mode == "select":
            instr = ("Click orange (open) nodes to select\n"
                    "Select 2 nodes then press C to reconnect\n\n"
                    "Shortcuts: N=Add node, E=Add edge\n"
                    "C=Reconnect, Z=Undo, S=Compute\n"
                    "R=Reset view\n\n"
                    "Zoom: Mouse wheel\n"
                    "Pan: Shift+drag or middle-click")
        elif self.mode == "add_node":
            instr = ("Click on empty space to add a node\n"
                    "New nodes can be connected with edges\n\n"
                    "Shortcuts: Esc=Select mode, R=Reset view\n\n"
                    "Zoom: Mouse wheel\n"
                    "Pan: Shift+drag or middle-click")
        elif self.mode == "add_edge":
            instr = ("Click two nodes to add edge between them\n"
                    "Added edges shown in GREEN\n"
                    "Open edges auto-detected (orange)\n\n"
                    "Shortcuts: Esc=Select mode, R=Reset view")
        else:
            instr = ""
        self.instructions.insert(1.0, instr)
        self.instructions.config(state=tk.DISABLED)

        # Stats
        open_nodes = self.get_open_nodes()
        stats_text = (
            f"Nodes: {len(self.graph.nodes())}\n" +
            f"Edges: {len(self.graph.edges())}\n" +
            f"Open Nodes: {len(open_nodes)}\n" +
            f"Added Edges: {len(self.added_edges)}\n" +
            f"Reconnections: {len(self.reconnections)}"
        )
        self.stats_label.config(text=stats_text)

        # Added edges
        if not self.added_edges:
            self.added_label.config(text="No edges added yet")
        else:
            added_text = ""
            for e in self.added_edges[:5]:
                n1, n2 = e['nodes']
                label = e['label']
                added_text += f"{n1}--[{label}]--{n2}\n"
            if len(self.added_edges) > 5:
                added_text += f"... and {len(self.added_edges)-5} more"
            self.added_label.config(text=added_text)

        # Reconnections
        if not self.reconnections:
            self.reconnect_label.config(text="No reconnections yet")
        else:
            recon_text = ""
            for r in self.reconnections:
                old1 = r['old_edges'][0]
                old2 = r['old_edges'][1]
                if r.get('compute_all', False):
                    recon_text += f"{old1['label']}+{old2['label']} → ALL\n"
                else:
                    new = r.get('new_edge', {})
                    new_label = new.get('label', '?')
                    recon_text += f"{old1['label']}+{old2['label']} → {new_label}\n"
            self.reconnect_label.config(text=recon_text)

    def redraw_all(self):
        """Redraw the entire graph."""
        self.canvas.delete("all")

        # Grid (in screen space)
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
                    is_added = self.is_added_edge(n1, n2, key)

                    self.draw_edge(n1, n2, key, label, is_hover, is_open, is_added)
        except Exception as e:
            print(f"Warning: Error drawing edges: {e}")

        # Draw nodes
        for node_id in list(self.nodes.keys()):
            if node_id in self.nodes:
                wx, wy = self.nodes[node_id]
                is_open = self.is_open_node(node_id)
                is_selected = node_id in self.selected_nodes
                is_hover = (node_id == self.hover_node)
                self.draw_node(node_id, wx, wy, is_open, is_selected, is_hover)

        # Display zoom level
        zoom_text = f"Zoom: {self.zoom_level:.1f}x (R to reset)"
        self.canvas.create_text(10, 10, anchor="nw", text=zoom_text,
                               font=("Arial", 9), fill="#666666")

    def draw_node(self, node_id, wx, wy, is_open=False, is_selected=False, is_hover=False):
        """Draw a node (wx, wy are world coordinates)."""
        # Transform to screen coordinates
        sx, sy = self.world_to_screen(wx, wy)
        radius = 12 * self.zoom_level

        if is_selected:
            fill = "#e74c3c"  # Red for selected
            outline = "#c0392b"
            width = 4
        elif is_hover and is_open:
            fill = "#3498db"  # Blue for hover on open
            outline = "#2980b9"
            width = 3
        elif is_open:
            fill = "#f39c12"  # Orange for open
            outline = "#d68910"
            width = 3
        else:
            fill = "#ecf0f1"  # Gray for regular
            outline = "#34495e"
            width = 2

        self.canvas.create_oval(sx-radius, sy-radius, sx+radius, sy+radius,
                               fill=fill, outline=outline, width=width)
        font_size = max(7, int(9 * self.zoom_level))
        self.canvas.create_text(sx, sy, text=str(node_id), font=("Arial", font_size, "bold"),
                               fill="white" if (is_selected or is_open or is_hover) else "#2c3e50")

    def draw_edge(self, node1, node2, key, label, is_hover, is_open, is_added):
        """Draw an edge (uses world coordinates internally)."""
        wx1, wy1 = self.nodes[node1]
        wx2, wy2 = self.nodes[node2]

        # Transform to screen coordinates
        sx1, sy1 = self.world_to_screen(wx1, wy1)
        sx2, sy2 = self.world_to_screen(wx2, wy2)

        # Determine color - added edges are GREEN
        if is_added:
            color = "#27ae60"  # Green for added edges
            width = 3
        elif is_hover:
            color = "#3498db"  # Blue for hover
            width = 3
        elif is_open:
            color = "#f39c12"  # Orange for open
            width = 2
        else:
            color = "#34495e"  # Default
            width = 2

        # Draw line
        self.canvas.create_line(sx1, sy1, sx2, sy2, fill=color, width=width)

        # Label
        lx, ly = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        font_size = max(7, int(10 * self.zoom_level))
        bbox = self.canvas.bbox(self.canvas.create_text(lx, ly, text=str(label), font=("Arial", font_size, "bold")))
        if bbox:
            bg_color = "#d4edda" if is_added else "#f5f5f5"
            self.canvas.create_rectangle(bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2, fill=bg_color, outline="")
        text_color = "#155724" if is_added else "#c0392b"
        self.canvas.create_text(lx, ly, text=str(label), font=("Arial", font_size, "bold"), fill=text_color)

    def save_and_exit(self):
        """Save the modified graph and structural metadata, then close.

        Writes two files into the same directory as the input graph:
          - transition_to_graph.graphml         — the new graph
          - transition_to_graph_transition.json — structural metadata
            (added_edges, reconnections) for SpinNetwork.transition_to()

        No norms or probabilities are computed here.  Call
        calculate_probability(n1, n2) in Python after closing this window.
        """
        if not self.added_edges and not self.reconnections:
            tk.messagebox.showwarning(
                "Nothing to Save",
                "No modifications were made.\n\n"
                "Add edges (E) or perform reconnections (C) first."
            )
            return

        if self.input_file:
            input_dir = os.path.dirname(self.input_file) or "."
            filename = os.path.join(input_dir, "transition_to_graph.graphml")
        else:
            filename = tk.filedialog.asksaveasfilename(
                title="Save Transition Graph",
                defaultextension=".graphml",
                filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
                initialfile="transition_to_graph.graphml"
            )
            if not filename:
                return

        # Sanitize node/edge attributes for GraphML serialisation
        graph_copy = self.graph.copy()
        _graphml_ok = (bool, int, float, str)
        for node, attrs in graph_copy.nodes(data=True):
            for key, value in list(attrs.items()):
                if not isinstance(value, _graphml_ok):
                    graph_copy.nodes[node][key] = str(value)
        for u, v, key, attrs in graph_copy.edges(keys=True, data=True):
            for k, value in list(attrs.items()):
                if isinstance(value, tuple) or hasattr(value, 'free_symbols'):
                    graph_copy.edges[u, v, key][k] = str(value)

        nx.write_graphml(graph_copy, filename)

        # Minimal structural metadata — no norms, no probability values
        import json

        def _serialize(obj):
            return list(obj) if isinstance(obj, tuple) else obj

        metadata = {
            'original_file': self.input_file,
            'new_file': filename,
            'added_edges': [
                {k: _serialize(v) for k, v in e.items()}
                for e in self.added_edges
            ],
            'reconnections': [
                {k: _serialize(v) if not isinstance(v, dict) else
                    {kk: _serialize(vv) for kk, vv in v.items()}
                 for k, v in r.items()}
                for r in self.reconnections
            ],
        }

        json_path = filename.replace('.graphml', '_transition.json')
        with open(json_path, 'w') as fh:
            json.dump(metadata, fh, indent=2, default=str)

        print(f"✓ Saved modified graph to {filename}")
        self.master.destroy()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spin network transition tool")
    parser.add_argument("input_file", nargs="?", help="Input .graphml file")
    args = parser.parse_args()

    root = tk.Tk()
    tool = TransitionTool(root, args.input_file)
    root.mainloop()
