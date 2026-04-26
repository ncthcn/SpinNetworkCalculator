#!/usr/bin/env python3
"""
Spin Network Graph Modifier

Loads an existing .graphml file and allows modifications:
- Add/delete nodes
- Add/delete edges
- Move nodes
- Triangular constraint validation
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

from src.utils import vertex_satisfies_triangular_conditions, parse_spin_label


# -----------------------------------------------------------------------
# Existing-graph editor (tkinter GUI)
# -----------------------------------------------------------------------
#
# Identical editing capabilities to graph.py but loads an existing .graphml
# and overwrites it on save (rather than always writing drawn_graph.graphml).
# Used by compare_graphs.py to let the user manually produce a modified graph
# before the norm comparison step.
#
# Open edges (degree < 3) are highlighted in orange.
# Keyboard shortcuts: N add node | E add edge | M move | D delete node |
#                     X delete edge | Z undo | S save & exit | R reset view
class GraphModifier:
    def __init__(self, master, input_file=None):
        self.master = master
        self.master.title("Spin Network Graph Modifier")

        # Store input file path for overwriting
        self.input_file = input_file

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
        self.nodes = {}  # {node_id: (x, y)}
        self.node_graphics = {}  # {node_id: [oval_id, text_id]}
        self.edge_graphics = {}  # {(node1, node2, key): [line_ids, label_id]}

        # Editor state
        self.mode = "add_node"
        self.selected_node = None
        self.dragging_node = None
        self.hover_node = None
        self.hover_edge = None
        self.history = []

        # Edge curvature
        self.curvature = 50

        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.panning = False
        self.pan_start = None

        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
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

        # Update display
        self.update_mode_display()
        self.update_stats()
        self.redraw_all()

    def create_toolbar(self, parent):
        """Create toolbar with mode buttons."""
        toolbar = tk.Frame(parent, bg="#2c3e50", pady=8, padx=10)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Title
        title = tk.Label(toolbar, text="Spin Network Modifier", font=("Arial", 16, "bold"),
                        bg="#2c3e50", fg="white")
        title.pack(side=tk.LEFT, padx=10)

        # Utility buttons
        util_frame = tk.Frame(toolbar, bg="#2c3e50")
        util_frame.pack(side=tk.RIGHT, padx=10)

        load_btn = tk.Button(util_frame, text="Load Graph", width=12, height=1,
                            command=self.load_graph_dialog, fg="black",
                            font=("Arial", 10, "bold"), cursor="hand2")
        load_btn.grid(row=0, column=0, padx=3)

        undo_btn = tk.Button(util_frame, text="Undo (Z)", width=12, height=1,
                           command=self.undo, fg="black",
                           font=("Arial", 10, "bold"), cursor="hand2")
        undo_btn.grid(row=0, column=1, padx=3)

        save_btn = tk.Button(util_frame, text="Save & Exit", width=12, height=1,
                           command=self.save_graph, fg="black",
                           font=("Arial", 10, "bold"), cursor="hand2")
        save_btn.grid(row=0, column=2, padx=3)

        self.mode_buttons = {}

    def create_info_panel(self, parent):
        """Create the right info panel."""
        info_frame = tk.Frame(parent, bg="white", width=280, relief=tk.SUNKEN, bd=1)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.pack_propagate(False)

        # Mode info
        tk.Label(info_frame, text="Current Mode", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(10, 5))

        self.mode_label = tk.Label(info_frame, text="Add Node", font=("Arial", 14),
                                   bg="#ecf0f1", fg="#2c3e50", relief=tk.RAISED, bd=2,
                                   width=22, height=2)
        self.mode_label.pack(padx=10, pady=5)

        # Instructions
        tk.Label(info_frame, text="Instructions", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(20, 5))

        self.instructions = tk.Text(info_frame, height=14, width=30, wrap=tk.WORD,
                                   bg="#ecf0f1", fg="#2c3e50", font=("Arial", 9),
                                   relief=tk.FLAT, padx=10, pady=10, state=tk.DISABLED)
        self.instructions.pack(padx=10, pady=5)

        # Graph info
        tk.Label(info_frame, text="Graph Statistics", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(15, 5))

        self.stats_label = tk.Label(info_frame, text="Nodes: 0\nEdges: 0",
                                   font=("Arial", 11), bg="#ecf0f1", fg="#2c3e50",
                                   relief=tk.RAISED, bd=1, width=22, height=4,
                                   justify=tk.LEFT, padx=10)
        self.stats_label.pack(padx=10, pady=5)

        # Edge curvature slider
        tk.Label(info_frame, text="Edge Curvature", font=("Arial", 11, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(15, 5))

        self.curvature_slider = tk.Scale(info_frame, from_=0, to=150, orient=tk.HORIZONTAL,
                                        bg="white", highlightthickness=0, length=240,
                                        command=self.on_curvature_change)
        self.curvature_slider.set(50)
        self.curvature_slider.pack(padx=10, pady=5)

    def load_graph_dialog(self):
        """Open file dialog to load a .graphml file."""
        filename = tk.filedialog.askopenfilename(
            title="Select GraphML file",
            filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        if filename:
            self.load_graph(filename)

    # Loads a .graphml into self.graph and self.nodes. Handles three position
    # sources in priority order: stored x/y attrs, "(x, y)" string in "pos"
    # attr, or spring layout as fallback. Clears undo history on load.
    def load_graph(self, filepath):
        """Load a graph from a .graphml file."""
        try:
            loaded_graph = nx.read_graphml(filepath, force_multigraph=True)

            # Clear current graph
            self.graph.clear()
            self.nodes.clear()
            self.history.clear()

            # Copy nodes with positions
            for node in loaded_graph.nodes():
                node_id = int(node) if node.isdigit() else node

                # Try to get position from various possible attributes
                if 'x' in loaded_graph.nodes[node] and 'y' in loaded_graph.nodes[node]:
                    x = float(loaded_graph.nodes[node]['x'])
                    y = float(loaded_graph.nodes[node]['y'])
                elif 'pos' in loaded_graph.nodes[node]:
                    pos_str = loaded_graph.nodes[node]['pos']
                    if isinstance(pos_str, str):
                        # Parse string like "(x, y)"
                        pos_str = pos_str.strip('()').split(',')
                        x = float(pos_str[0])
                        y = float(pos_str[1])
                    else:
                        x, y = pos_str
                else:
                    # Use layout algorithm
                    pos = nx.spring_layout(loaded_graph, seed=42)
                    x, y = pos[node]
                    x = (x + 1) * 400 + 50  # Scale to canvas
                    y = (y + 1) * 300 + 25

                self.nodes[node_id] = (x, y)
                self.graph.add_node(node_id, pos=(x, y))

            # Copy edges with labels
            for u, v, key, data in loaded_graph.edges(keys=True, data=True):
                u_id = int(u) if (isinstance(u, str) and u.isdigit()) else u
                v_id = int(v) if (isinstance(v, str) and v.isdigit()) else v

                raw_label = data.get('label', '?')
                label = parse_spin_label(str(raw_label)) if raw_label != '?' else raw_label

                self.graph.add_edge(u_id, v_id, label=label, key=key)

            self.redraw_all()
            self.update_stats()

            print(f"✓ Loaded graph from {filepath}")
            print(f"  Nodes: {len(self.graph.nodes())}, Edges: {len(self.graph.edges())}")

        except Exception as e:
            tk.messagebox.showerror("Load Error", f"Failed to load graph:\n{e}")
            print(f"Error loading graph: {e}")

    def set_mode(self, mode):
        """Change the current editing mode."""
        self.mode = mode
        self.selected_node = None
        self.update_mode_display()
        self.redraw_all()

    def update_mode_display(self):
        """Update UI to reflect current mode."""
        mode_info = {
            "add_node": ("Add Node", "Click anywhere to add a node", "#27ae60"),
            "add_edge": ("Add Edge", "1. Click first node (turns blue)\n2. Click second node\n3. Enter spin value", "#3498db"),
            "move_node": ("Move Node", "Click and HOLD, then drag a node", "#f39c12"),
            "delete_node": ("Delete Node", "Click a node to delete it", "#95a5a6"),
            "delete_edge": ("Delete Edge", "Click near an edge to delete it\n(edges turn red on hover)", "#7f8c8d"),
        }

        title, instructions, color = mode_info[self.mode]
        self.mode_label.config(text=title, bg=color, fg="white")

        self.instructions.config(state=tk.NORMAL)
        self.instructions.delete(1.0, tk.END)
        self.instructions.insert(1.0, instructions + "\n\n" +
                               "Shortcuts:\n" +
                               "N - Add Node\n" +
                               "E - Add Edge\n" +
                               "M - Move Node\n" +
                               "D - Delete Node\n" +
                               "X - Delete Edge\n" +
                               "Z - Undo\n" +
                               "S - Save & Exit")
        self.instructions.config(state=tk.DISABLED)

    def update_stats(self):
        """Update graph statistics display."""
        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())

        # Count open edges
        open_edges = self.get_open_edges()

        self.stats_label.config(
            text=f"Nodes: {num_nodes}\nEdges: {num_edges}\nOpen Edges: {len(open_edges)}"
        )

    def get_open_edges(self):
        """
        Get all open edges in the graph.
        An open edge is one where at least one endpoint has degree < 3.
        Returns list of (node1, node2, key, label) tuples.
        """
        open_edges = []
        for n1, n2, key, data in self.graph.edges(keys=True, data=True):
            if self.graph.degree(n1) < 3 or self.graph.degree(n2) < 3:
                label = data.get('label', '?')
                open_edges.append((n1, n2, key, label))
        return open_edges

    def is_open_edge(self, n1, n2, key):
        """Check if an edge is open (connected to vertex with degree < 3)."""
        return self.graph.degree(n1) < 3 or self.graph.degree(n2) < 3

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        key = event.char.lower()
        if key == 'n':
            self.set_mode("add_node")
        elif key == 'e':
            self.set_mode("add_edge")
        elif key == 'm':
            self.set_mode("move_node")
        elif key == 'd':
            self.set_mode("delete_node")
        elif key == 'x':
            self.set_mode("delete_edge")
        elif key == 'z':
            self.undo()
        elif key == 's':
            self.save_graph()
        elif key == 'r':
            self.reset_view()

    def on_canvas_click(self, event):
        """Handle canvas click based on current mode."""
        x, y = event.x, event.y
        clicked_node = self.find_node_at(x, y)

        if self.mode == "add_node":
            if clicked_node is None:
                self.add_node(x, y)

        elif self.mode == "add_edge":
            if clicked_node is not None:
                if self.selected_node is None:
                    self.selected_node = clicked_node
                    self.redraw_all()
                elif self.selected_node == clicked_node:
                    self.selected_node = None
                    self.redraw_all()
                else:
                    self.attempt_add_edge(self.selected_node, clicked_node)
                    self.selected_node = None
                    self.redraw_all()

        elif self.mode == "delete_node":
            if clicked_node is not None:
                self.delete_node(clicked_node)

        elif self.mode == "delete_edge":
            edge = self.find_edge_at(x, y)
            if edge is not None:
                self.delete_edge(*edge)

        elif self.mode == "move_node":
            if clicked_node is not None:
                self.dragging_node = clicked_node

    def on_canvas_drag(self, event):
        """Handle canvas drag for moving nodes."""
        if self.mode == "move_node" and self.dragging_node is not None:
            wx, wy = self.screen_to_world(event.x, event.y)
            self.nodes[self.dragging_node] = (wx, wy)
            self.graph.nodes[self.dragging_node]['pos'] = (wx, wy)
            self.redraw_all()

    def on_canvas_release(self, event):
        """Handle mouse release."""
        if self.dragging_node is not None:
            self.save_state(f"Move node {self.dragging_node}")
            self.dragging_node = None

    def on_canvas_hover(self, event):
        """Handle mouse hover for visual feedback."""
        x, y = event.x, event.y
        hover_node = self.find_node_at(x, y)
        hover_edge = self.find_edge_at(x, y) if self.mode == "delete_edge" else None

        if hover_node != self.hover_node or hover_edge != self.hover_edge:
            self.hover_node = hover_node
            self.hover_edge = hover_edge
            self.redraw_all()

    def on_curvature_change(self, value):
        """Handle curvature slider change."""
        self.curvature = int(value)
        self.redraw_all()

    def find_node_at(self, x, y):
        """Find node at given screen coordinates."""
        wx, wy = self.screen_to_world(x, y)
        threshold = 15 / self.zoom_level
        for node_id, (nx, ny) in self.nodes.items():
            if (wx - nx)**2 + (wy - ny)**2 <= threshold**2:
                return node_id
        return None

    def find_edge_at(self, x, y):
        """Find edge near given screen coordinates."""
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
        """Calculate distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def add_node(self, x, y):
        """Add a new node at the specified screen position."""
        wx, wy = self.screen_to_world(x, y)
        node_id = max(self.nodes.keys(), default=0) + 1
        self.nodes[node_id] = (wx, wy)
        self.graph.add_node(node_id, pos=(wx, wy))
        self.save_state(f"Add node {node_id}")
        self.redraw_all()
        self.update_stats()
        print(f"✓ Node {node_id} added at ({wx:.0f}, {wy:.0f})")

    def delete_node(self, node_id):
        """Delete a node and all connected edges."""
        if node_id in self.nodes:
            self.save_state(f"Delete node {node_id}")
            del self.nodes[node_id]
            self.graph.remove_node(node_id)
            self.redraw_all()
            self.update_stats()
            print(f"✓ Node {node_id} deleted")

    # Validates degree ≤ 3, triangle inequality, prompts for label, adds edge.
    def attempt_add_edge(self, node1, node2):
        """Attempt to add an edge between two nodes."""
        if node1 == node2:
            tk.messagebox.showwarning("Invalid Edge", "Cannot create a self-loop!")
            return

        if self.graph.degree(node1) >= 3:
            tk.messagebox.showwarning("Invalid Edge", f"Node {node1} already has 3 edges!")
            return

        if self.graph.degree(node2) >= 3:
            tk.messagebox.showwarning("Invalid Edge", f"Node {node2} already has 3 edges!")
            return

        label = self.get_edge_label()
        if label is None:
            return

        self.graph.add_edge(node1, node2, label=label)

        # Check triangular conditions
        if isinstance(label, (int, float)):
            if not self.check_conditions(node1) or not self.check_conditions(node2):
                tk.messagebox.showerror(
                    "Triangular Condition Violated",
                    f"Edge with label {label} violates triangular inequality!\n" +
                    f"For edges j₁, j₂, j₃ at a node: |j₁-j₂| ≤ j₃ ≤ j₁+j₂"
                )
                self.graph.remove_edge(node1, node2)
                return

        self.save_state(f"Add edge {node1}-{node2}")
        self.redraw_all()
        self.update_stats()
        print(f"✓ Edge added: {node1} -- {label} -- {node2}")

    def delete_edge(self, node1, node2, key):
        """Delete a specific edge."""
        self.save_state(f"Delete edge {node1}-{node2}")
        label = self.graph.edges[node1, node2, key].get('label', '?')
        self.graph.remove_edge(node1, node2, key)
        self.redraw_all()
        self.update_stats()
        print(f"✓ Edge deleted: {node1} -- {label} -- {node2}")

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
        """Check triangular conditions at a node."""
        labels = []
        for neighbor, edges in self.graph[node].items():
            for key, edge_data in edges.items():
                if "label" in edge_data:
                    labels.append(edge_data["label"])

        if len(labels) == 3:
            return vertex_satisfies_triangular_conditions(labels)
        return True

    def redraw_all(self):
        """Redraw the entire graph."""
        self.canvas.delete("all")

        # Draw grid (transformed)
        grid_step = 50 * self.zoom_level
        start_x = self.pan_offset[0] % grid_step
        start_y = self.pan_offset[1] % grid_step
        for i in range(int(start_x), 900, max(1, int(grid_step))):
            self.canvas.create_line(i, 0, i, 650, fill="#e0e0e0", width=1)
        for i in range(int(start_y), 650, max(1, int(grid_step))):
            self.canvas.create_line(0, i, 900, i, fill="#e0e0e0", width=1)

        # Draw zoom indicator
        self.canvas.create_text(10, 10, text=f"Zoom: {self.zoom_level:.1f}x", anchor="nw",
                               font=("Arial", 9), fill="#7f8c8d")

        # Draw edges
        try:
            for n1, n2, key in self.graph.edges(keys=True):
                if n1 in self.nodes and n2 in self.nodes:
                    edge_data = self.graph.edges[n1, n2, key]
                    label = edge_data.get('label', '?')
                    is_hover = (self.hover_edge == (n1, n2, key))
                    is_open = self.is_open_edge(n1, n2, key)
                    self.draw_edge(n1, n2, key, label, is_hover, is_open)
        except Exception as e:
            print(f"Warning: Error drawing edges: {e}")

        # Draw nodes
        for node_id in list(self.nodes.keys()):
            if node_id in self.nodes:
                x, y = self.nodes[node_id]
                is_selected = (node_id == self.selected_node)
                is_hover = (node_id == self.hover_node)
                is_dragging = (node_id == self.dragging_node)
                self.draw_node(node_id, x, y, is_selected, is_hover, is_dragging)

    def draw_node(self, node_id, x, y, is_selected, is_hover, is_dragging):
        """Draw a single node."""
        sx, sy = self.world_to_screen(x, y)
        radius = 15 * self.zoom_level

        if is_dragging:
            fill = "#f39c12"
            outline = "#e67e22"
            width = 3
        elif is_selected:
            fill = "#3498db"
            outline = "#2980b9"
            width = 3
        elif is_hover:
            fill = "#ecf0f1"
            outline = "#3498db"
            width = 2
        else:
            fill = "#ecf0f1"
            outline = "#34495e"
            width = 2

        self.canvas.create_oval(sx-radius, sy-radius, sx+radius, sy+radius,
                               fill=fill, outline=outline, width=width)
        font_size = max(8, int(11 * self.zoom_level))
        self.canvas.create_text(sx, sy, text=str(node_id), font=("Arial", font_size, "bold"),
                               fill="#2c3e50")

    def draw_edge(self, node1, node2, key, label, is_hover, is_open):
        """Draw a single edge."""
        wx1, wy1 = self.nodes[node1]
        wx2, wy2 = self.nodes[node2]
        x1, y1 = self.world_to_screen(wx1, wy1)
        x2, y2 = self.world_to_screen(wx2, wy2)

        if is_hover:
            color = "#e74c3c"
            width = 3
        elif is_open:
            color = "#f39c12"
            width = 2
        else:
            color = "#34495e"
            width = 2

        num_edges = self.graph.number_of_edges(node1, node2)

        if num_edges > 1:
            edge_keys = [k for (n1, n2, k) in self.graph.edges(keys=True) if {n1, n2} == {node1, node2}]
            edge_index = edge_keys.index(key)
            offset = (edge_index - (num_edges - 1) / 2) * self.curvature * self.zoom_level
            self.draw_curved_edge(x1, y1, x2, y2, offset, label, color, width)
        else:
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
            font_size = max(8, int(10 * self.zoom_level))
            bbox = self.canvas.bbox(self.canvas.create_text(lx, ly, text=str(label), font=("Arial", font_size, "bold")))
            if bbox:
                self.canvas.create_rectangle(bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2, fill="#f5f5f5", outline="")
            self.canvas.create_text(lx, ly, text=str(label), font=("Arial", font_size, "bold"),
                                   fill="#c0392b")

    def draw_curved_edge(self, x1, y1, x2, y2, offset, label, color, width):
        """Draw a curved edge (coordinates already in screen space)."""
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = y2 - y1, x1 - x2
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx / length, dy / length

        cx = mx + offset * dx
        cy = my + offset * dy

        self.canvas.create_line(x1, y1, cx, cy, x2, y2, smooth=True,
                               fill=color, width=width)

        lx = (x1 + x2 + 2 * cx) / 4
        ly = (y1 + y2 + 2 * cy) / 4
        font_size = max(8, int(10 * self.zoom_level))
        bbox = self.canvas.bbox(self.canvas.create_text(lx, ly, text=str(label), font=("Arial", font_size, "bold")))
        if bbox:
            self.canvas.create_rectangle(bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2, fill="#f5f5f5", outline="")
        self.canvas.create_text(lx, ly, text=str(label), font=("Arial", 10, "bold"),
                               fill="#c0392b")

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

    def save_state(self, action):
        """Save current state for undo."""
        state = {
            'graph': self.graph.copy(),
            'nodes': self.nodes.copy(),
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
        self.selected_node = None
        self.redraw_all()
        self.update_stats()
        print(f"↶ Undid: {state['action']}")

    # Serialises to self.input_file if set, otherwise prompts with a file dialog.
    # Converts tuple attributes (e.g. "pos") to strings for GraphML compatibility.
    def save_graph(self):
        """Save graph to GraphML file (overwrites original if loaded from file)."""
        if len(self.graph.nodes()) == 0:
            tk.messagebox.showwarning("Empty Graph", "Cannot save empty graph!")
            return

        # Use original file if available, otherwise ask for filename
        if self.input_file:
            filename = self.input_file
        else:
            filename = tk.filedialog.asksaveasfilename(
                title="Save GraphML file",
                defaultextension=".graphml",
                filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
                initialfile="modified_graph.graphml"
            )
            if not filename:
                return

        # Create copy and convert attributes
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

            tk.messagebox.showinfo(
                "Success",
                f"Graph saved to:\n{filename}\n\n" +
                f"Nodes: {len(self.graph.nodes())}\n" +
                f"Edges: {len(self.graph.edges())}"
            )
            print(f"✓ Graph saved successfully to {filename}")
            self.master.destroy()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save graph:\n{e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modify an existing spin network graph")
    parser.add_argument("input_file", nargs="?", help="Input .graphml file to load")
    args = parser.parse_args()

    root = tk.Tk()
    modifier = GraphModifier(root, args.input_file)
    root.mainloop()
