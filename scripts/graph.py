import tkinter as tk
import tkinter.simpledialog
import tkinter.messagebox
import networkx as nx
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import vertex_satisfies_triangular_conditions, parse_spin_label
from src.orientation import set_reference_orientation, calculate_layout_phase


# -----------------------------------------------------------------------
# Interactive spin network editor (tkinter GUI)
# -----------------------------------------------------------------------
#
# First step in the norm-computation workflow:
#   graph.py → compute_norm.py → evaluate_norm.py
#
# The user draws a trivalent graph by placing nodes and edges, then saves it
# as drawn_graph.graphml. The editor enforces:
#   - max degree 3 per node (trivalency)
#   - triangle inequality at each completed vertex (when all 3 edges are set)
#   - integer or half-integer edge labels
#
# Coordinate systems:
#   World coordinates  – stored in self.nodes and graph "pos" attributes
#   Screen coordinates – world * zoom_level + pan_offset (used for drawing)
#
# Keyboard shortcuts: N add node | E add edge | M move | D delete node |
#                     X delete edge | Z undo | S save & exit | R reset view
class GraphEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Spin Network Graph Editor")

        # Create main frame
        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create toolbar
        self.create_toolbar(main_frame)

        # Create canvas
        self.canvas = tk.Canvas(main_frame, width=900, height=650, bg="#f5f5f5", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create info panel
        self.create_info_panel(main_frame)

        # Graph data
        self.graph = nx.MultiGraph()
        self.graph.graph['phase'] = complex(1)  # cumulative orientation phase
        self.nodes = {}  # {node_id: (x, y)}
        self.node_graphics = {}  # {node_id: [oval_id, text_id]}
        self.edge_graphics = {}  # {(node1, node2, key): [line_ids, label_id]}

        # Editor state
        self.mode = "add_node"  # Modes: add_node, add_edge, delete_node, delete_edge, move_node
        self.selected_node = None
        self.dragging_node = None
        self.hover_node = None
        self.hover_edge = None
        self.history = []  # Undo history

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

        # Zoom bindings (mouse wheel)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows/Mac
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down

        # Pan bindings (middle mouse button or Shift+drag)
        self.canvas.bind("<Button-2>", self.on_pan_start)      # Middle click
        self.canvas.bind("<B2-Motion>", self.on_pan_motion)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.canvas.bind("<Shift-Button-1>", self.on_pan_start)
        self.canvas.bind("<Shift-B1-Motion>", self.on_pan_motion)
        self.canvas.bind("<Shift-ButtonRelease-1>", self.on_pan_end)

        # Keyboard shortcuts
        self.master.bind("<Key>", self.on_key_press)

        # Update mode display and draw initial canvas
        self.update_mode_display()
        self.update_stats()
        self.redraw_all()

    def create_toolbar(self, parent):
        """Create a minimal toolbar with just title and essential buttons."""
        toolbar = tk.Frame(parent, bg="#2c3e50", pady=8, padx=10)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Title
        title = tk.Label(toolbar, text="Spin Network Editor", font=("Arial", 16, "bold"),
                        bg="#2c3e50", fg="white")
        title.pack(side=tk.LEFT, padx=10)

        # Just essential utility buttons on the right
        util_frame = tk.Frame(toolbar, bg="#2c3e50")
        util_frame.pack(side=tk.RIGHT, padx=10)

        undo_btn = tk.Button(util_frame, text="↶ Undo (Z)", width=12, height=1,
                           command=self.undo, fg="black",
                           font=("Arial", 10, "bold"), cursor="hand2")
        undo_btn.grid(row=0, column=0, padx=3)

        clear_btn = tk.Button(util_frame, text="Clear All", width=12, height=1,
                            command=self.clear_all, fg="black",
                            font=("Arial", 10, "bold"), cursor="hand2")
        clear_btn.grid(row=0, column=1, padx=3)

        save_btn = tk.Button(util_frame, text="Save & Exit", width=12, height=1,
                           command=self.save_graph, fg="black",
                           font=("Arial", 10, "bold"), cursor="hand2")
        save_btn.grid(row=0, column=2, padx=3)

        # Store empty dict for mode_buttons to avoid errors
        self.mode_buttons = {}

    def create_info_panel(self, parent):
        """Create the right info panel."""
        info_frame = tk.Frame(parent, bg="white", width=250, relief=tk.SUNKEN, bd=1)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.pack_propagate(False)

        # Mode info
        tk.Label(info_frame, text="Current Mode", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(10, 5))

        self.mode_label = tk.Label(info_frame, text="Add Node", font=("Arial", 14),
                                   bg="#ecf0f1", fg="#2c3e50", relief=tk.RAISED, bd=2,
                                   width=20, height=2)
        self.mode_label.pack(padx=10, pady=5)

        # Instructions
        tk.Label(info_frame, text="Instructions", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(20, 5))

        self.instructions = tk.Text(info_frame, height=15, width=28, wrap=tk.WORD,
                                   bg="#ecf0f1", fg="#2c3e50", font=("Arial", 10),
                                   relief=tk.FLAT, padx=10, pady=10, state=tk.DISABLED)
        self.instructions.pack(padx=10, pady=5)

        # Graph info
        tk.Label(info_frame, text="Graph Statistics", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(20, 5))

        self.stats_label = tk.Label(info_frame, text="Nodes: 0\nEdges: 0",
                                   font=("Arial", 11), bg="#ecf0f1", fg="#2c3e50",
                                   relief=tk.RAISED, bd=1, width=20, height=3,
                                   justify=tk.LEFT, padx=10)
        self.stats_label.pack(padx=10, pady=5)

        # Graph phase display
        tk.Label(info_frame, text="Graph Phase", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(20, 5))

        self.phase_label = tk.Label(info_frame, text="phase: N/A",
                                    font=("Arial", 11), bg="#ecf0f1", fg="#2c3e50",
                                    relief=tk.RAISED, bd=1, width=20, height=2)
        self.phase_label.pack(padx=10, pady=5)

        # Edge curvature slider
        tk.Label(info_frame, text="Edge Curvature", font=("Arial", 11, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(20, 5))

        self.curvature_slider = tk.Scale(info_frame, from_=0, to=150, orient=tk.HORIZONTAL,
                                        bg="white", highlightthickness=0, length=200,
                                        command=self.on_curvature_change)
        self.curvature_slider.set(50)
        self.curvature_slider.pack(padx=10, pady=5)

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
            "add_edge": ("Add Edge", "1. Click first node (it will turn blue)\n2. Click second node\n3. Enter spin value", "#3498db"),
            "move_node": ("Move Node", "Click and HOLD, then drag a node to move it", "#f39c12"),
            "delete_node": ("Delete Node", "Click a node to delete it.\nConnected edges will be removed.", "#95a5a6"),
            "delete_edge": ("Delete Edge", "Click near an edge to delete it\n(edges turn red when hovering)", "#7f8c8d"),
        }

        title, instructions, color = mode_info[self.mode]
        self.mode_label.config(text=title, bg=color, fg="white")

        self.instructions.config(state=tk.NORMAL)
        self.instructions.delete(1.0, tk.END)
        self.instructions.insert(1.0, instructions + "\n\n" +
                               "Keyboard Shortcuts:\n" +
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
        self.stats_label.config(text=f"Nodes: {num_nodes}\nEdges: {num_edges}")

    def _recompute_phase(self):
        """Recompute graph phase from current node positions and store it."""
        layout = {v: self.nodes[v] for v in self.graph.nodes() if v in self.nodes}
        phase = calculate_layout_phase(self.graph, layout)
        self.graph.graph['phase'] = phase
        self.update_phase_display()

    def update_phase_display(self):
        """Refresh the phase label in the info panel."""
        phase = self.graph.graph.get('phase', None)
        # Only show a numeric value if at least one vertex has a reference orientation.
        has_ref = any(
            self.graph.nodes[v].get('reference_orientation') is not None
            for v in self.graph.nodes()
        )
        if phase is None or not has_ref:
            text = "phase: N/A"
            color = "#ecf0f1"
        else:
            val = phase.real
            text = f"phase: {val:+.0f}"
            color = "#d5f5e3" if val > 0 else "#fadbd8"
        self.phase_label.config(text=text, bg=color)

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
                    # Clicked same node twice - deselect
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
                self.save_state(f"Move node {clicked_node}")  # before any drag
                self.dragging_node = clicked_node

    def on_canvas_drag(self, event):
        """Handle canvas drag for moving nodes."""
        if self.mode == "move_node" and self.dragging_node is not None:
            wx, wy = self.screen_to_world(event.x, event.y)
            self.nodes[self.dragging_node] = (wx, wy)
            self.graph.nodes[self.dragging_node]['pos'] = (wx, wy)
            self._recompute_phase()  # live phase update
            self.redraw_all()

    def on_canvas_release(self, event):
        """Handle mouse release."""
        self.dragging_node = None  # state was already saved at drag-start

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
        # Convert screen coordinates to world coordinates
        wx, wy = self.screen_to_world(x, y)
        threshold = 15 / self.zoom_level  # Adjust threshold for zoom
        for node_id, (nx, ny) in self.nodes.items():
            if (wx - nx)**2 + (wy - ny)**2 <= threshold**2:
                return node_id
        return None

    def find_edge_at(self, x, y):
        """Find edge near given screen coordinates."""
        # Convert to world coordinates
        wx, wy = self.screen_to_world(x, y)
        threshold = 10 / self.zoom_level
        for n1, n2, key in self.graph.edges(keys=True):
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
        # Convert screen coordinates to world coordinates
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

    # Validates degree constraints (≤ 3) and triangle inequality, prompts for
    # a label, then adds the edge. Rolls back if conditions are violated.
    def attempt_add_edge(self, node1, node2):
        """Attempt to add an edge between two nodes."""
        if node1 == node2:
            tk.messagebox.showwarning("Invalid Edge", "Cannot create a self-loop!")
            return

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

        # Temporarily add edge
        self.graph.add_edge(node1, node2, label=label)

        # Check triangular conditions for numeric labels
        if isinstance(label, (int, float)):
            if not self.check_conditions(node1) or not self.check_conditions(node2):
                tk.messagebox.showerror("Triangular Condition Violated",
                    f"Edge with label {label} violates triangular inequality!\n" +
                    f"For edges j₁, j₂, j₃ at a node: |j₁-j₂| ≤ j₃ ≤ j₁+j₂")
                self.graph.remove_edge(node1, node2)
                return

        self.save_state(f"Add edge {node1}-{node2}")
        # Fix reference orientation the moment a vertex becomes trivalent.
        for node in (node1, node2):
            if self.graph.degree(node) == 3:
                set_reference_orientation(self.graph, node)
        self._recompute_phase()
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

        # Validate numeric labels: must be integer or half-integer
        if isinstance(label, (int, float)):
            if label * 2 != int(label * 2):
                tk.messagebox.showwarning("Invalid Label", "Numeric spin must be integer or half-integer!")
                return None

        return label

    # Returns True if the node's edge labels satisfy the triangle inequality.
    # Only checked when all 3 edges are present (degree == 3).
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
        zoom_text = f"Zoom: {self.zoom_level:.1f}x"
        self.canvas.create_text(10, 10, text=zoom_text, anchor="nw",
                               font=("Arial", 9), fill="#7f8c8d")

        # Draw edges first (so nodes appear on top)
        try:
            for n1, n2, key in self.graph.edges(keys=True):
                if n1 in self.nodes and n2 in self.nodes:
                    edge_data = self.graph.edges[n1, n2, key]
                    label = edge_data.get('label', '?')
                    is_hover = (self.hover_edge == (n1, n2, key))
                    self.draw_edge(n1, n2, key, label, is_hover)
        except Exception as e:
            print(f"Warning: Error drawing edges: {e}")

        # Draw nodes on top
        for node_id in list(self.nodes.keys()):
            if node_id in self.nodes:
                x, y = self.nodes[node_id]
                is_selected = (node_id == self.selected_node)
                is_hover = (node_id == self.hover_node)
                is_dragging = (node_id == self.dragging_node)
                self.draw_node(node_id, x, y, is_selected, is_hover, is_dragging)

    def draw_node(self, node_id, x, y, is_selected, is_hover, is_dragging):
        """Draw a single node."""
        # Transform to screen coordinates
        sx, sy = self.world_to_screen(x, y)
        radius = 15 * self.zoom_level

        # Determine colors
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

    def draw_edge(self, node1, node2, key, label, is_hover):
        """Draw a single edge."""
        # Get world coordinates and transform to screen
        wx1, wy1 = self.nodes[node1]
        wx2, wy2 = self.nodes[node2]
        x1, y1 = self.world_to_screen(wx1, wy1)
        x2, y2 = self.world_to_screen(wx2, wy2)

        # Determine color and width
        color = "#e74c3c" if is_hover else "#34495e"
        width = 3 if is_hover else 2

        # Check for parallel edges
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
        self.canvas.create_text(lx, ly, text=str(label), font=("Arial", font_size, "bold"),
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
        # Get mouse position for zoom center
        mx, my = event.x, event.y

        # Determine zoom direction
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = 1.1
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = 0.9
        else:
            return

        # Limit zoom range
        new_zoom = self.zoom_level * factor
        if new_zoom < 0.2 or new_zoom > 5.0:
            return

        # Adjust pan to keep mouse position fixed
        # Before zoom: screen = world * old_zoom + old_pan
        # After zoom:  screen = world * new_zoom + new_pan
        # We want screen to stay the same at mouse position
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
        if len(self.history) > 50:  # Limit history
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

    def clear_all(self):
        """Clear the entire graph."""
        if tk.messagebox.askyesno("Clear All", "Delete all nodes and edges?"):
            self.save_state("Clear all")
            self.graph.clear()
            self.nodes.clear()
            self.selected_node = None
            self.redraw_all()
            self.update_stats()
            print("✓ Graph cleared")

    # Serialises the graph to drawn_graph.graphml in the current directory.
    # Tuple-valued node attributes (e.g. "pos") are converted to strings
    # because the GraphML format does not support tuples natively.
    def save_graph(self):
        """Save graph to GraphML file."""
        if len(self.graph.nodes()) == 0:
            tk.messagebox.showwarning("Empty Graph", "Cannot save empty graph!")
            return

        # Create copy and convert attributes
        graph_copy = self.graph.copy()

        # Graph-level attributes: phase (complex) → string
        if 'phase' in graph_copy.graph:
            graph_copy.graph['phase'] = str(graph_copy.graph['phase'])

        for node, attrs in graph_copy.nodes(data=True):
            for key, value in list(attrs.items()):
                if isinstance(value, (tuple, list)):
                    graph_copy.nodes[node][key] = str(value)

        for u, v, k, attrs in graph_copy.edges(keys=True, data=True):
            for attr_key, value in list(attrs.items()):
                if isinstance(value, tuple):
                    graph_copy.edges[u, v, k][attr_key] = str(value)
                elif hasattr(value, 'free_symbols'):   # sympy expression
                    graph_copy.edges[u, v, k][attr_key] = str(value)

        try:
            nx.write_graphml(graph_copy, "drawn_graph.graphml")
            tk.messagebox.showinfo("Success",
                "Graph saved to 'drawn_graph.graphml'\n\n" +
                f"Nodes: {len(self.graph.nodes())}\n" +
                f"Edges: {len(self.graph.edges())}")
            print("✓ Graph saved successfully")
            self.master.destroy()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to save graph:\n{e}")


# Create and run the editor
if __name__ == "__main__":
    root = tk.Tk()
    editor = GraphEditor(root)
    root.mainloop()
