#!/usr/bin/env python3
"""
Inspect Graph - Visual graph viewer with open edge highlighting

This tool displays a spin network graph in an interactive canvas,
highlighting open edges and nodes for easy inspection.

Usage:
    python inspect_graph.py graph.graphml
"""

import tkinter as tk
import tkinter.filedialog
import networkx as nx
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# -----------------------------------------------------------------------
# Read-only spin network viewer (tkinter GUI)
# -----------------------------------------------------------------------
#
# Useful for quickly inspecting a .graphml file before running compute_norm.py.
# Highlights open nodes (degree < 3) and open edges in orange so you can
# identify the external legs of the network.
# No editing; use modify_graph.py for modifications.
#
# Keyboard shortcuts: L load | R reset view | Q quit
class GraphInspector:
    def __init__(self, master, input_file=None):
        self.master = master
        self.master.title("Spin Network Graph Inspector")

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
        self.nodes = {}
        self.hover_node = None
        self.hover_edge = None

        # Edge curvature
        self.curvature = 50

        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.panning = False
        self.pan_start = None

        # Bind events
        self.canvas.bind("<Motion>", self.on_canvas_hover)

        # Zoom bindings
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)

        # Pan bindings
        self.canvas.bind("<Button-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_motion)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.canvas.bind("<Button-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_pan_end)

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

        title = tk.Label(toolbar, text="Graph Inspector", font=("Arial", 16, "bold"),
                        bg="#2c3e50", fg="white")
        title.pack(side=tk.LEFT, padx=10)

        util_frame = tk.Frame(toolbar, bg="#2c3e50")
        util_frame.pack(side=tk.RIGHT, padx=10)

        load_btn = tk.Button(util_frame, text="Load Graph", width=12, height=1,
                            command=self.load_graph_dialog, fg="black",
                            font=("Arial", 10, "bold"), cursor="hand2")
        load_btn.grid(row=0, column=0, padx=3)

        quit_btn = tk.Button(util_frame, text="Quit (Q)", width=12, height=1,
                            command=self.master.destroy, fg="black",
                            font=("Arial", 10, "bold"), cursor="hand2")
        quit_btn.grid(row=0, column=1, padx=3)

    def create_info_panel(self, parent):
        """Create info panel."""
        info_frame = tk.Frame(parent, bg="white", width=280, relief=tk.SUNKEN, bd=1)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.pack_propagate(False)

        tk.Label(info_frame, text="Instructions", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(10, 5))

        self.instructions = tk.Text(info_frame, height=8, width=30, wrap=tk.WORD,
                                   bg="#ecf0f1", fg="#2c3e50", font=("Arial", 9),
                                   relief=tk.FLAT, padx=10, pady=10, state=tk.DISABLED)
        self.instructions.pack(padx=10, pady=5)

        tk.Label(info_frame, text="Graph Statistics", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(15, 5))

        self.stats_label = tk.Label(info_frame, text="", font=("Arial", 10),
                                   bg="#ecf0f1", fg="#2c3e50", relief=tk.RAISED,
                                   bd=1, width=30, height=5, justify=tk.LEFT, padx=10)
        self.stats_label.pack(padx=10, pady=5)

        tk.Label(info_frame, text="Open Edges", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(15, 5))

        self.open_edges_text = tk.Text(info_frame, height=12, width=30, wrap=tk.WORD,
                                       bg="#fff3cd", fg="#856404", font=("Arial", 9),
                                       relief=tk.RAISED, bd=1, padx=5, state=tk.DISABLED)
        self.open_edges_text.pack(padx=10, pady=5)

        tk.Label(info_frame, text="Hover Info", font=("Arial", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(15, 5))

        self.hover_label = tk.Label(info_frame, text="Hover over nodes/edges",
                                    font=("Arial", 9), bg="#d1ecf1", fg="#0c5460",
                                    relief=tk.RAISED, bd=1, width=30, height=4,
                                    justify=tk.LEFT, padx=5, anchor="nw")
        self.hover_label.pack(padx=10, pady=5)

        # Edge curvature slider
        tk.Label(info_frame, text="Edge Curvature", font=("Arial", 11, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(15, 5))

        self.curvature_slider = tk.Scale(info_frame, from_=0, to=150, orient=tk.HORIZONTAL,
                                        bg="white", highlightthickness=0, length=240,
                                        command=self.on_curvature_change)
        self.curvature_slider.set(50)
        self.curvature_slider.pack(padx=10, pady=5)

    def load_graph_dialog(self):
        """Open file dialog to load graph."""
        filename = tk.filedialog.askopenfilename(
            title="Select GraphML file",
            filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        if filename:
            self.load_graph(filename)

    # Parses the GraphML file, converts integer node IDs, coerces labels to
    # float, and falls back to spring layout when no stored positions exist.
    def load_graph(self, filepath):
        """Load a graph from .graphml file."""
        try:
            loaded_graph = nx.read_graphml(filepath, force_multigraph=True)

            self.graph.clear()
            self.nodes.clear()

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

            self.master.title(f"Graph Inspector - {os.path.basename(filepath)}")
            self.update_display()
            self.redraw_all()

            print(f"\n{'='*60}")
            print(f"LOADED: {filepath}")
            print(f"{'='*60}")
            print(f"Nodes: {len(self.graph.nodes())}")
            print(f"Edges: {len(self.graph.edges())}")
            print(f"{'='*60}\n")

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

    def on_canvas_hover(self, event):
        """Handle mouse hover."""
        x, y = event.x, event.y

        hover_node = self.find_node_at(x, y)
        hover_edge = self.find_edge_at(x, y)

        if hover_node != self.hover_node or hover_edge != self.hover_edge:
            self.hover_node = hover_node
            self.hover_edge = hover_edge
            self.update_hover_info()
            self.redraw_all()

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

    def on_curvature_change(self, value):
        """Handle curvature slider change."""
        self.curvature = int(value)
        self.redraw_all()

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        key = event.char.lower()
        if key == 'q':
            self.master.destroy()
        elif key == 'l':
            self.load_graph_dialog()
        elif key == 'r':
            self.reset_view()

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

    def update_display(self):
        """Update all display elements."""
        # Instructions
        self.instructions.config(state=tk.NORMAL)
        self.instructions.delete(1.0, tk.END)
        self.instructions.insert(1.0,
            "Visual inspection of spin network\n\n" +
            "Orange = Open nodes/edges\n" +
            "(degree < 3)\n\n" +
            "Shortcuts:\n" +
            "L - Load graph\n" +
            "R - Reset view\n" +
            "Q - Quit\n\n" +
            "Zoom: Mouse wheel\n" +
            "Pan: Click + drag"
        )
        self.instructions.config(state=tk.DISABLED)

        # Stats
        open_nodes = self.get_open_nodes()
        open_edges = self.get_open_edges()
        stats_text = (
            f"Nodes: {len(self.graph.nodes())}\n" +
            f"Edges: {len(self.graph.edges())}\n" +
            f"Open Nodes: {len(open_nodes)}\n" +
            f"Open Edges: {len(open_edges)}"
        )
        self.stats_label.config(text=stats_text)

        # Open edges list
        self.open_edges_text.config(state=tk.NORMAL)
        self.open_edges_text.delete(1.0, tk.END)
        open_edges = self.get_open_edges()
        if open_edges:
            for n1, n2, key, label in open_edges:
                deg1 = self.graph.degree(n1)
                deg2 = self.graph.degree(n2)
                self.open_edges_text.insert(tk.END, f"{n1}--[{label}]--{n2}\n")
                self.open_edges_text.insert(tk.END, f"  (deg: {deg1}, {deg2})\n\n")
        else:
            self.open_edges_text.insert(tk.END, "No open edges")
        self.open_edges_text.config(state=tk.DISABLED)

    def update_hover_info(self):
        """Update hover info display."""
        if self.hover_node is not None:
            node = self.hover_node
            degree = self.graph.degree(node)
            is_open = "Yes" if self.is_open_node(node) else "No"

            # Get connected edges
            edges = []
            for neighbor, edge_dict in self.graph[node].items():
                for key, data in edge_dict.items():
                    label = data.get('label', '?')
                    edges.append(f"  --[{label}]-- {neighbor}")

            edges_str = "\n".join(edges[:5])  # Limit to 5
            if len(edges) > 5:
                edges_str += f"\n  ... and {len(edges)-5} more"

            self.hover_label.config(text=f"Node {node}\nDegree: {degree}\nOpen: {is_open}\n{edges_str}")

        elif self.hover_edge is not None:
            n1, n2, key = self.hover_edge
            edge_data = self.graph.edges[n1, n2, key]
            label = edge_data.get('label', '?')
            deg1 = self.graph.degree(n1)
            deg2 = self.graph.degree(n2)
            is_open = "Yes" if self.is_open_edge(n1, n2, key) else "No"

            self.hover_label.config(text=f"Edge: {n1}--{n2}\nLabel: {label}\nDegrees: {deg1}, {deg2}\nOpen: {is_open}")
        else:
            self.hover_label.config(text="Hover over nodes/edges")

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

                    self.draw_edge(n1, n2, key, label, is_hover, is_open)
        except Exception as e:
            print(f"Warning: Error drawing edges: {e}")

        # Draw nodes
        for node_id in list(self.nodes.keys()):
            if node_id in self.nodes:
                wx, wy = self.nodes[node_id]
                is_open = self.is_open_node(node_id)
                is_hover = (node_id == self.hover_node)
                self.draw_node(node_id, wx, wy, is_open, is_hover)

        # Display zoom level
        zoom_text = f"Zoom: {self.zoom_level:.1f}x (R to reset)"
        self.canvas.create_text(10, 10, anchor="nw", text=zoom_text,
                               font=("Arial", 9), fill="#666666")

    def draw_node(self, node_id, wx, wy, is_open=False, is_hover=False):
        """Draw a node (wx, wy are world coordinates)."""
        # Transform to screen coordinates
        sx, sy = self.world_to_screen(wx, wy)
        radius = 12 * self.zoom_level

        # Determine colors based on state
        if is_hover:
            fill = "#3498db"  # Blue for hover
            outline = "#2980b9"
            width = 4
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
                               fill="white" if (is_hover or is_open) else "#2c3e50")

    def draw_edge(self, node1, node2, key, label, is_hover, is_open):
        """Draw an edge (uses world coordinates internally)."""
        wx1, wy1 = self.nodes[node1]
        wx2, wy2 = self.nodes[node2]

        # Transform to screen coordinates
        sx1, sy1 = self.world_to_screen(wx1, wy1)
        sx2, sy2 = self.world_to_screen(wx2, wy2)

        # Determine color
        if is_hover:
            color = "#3498db"  # Blue for hover
            width = 4
        elif is_open:
            color = "#f39c12"  # Orange for open
            width = 3
        else:
            color = "#34495e"  # Default
            width = 2

        # Check for parallel edges
        num_edges = self.graph.number_of_edges(node1, node2)

        if num_edges > 1:
            edge_keys = [k for (n1, n2, k) in self.graph.edges(keys=True) if {n1, n2} == {node1, node2}]
            edge_index = edge_keys.index(key)
            offset = (edge_index - (num_edges - 1) / 2) * self.curvature * self.zoom_level
            self.draw_curved_edge(sx1, sy1, sx2, sy2, offset, label, color, width)
        else:
            # Straight edge
            self.canvas.create_line(sx1, sy1, sx2, sy2, fill=color, width=width)

            lx, ly = (sx1 + sx2) / 2, (sy1 + sy2) / 2
            font_size = max(7, int(10 * self.zoom_level))
            bbox = self.canvas.bbox(self.canvas.create_text(lx, ly, text=str(label), font=("Arial", font_size, "bold")))
            if bbox:
                self.canvas.create_rectangle(bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2, fill="#f5f5f5", outline="")
            self.canvas.create_text(lx, ly, text=str(label), font=("Arial", font_size, "bold"), fill="#c0392b")

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
        font_size = max(7, int(10 * self.zoom_level))
        bbox = self.canvas.bbox(self.canvas.create_text(lx, ly, text=str(label), font=("Arial", font_size, "bold")))
        if bbox:
            self.canvas.create_rectangle(bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2, fill="#f5f5f5", outline="")
        self.canvas.create_text(lx, ly, text=str(label), font=("Arial", font_size, "bold"),
                               fill="#c0392b")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect a spin network graph")
    parser.add_argument("input_file", nargs="?", help="Input .graphml file")
    args = parser.parse_args()

    root = tk.Tk()
    inspector = GraphInspector(root, args.input_file)
    root.mainloop()
