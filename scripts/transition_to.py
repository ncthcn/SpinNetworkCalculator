#!/usr/bin/env python3
"""
Transition Tool - Interactive tool for spin network state transitions

This script loads a graph and allows you to:
1. Add extra edges to the graph (shown in green, flagged as "added")
2. Connect open edges together (reconnection)

The probability formula is:
    P = [Δ(c₁)⋯Δ(cₙ) / Θ(c₁,s₁,t₁)⋯Θ(cₙ,sₙ,tₙ)] × [1/||subgraph||] × [||new_graph|| / ||old_graph||]

Where:
- cᵢ = newly created open edge from reconnection
- sᵢ, tᵢ = the two edges reconnected to create cᵢ (at least one from old graph's open ends)
- ||subgraph|| = 1 if only reconnecting existing open ends (no added edges)
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
from src.gluer import glue_open_edges
from src.graph_reducer import reduce_all_cycles
from src.norm_reducer import canonicalise_terms, apply_kroneckers, expand_6j_symbolic
from src.spin_evaluator import evaluate_spin_network, SpinNetworkEvaluator
from src.LaTeX_rendering import terms_to_formula_string, save_formula_txt, _sanitize_py


# -----------------------------------------------------------------------
# Transition probability tool (tkinter GUI)
# -----------------------------------------------------------------------
#
# Combines graph editing with in-place norm computation to produce a
# transition probability P between two spin network states.
#
# Workflow:
#   1. Load original graph G1 (via CLI arg or Load button)
#   2. Optionally add new edges (shown in green, flagged as "added")
#   3. Select two open nodes and press C to mark a reconnection
#   4. Press S (Compute) — the tool:
#       a. Saves the modified graph to transition_to_graph.graphml
#       b. Computes ||G1|| and ||G2|| using the full norm pipeline
#       c. Computes ||subgraph|| (norm of added-edges subgraph; 1 if none)
#       d. Computes Δ and Θ products from the reconnection labels
#       e. Evaluates  P = |Δ/Θ × (1/||subgraph||) × ||G2||/||G1|||
#       f. Saves all data to transition_to_graph_transition.json
#
# Open nodes/edges are highlighted in orange; added edges in green.
# Keyboard shortcuts: N add node | E add edge | C reconnect selected |
#                     Z undo | S compute | R reset view | Esc → select mode
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

        save_btn = tk.Button(util_frame, text="Compute (S)", width=12, height=1,
                           command=self.save_and_compute, fg="black",
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

    def _has_symbolic_labels(self):
        """Return True if any edge in the current graph has a non-numeric label."""
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            label = data.get('label')
            if label is not None and label != '?' and not is_numeric_label(label):
                return True
        return False

    def symbolic_norm_from_graph(self, graph):
        """Run the symbolic pipeline and return canonical terms (no numerical eval)."""
        glued_graph = glue_open_edges(graph)
        terms = reduce_all_cycles(glued_graph, animator=None)

        clean_terms = []
        for T in terms:
            t = apply_kroneckers(T)
            if t is not None:
                clean_terms.append(t)

        for term in clean_terms:
            new = []
            for c in term["coeffs"]:
                if isinstance(c, dict) and c.get("type") == "6j":
                    new.extend(expand_6j_symbolic(c))
                else:
                    new.append(c)
            term["coeffs"] = new

        return canonicalise_terms(clean_terms)

    def _apply_reconnection(self, base_graph, old_edges, new_label):
        """Return a copy of base_graph with the reconnection applied (new_label on new open edge)."""
        g = base_graph.copy()
        e0, e1 = old_edges[0], old_edges[1]
        n0_open, n0_other = e0['nodes']
        n1_open, n1_other = e1['nodes']
        label0, label1 = e0['label'], e1['label']

        # Remove old open edges
        for open_n, other_n, lbl in [(n0_open, n0_other, label0), (n1_open, n1_other, label1)]:
            for u, v, k, data in list(g.edges(keys=True, data=True)):
                if {u, v} == {open_n, other_n}:
                    g.remove_edge(u, v, k)
                    break
            if open_n in g.nodes():
                g.remove_node(open_n)

        # New trivalent reconnection node
        all_ids = [int(n) for n in g.nodes() if str(n).lstrip('-').isdigit()]
        new_node = str(max(all_ids, default=0) + 1)
        g.add_node(new_node)
        g.add_edge(str(n0_other), new_node, label=label0)
        g.add_edge(str(n1_other), new_node, label=label1)

        # New open edge
        ext_node = str(max(int(n) for n in g.nodes() if str(n).lstrip('-').isdigit()) + 1)
        g.add_node(ext_node)
        g.add_edge(new_node, ext_node, label=new_label)
        return g

    def _save_and_compute_symbolic(self, filename):
        """Symbolic probability path: build formula expressions, save txt files, show dialog."""
        print("\n" + "="*70)
        print("COMPUTING SYMBOLIC TRANSITION PROBABILITY")
        print("="*70)

        print("\n[1/3] Symbolic reduction of G₁ (original)...")
        canon_G1 = self.symbolic_norm_from_graph(self.original_graph)
        print(f"      → {len(canon_G1)} canonical term(s)")
        norm_G1_str = terms_to_formula_string(canon_G1)

        base = os.path.splitext(filename)[0]
        norm_G1_file = base + "_norm_G1.txt"
        save_formula_txt(canon_G1, norm_G1_file)

        print("[2/3] Symbolic reduction of G₂ (modified)...")
        formula_blocks = []  # one entry per reconnection channel

        for r in self.reconnections:
            old1, old2 = r['old_edges'][0], r['old_edges'][1]
            a_raw, b_raw = old1['label'], old2['label']
            a_str = _sanitize_py(str(a_raw))
            b_str = _sanitize_py(str(b_raw))

            if r.get('compute_all', False):
                # Enumerate all possible c values; build G2 for each
                for c in r.get('possible_values', []):
                    g2 = self._apply_reconnection(self.original_graph, r['old_edges'], c)
                    canon_G2 = self.symbolic_norm_from_graph(g2)
                    norm_G2_str = terms_to_formula_string(canon_G2)
                    c_str = _sanitize_py(str(c))
                    block = (
                        f"# c = {c}\n"
                        f"abs(\n"
                        f"    delta({c_str})\n"
                        f"    / theta({a_str}, {b_str}, {c_str})\n"
                        f"    * ({norm_G2_str})\n"
                        f"    / ({norm_G1_str})\n"
                        f")"
                    )
                    formula_blocks.append(block)
                    print(f"      → c={c}: {len(canon_G2)} canonical term(s)")
            else:
                # Single-value reconnection: G2 is already self.graph
                canon_G2 = self.symbolic_norm_from_graph(self.graph)
                norm_G2_str = terms_to_formula_string(canon_G2)
                c_raw = r['new_edge']['label']
                c_str = _sanitize_py(str(c_raw))
                block = (
                    f"abs(\n"
                    f"    delta({c_str})\n"
                    f"    / theta({a_str}, {b_str}, {c_str})\n"
                    f"    * ({norm_G2_str})\n"
                    f"    / ({norm_G1_str})\n"
                    f")"
                )
                formula_blocks.append(block)
                norm_G2_file = base + "_norm_G2.txt"
                save_formula_txt(canon_G2, norm_G2_file)
                print(f"      → {len(canon_G2)} canonical term(s)")

        print("[3/3] Saving outputs...")
        txt_file = base + "_symbolic_probability.txt"
        with open(txt_file, "w") as f:
            f.write("# Symbolic reconnection probability\n")
            f.write(
                f"# Evaluate: python scripts/evaluate_formula.py "
                f"-f {os.path.basename(txt_file)}\n"
            )
            if len(formula_blocks) == 1:
                f.write(formula_blocks[0] + "\n")
            else:
                for i, block in enumerate(formula_blocks):
                    f.write(f"# --- Channel {i+1} ---\n")
                    f.write(block + "\n\n")

        print(f"\n✓ Saved symbolic probability to {txt_file}")
        print(f"✓ Saved G₁ norm formula to {norm_G1_file}")

        summary = (
            f"Symbolic computation complete!\n\n"
            f"Graph has symbolic labels → numerical evaluation skipped.\n\n"
            f"Output files:\n"
            f"  {os.path.basename(txt_file)}\n"
            f"  {os.path.basename(norm_G1_file)}\n\n"
            f"To evaluate numerically (after binding variables):\n"
            f"  python scripts/evaluate_formula.py \\\n"
            f"    -f {os.path.basename(txt_file)} \\\n"
            f"    --var n=1 --var x=2  (example)"
        )
        tk.messagebox.showinfo("Symbolic Computation Complete", summary)
        self.master.destroy()

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

        # Detect symbolic labels (sympy expressions cannot be enumerated)
        labels_are_symbolic = not (isinstance(label1, (int, float)) and
                                   isinstance(label2, (int, float)))

        if labels_are_symbolic:
            # Symbolic labels: skip compute-all, go straight to single-value prompt
            choice = False
            possible_values = ["(symbolic — any admissible value)"]
        else:
            # Calculate all possible values based on triangle inequality
            possible_values = self.calculate_possible_values(label1, label2)

            # Ask user if they want all values or just one
            choice = tk.messagebox.askyesnocancel(
                "Reconnection Mode",
                f"Connecting edges with labels {label1} and {label2}.\n\n" +
                f"Possible new edge values: {possible_values}\n\n" +
                "Compute probabilities for:\n" +
                "  YES - All possible values\n" +
                "  NO - Single specific value\n" +
                "  CANCEL - Abort reconnection",
                parent=self.master
            )

            if choice is None:
                return

        if choice:  # Yes - compute all (numeric labels only)
            reconnection_data = {
                'old_edges': [
                    {'nodes': (open_node1, other_node1), 'label': label1},
                    {'nodes': (open_node2, other_node2), 'label': label2}
                ],
                'possible_values': possible_values,
                'compute_all': True
            }
            self.reconnections.append(reconnection_data)
            self.selected_nodes = []
            print(f"\n✓ Marked reconnection: {label1} + {label2} → ALL values")

        else:  # No - single value
            prompt_suffix = (
                "Enter label for new open edge (numeric or symbolic):"
                if labels_are_symbolic else
                f"Possible values: {possible_values}\nEnter label for new open edge:"
            )
            new_label = tk.simpledialog.askstring(
                "New Edge Label",
                f"Connecting {label1} + {label2}.\n{prompt_suffix}",
                parent=self.master
            )

            if new_label is None:
                return

            try:
                new_label = float(new_label)
            except ValueError:
                pass

            # Validate numeric labels against possible values (skip for symbolic graphs)
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

            # Save state
            self.save_state("Reconnection")

            # Perform the reconnection
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
            self.save_and_compute()
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

    # Full norm pipeline: glue → reduce → Kronecker → expand 6j → canonicalise
    # → evaluate. Returns a float. Called for G1, G2, and the subgraph.
    def compute_norm_from_graph(self, graph, quiet=True):
        glued_graph = glue_open_edges(graph)
        terms = reduce_all_cycles(glued_graph, animator=None)

        clean_terms = []
        for T in terms:
            t = apply_kroneckers(T)
            if t is not None:
                clean_terms.append(t)

        for term in clean_terms:
            new = []
            for c in term["coeffs"]:
                if isinstance(c, dict) and c.get("type") == "6j":
                    expanded = expand_6j_symbolic(c)
                    new.extend(expanded)
                else:
                    new.append(c)
            term["coeffs"] = new

        canon_terms = canonicalise_terms(clean_terms)
        result = evaluate_spin_network(canon_terms)

        if not quiet:
            print(f"  → Norm = {result}")

        return result

    # Loads the file and delegates to compute_norm_from_graph.
    # Used to compute ||G1|| from the original input file path.
    def compute_norm_from_file(self, graph_file, quiet=True):
        if not quiet:
            print(f"\nComputing norm for {graph_file}...")

        graph = nx.read_graphml(graph_file, force_multigraph=True)

        # Convert edge labels to float or keep as string
        for u, v, data in graph.edges(data=True):
            try:
                data["label"] = float(data["label"])
            except ValueError:
                pass

        # Ensure all nodes have valid positions
        pos = nx.kamada_kawai_layout(graph)
        for node in graph.nodes:
            if "x" in graph.nodes[node] and "y" in graph.nodes[node]:
                graph.nodes[node]["pos"] = (float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
            else:
                graph.nodes[node]["pos"] = pos[node]

        return self.compute_norm_from_graph(graph, quiet=quiet)

    # Constructs the subgraph whose norm appears in the denominator of P.
    # Includes all "added" edges plus the original open edges at their attachment
    # nodes, so the subgraph is a valid closed spin network when glued.
    # Returns None if no edges were added.
    def build_subgraph_from_added_edges(self):
        if not self.added_edges:
            return None

        subgraph = nx.MultiGraph()

        # 1) Add all explicitly added edges
        for edge_info in self.added_edges:
            n1, n2 = edge_info['nodes']
            label = edge_info['label']

            if n1 in self.nodes:
                subgraph.add_node(n1, pos=self.nodes[n1])
            if n2 in self.nodes:
                subgraph.add_node(n2, pos=self.nodes[n2])

            subgraph.add_edge(n1, n2, label=label)

        # 2) Include original open edges at attachment nodes
        if self.original_graph is not None:
            for node in list(subgraph.nodes()):
                if (node in self.original_graph.nodes()
                        and self.original_graph.degree(node) < 3):
                    for u, v, key, data in self.original_graph.edges(node, keys=True, data=True):
                        other = v if u == node else u
                        label = data.get('label', 1.0)
                        if other in self.nodes:
                            subgraph.add_node(other, pos=self.nodes[other])
                        if not subgraph.has_edge(node, other):
                            subgraph.add_edge(node, other, label=label)

        return subgraph

    # Computes ∏ Δ(j) for each j in labels. Δ(j) = (-1)^(2j) × (2j+1).
    def compute_delta_product(self, labels):
        """Compute product of Δ symbols."""
        evaluator = SpinNetworkEvaluator()
        total_sign_exponent = 0.0
        total_magnitude = 1.0
        for j in labels:
            sign_exp, mag = evaluator.delta_symbol(j, power=1)
            total_sign_exponent += sign_exp
            total_magnitude *= mag
        evaluator.cleanup()
        # Combine sign and magnitude
        sign = (-1.0) ** int(round(total_sign_exponent))
        return sign * total_magnitude

    # Computes ∏ Θ(j1,j2,j3) for each triplet in the reconnection list.
    def compute_theta_product(self, triplets):
        evaluator = SpinNetworkEvaluator()
        total_sign_exponent = 0.0
        total_magnitude = 1.0
        for j1, j2, j3 in triplets:
            sign_exp, mag = evaluator.theta_symbol(j1, j2, j3, power=1)
            total_sign_exponent += sign_exp
            total_magnitude *= mag
        evaluator.cleanup()
        # Combine sign and magnitude
        sign = (-1.0) ** int(round(total_sign_exponent))
        return sign * total_magnitude

    # Builds G₂(c) for a compute_all reconnection record: copies self.graph,
    # removes each old open end and its edge, then inserts a new trivalent
    # reconnection node + a new open end carrying label c.
    def build_reconnected_graph(self, reconnection_record, c):
        G = self.graph.copy()
        other_nodes = []
        other_labels = []

        for old_edge in reconnection_record['old_edges']:
            na, nb = old_edge['nodes']
            label = old_edge['label']

            # Open node = lower-degree endpoint (degree 1 = external stub).
            if G.degree(na) <= G.degree(nb):
                open_n, other_n = na, nb
            else:
                open_n, other_n = nb, na

            # Remove the stub edge then the open node.
            if G.has_edge(open_n, other_n):
                k = list(G[open_n][other_n].keys())[0]
                G.remove_edge(open_n, other_n, k)
            if G.has_node(open_n):
                G.remove_node(open_n)

            other_nodes.append(other_n)
            other_labels.append(label)

        # New reconnection node + new open end.
        int_nodes = [n for n in G.nodes() if isinstance(n, int)]
        new_node = (max(int_nodes) if int_nodes else 0) + 1
        G.add_node(new_node)
        G.add_edge(other_nodes[0], new_node, label=other_labels[0])
        G.add_edge(other_nodes[1], new_node, label=other_labels[1])
        external = new_node + 1
        G.add_node(external)
        G.add_edge(new_node, external, label=c)
        return G

    # Main computation button (S key): saves the modified graph, then runs all
    # five computation steps (||G1||, ||G2||, ||subgraph||, Δ product, Θ product)
    # and prints/saves P = |Δ/Θ × 1/||sub|| × ||G2||/||G1|||.
    def save_and_compute(self):
        """Save modified graph and compute probability."""
        if not self.added_edges and not self.reconnections:
            tk.messagebox.showwarning(
                "Nothing to Compute",
                "You haven't made any modifications yet!\n\n" +
                "Use E to add edges, O to add open edges,\n" +
                "or select open nodes and press C to reconnect."
            )
            return

        # Determine output filename: same directory as input, named transition_to_graph.graphml
        if self.input_file:
            input_dir = os.path.dirname(self.input_file)
            filename = os.path.join(input_dir, "transition_to_graph.graphml")
        else:
            # Fallback to file dialog if no input file
            filename = tk.filedialog.asksaveasfilename(
                title="Save Transition Data",
                defaultextension=".graphml",
                filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
                initialfile="transition_to_graph.graphml"
            )
            if not filename:
                return

        # Save the current graph
        graph_copy = self.graph.copy()
        for node, attrs in graph_copy.nodes(data=True):
            for key, value in attrs.items():
                if isinstance(value, tuple) or hasattr(value, 'free_symbols'):
                    graph_copy.nodes[node][key] = str(value)

        for u, v, key, attrs in graph_copy.edges(keys=True, data=True):
            for k, value in attrs.items():
                if isinstance(value, tuple) or hasattr(value, 'free_symbols'):
                    graph_copy.edges[u, v, key][k] = str(value)

        try:
            nx.write_graphml(graph_copy, filename)

            # Route to symbolic path if the graph has non-numeric edge labels
            if self._has_symbolic_labels():
                self._save_and_compute_symbolic(filename)
                return

            # Compute norms using the same methods as evaluate_norm.py
            print("\n" + "="*70)
            print("COMPUTING TRANSITION PROBABILITY")
            print("="*70)

            # Compute ||G₁|| (original graph norm)
            print("\n[STEP 1] Computing original graph norm ||G₁||...")
            norm_original = self.compute_norm_from_file(self.input_file, quiet=False)
            print(f"  ||G₁|| = {norm_original}")

            # Determine whether any reconnection uses compute_all mode.
            compute_all_recs = [r for r in self.reconnections if r.get('compute_all', False)]

            # Compute ||G₂|| only for single-value reconnections (graph already
            # physically modified).  For compute_all, ||G₂(c)|| is computed
            # separately for each c inside the per-channel loop below.
            print("\n[STEP 2] Computing new graph norm ||G₂||...")
            if not compute_all_recs:
                norm_new = self.compute_norm_from_graph(self.graph, quiet=False)
                print(f"  ||G₂|| = {norm_new}")
            else:
                norm_new = None
                print("  Deferred (compute_all mode: ||G₂(c)|| computed per channel)")

            # Compute ||subgraph|| (norm of added edges subgraph)
            print("\n[STEP 3] Computing subgraph norm ||subgraph||...")
            if self.added_edges:
                subgraph = self.build_subgraph_from_added_edges()
                if subgraph and subgraph.number_of_edges() > 0:
                    norm_subgraph = self.compute_norm_from_graph(subgraph, quiet=False)
                else:
                    norm_subgraph = 1.0
                    print("  No valid subgraph, using ||subgraph|| = 1")
            else:
                norm_subgraph = 1.0
                print("  No added edges, using ||subgraph|| = 1")
            print(f"  ||subgraph|| = {norm_subgraph}")

            # Collect Δ terms for original open ends closed by the subgraph.
            # These are the edges in the original graph at each attachment node
            # (a node that was degree < 3 in the original and now has added edges).
            old_open_end_labels = []
            seen_attachment_nodes = set()
            if self.original_graph is not None and self.added_edges:
                for edge_info in self.added_edges:
                    n1, n2 = edge_info['nodes']
                    for node in [n1, n2]:
                        if (node not in seen_attachment_nodes
                                and node in self.original_graph.nodes()
                                and self.original_graph.degree(node) < 3):
                            seen_attachment_nodes.add(node)
                            for u, v, key, data in self.original_graph.edges(
                                    node, keys=True, data=True):
                                old_open_end_labels.append(data.get('label', 1.0))

            delta_old = 1.0
            if old_open_end_labels:
                delta_old = self.compute_delta_product(old_open_end_labels)

            # ── compute_all path: iterate over every admissible c value ──────
            if compute_all_recs:
                # Deduplicate records by unordered label pair.
                seen_pairs = set()
                unique_recs = []
                for r in compute_all_recs:
                    key = frozenset([r['old_edges'][0]['label'], r['old_edges'][1]['label']])
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        unique_recs.append(r)

                per_channel = []   # [{c, norm, delta, theta, probability}, ...]

                print("\n[STEP 4+5] Computing probability for each possible channel...")
                if old_open_end_labels:
                    print(f"  Δ_old({', '.join(map(str, old_open_end_labels))}) = {delta_old}")

                for rec in unique_recs:
                    j1 = rec['old_edges'][0]['label']
                    j2 = rec['old_edges'][1]['label']
                    possible_values = rec['possible_values']

                    print(f"\n  Coupling {j1} ⊗ {j2}  →  possible values: {possible_values}")
                    print(f"  {'c':<8} {'||G₂(c)||':<15} {'Δ(c)':<10}"
                          f" {'Θ(j₁,j₂,c)':<14} {'P(c)':<15}")
                    print("  " + "─"*62)

                    total_prob = 0.0
                    for c in possible_values:
                        G2c = self.build_reconnected_graph(rec, c)
                        norm_c = self.compute_norm_from_graph(G2c)
                        delta_c = self.compute_delta_product([c])
                        theta_c = self.compute_theta_product([(j1, j2, c)])

                        if theta_c == 0 or norm_original == 0 or norm_subgraph == 0:
                            p_c = 0.0
                        else:
                            p_c = abs(
                                (delta_c * delta_old / theta_c)
                                * (1.0 / norm_subgraph) * (norm_c / norm_original)
                            )

                        total_prob += p_c
                        per_channel.append({
                            'c': float(c), 'norm_G2': float(norm_c),
                            'delta': float(delta_c), 'theta': float(theta_c),
                            'probability': float(p_c)
                        })
                        print(f"  {c:<8} {norm_c:<15.6e} {delta_c:<10.3f}"
                              f" {theta_c:<14.3f} {p_c:<15.6e}")

                    print("  " + "─"*62)
                    norm_ok = abs(total_prob - 1.0) < 1e-8
                    print(f"  Sum = {total_prob:.15e}  "
                          f"{'✓ sums to 1' if norm_ok else '✗ does NOT sum to 1'}")

                print("\n" + "★"*70)
                for entry in per_channel:
                    print(f"  P(c={entry['c']}) = {entry['probability']:.15e}")
                print("★"*70)

                # Placeholders for JSON / dialog (use first channel as representative).
                probability = per_channel[0]['probability'] if per_channel else 0.0
                norm_new = per_channel[0]['norm_G2'] if per_channel else 0.0
                delta_product = per_channel[0]['delta'] if per_channel else 1.0
                theta_product = per_channel[0]['theta'] if per_channel else 1.0
                new_labels = [entry['c'] for entry in per_channel]
                theta_triplets = []

            # ── single-value path: one probability per reconnection ───────────
            else:
                print("\n[STEP 4] Computing Δ and Θ products...")
                new_labels = []
                theta_triplets = []

                for r in self.reconnections:
                    new_edge = r.get('new_edge', {})
                    old1 = r['old_edges'][0]
                    old2 = r['old_edges'][1]
                    if 'label' in new_edge:
                        new_labels.append(new_edge['label'])
                        theta_triplets.append((old1['label'], old2['label'], new_edge['label']))

                delta_product = 1.0
                theta_product = 1.0

                if new_labels:
                    delta_product = self.compute_delta_product(new_labels)
                    print(f"  Δ({', '.join(map(str, new_labels))}) = {delta_product}")

                if old_open_end_labels:
                    print(f"  Δ_old({', '.join(map(str, old_open_end_labels))}) = {delta_old}")

                if theta_triplets:
                    theta_product = self.compute_theta_product(theta_triplets)
                    theta_str = ' × '.join([f"Θ({a},{b},{c})" for a, b, c in theta_triplets])
                    print(f"  {theta_str} = {theta_product}")

                print("\n[STEP 5] Computing probability...")
                if theta_product == 0 or norm_original == 0 or norm_subgraph == 0:
                    probability = 0.0
                    print("  Warning: Division by zero encountered!")
                else:
                    probability = abs(
                        (delta_product * delta_old / theta_product)
                        * (1.0 / norm_subgraph) * (norm_new / norm_original)
                    )

                print(f"\n  P = |Δ × Δ_old / Θ × 1/||sub|| × ||G₂||/||G₁|||")
                print(f"    = |{delta_product} × {delta_old} / {theta_product}"
                      f" × 1/{norm_subgraph} × {norm_new}/{norm_original}|")
                print(f"    = {probability:.15e}")
                per_channel = []

                print("\n" + "★"*70)
                print(f"PROBABILITY: P = {probability:.15e}")
                print("★"*70)

            # Save transition data
            import json
            data_file = filename.replace('.graphml', '_transition.json')

            # Serialize added_edges and reconnections
            def serialize_tuple(obj):
                if isinstance(obj, tuple):
                    return list(obj)
                return obj

            transition_data = {
                'original_file': self.input_file,
                'new_file': filename,
                'added_edges': [
                    {k: serialize_tuple(v) for k, v in e.items()}
                    for e in self.added_edges
                ],
                'reconnections': [
                    {k: serialize_tuple(v) if not isinstance(v, dict) else
                        {kk: serialize_tuple(vv) for kk, vv in v.items()}
                     for k, v in r.items()}
                    for r in self.reconnections
                ],
                'norms': {
                    'original_graph': float(norm_original),
                    'new_graph': float(norm_new) if norm_new is not None else None,
                    'subgraph': float(norm_subgraph)
                },
                'computation': {
                    'delta_product': float(delta_product),
                    'delta_old_open_ends': float(delta_old),
                    'old_open_end_labels': [float(l) if isinstance(l, (int, float)) else l
                                            for l in old_open_end_labels],
                    'theta_product': float(theta_product),
                    'new_edge_labels': [float(l) if isinstance(l, (int, float)) else l for l in new_labels],
                    'theta_triplets': theta_triplets,
                    'per_channel': per_channel
                },
                'probability': float(probability),
                'probability_formula': (
                    "P = |[Delta(c1)...Delta(cn) * Delta(j1)...Delta(jm)"
                    " / Theta(c1,s1,t1)...Theta(cn,sn,tn)]"
                    " * [1/||subgraph||] * [||new_graph|| / ||old_graph||]|"
                ),
                'formula_notes': {
                    'ci': 'new edge label produced by reconnection i',
                    'ji': 'original open end label at each subgraph attachment point',
                    'si_ti': 'the two edges coupled at reconnection vertex i',
                    'subgraph_norm': '1 if only reconnecting existing open ends (no added edges)'
                }
            }

            with open(data_file, 'w') as f:
                json.dump(transition_data, f, indent=2, default=str)

            print(f"\n✓ Saved modified graph to {filename}")
            print(f"✓ Saved transition data to {data_file}")

            # Show summary
            if per_channel:
                total_prob = sum(e['probability'] for e in per_channel)
                norm_ok = abs(total_prob - 1.0) < 1e-8
                prob_lines = "\n".join(
                    f"  P(c={e['c']}) = {e['probability']:.6e}"
                    for e in per_channel
                )
                summary = (
                    f"Transition computed!\n\n"
                    f"Graph: {filename}\n"
                    f"Data: {data_file}\n\n"
                    f"Norms:\n"
                    f"  ||G₁|| = {norm_original}\n"
                    f"  ||subgraph|| = {norm_subgraph}\n\n"
                    f"Probabilities per channel:\n{prob_lines}\n\n"
                    f"Sum = {total_prob:.6e}  "
                    f"{'✓ normalised' if norm_ok else '✗ not normalised'}"
                )
            else:
                summary = (
                    f"Transition computed!\n\n"
                    f"Graph: {filename}\n"
                    f"Data: {data_file}\n\n"
                    f"Norms:\n"
                    f"  ||G₁|| (original) = {norm_original}\n"
                    f"  ||G₂|| (new) = {norm_new}\n"
                    f"  ||subgraph|| = {norm_subgraph}\n\n"
                    f"Added edges: {len(self.added_edges)}\n"
                    f"Reconnections: {len(self.reconnections)}\n\n"
                    f"PROBABILITY: P = {probability:.6e}"
                )

            tk.messagebox.showinfo("Computation Complete", summary)
            self.master.destroy()

        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.messagebox.showerror("Error", f"Failed to compute:\n{e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spin network transition tool")
    parser.add_argument("input_file", nargs="?", help="Input .graphml file")
    args = parser.parse_args()

    root = tk.Tk()
    tool = TransitionTool(root, args.input_file)
    root.mainloop()
