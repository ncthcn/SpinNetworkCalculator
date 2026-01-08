"""
Graph Reduction Animation Module

Creates beautiful visualizations of the spin network reduction process.
Generates both:
- Individual PNG slides for each step
- Animated GIF showing the complete reduction
- Optional PDF slideshow

The animation shows:
1. Original graph
2. Glued graph (after theta insertion)
3. Each F-move application
4. Each triangle reduction
5. Final result
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional


class ReductionAnimator:
    """
    Captures and animates the graph reduction process step by step.

    Usage:
        animator = ReductionAnimator(output_dir="reduction_steps")
        animator.add_step(graph, title="Original Graph", description="...")
        animator.add_step(graph2, title="After F-move", description="...")
        animator.save_gif("reduction.gif", duration=2.0)
        animator.save_slides_pdf("reduction_slides.pdf")
    """

    def __init__(self, output_dir: str = "reduction_steps", dpi: int = 150):
        """
        Initialize the animator.

        Parameters:
        -----------
        output_dir : str
            Directory to save individual step images
        dpi : int
            Resolution for saved images (higher = better quality)
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.steps = []  # List of (graph, title, description, metadata)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Style configuration - Beautiful modern design
        self.fig_size = (14, 10)
        self.node_size = 1200
        self.node_color = "#5DADE2"      # Softer blue
        self.edge_color = "#34495E"       # Dark blue-gray
        self.edge_width = 3.0
        self.font_size = 13
        self.title_font_size = 18

        # Beautiful color scheme (inspired by modern UI design)
        self.colors = {
            "original": "#5DADE2",       # Sky blue - original edges
            "new": "#EC7063",            # Soft red - newly added
            "modified": "#F4D03F",       # Golden yellow - modified
            "removed": "#BDC3C7",        # Light gray - removed
            "highlighted": "#58D68D"     # Mint green - highlighted
        }

        # Background gradient colors
        self.bg_gradient_start = "#F8F9FA"
        self.bg_gradient_end = "#FFFFFF"

    def add_step(self,
                 graph: nx.MultiGraph,
                 title: str,
                 description: str = "",
                 highlight_nodes: List = None,
                 highlight_edges: List = None,
                 operation: str = None):
        """
        Add a reduction step to the animation.

        Parameters:
        -----------
        graph : nx.MultiGraph
            The graph at this step (will be copied)
        title : str
            Title for this step (e.g., "F-move on triangle ABC")
        description : str
            Detailed description of what happened
        highlight_nodes : List
            Nodes to highlight in this step
        highlight_edges : List of (u, v, key)
            Edges to highlight in this step
        operation : str
            Type of operation: "f-move", "triangle", "glue", etc.
        """
        # Make a copy to avoid mutations
        graph_copy = graph.copy()

        step_data = {
            "graph": graph_copy,
            "title": title,
            "description": description,
            "highlight_nodes": highlight_nodes or [],
            "highlight_edges": highlight_edges or [],
            "operation": operation,
            "step_number": len(self.steps) + 1
        }

        self.steps.append(step_data)

        # Save individual PNG
        self._save_step_image(step_data)

    def _save_step_image(self, step_data: Dict):
        """Save a single step as a PNG image."""
        fig, ax = plt.subplots(figsize=self.fig_size, facecolor='white')

        self._draw_step(ax, step_data)

        # Save
        filename = f"step_{step_data['step_number']:03d}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _draw_step(self, ax, step_data: Dict):
        """Draw a single step on the given axes with beautiful styling."""
        graph = step_data["graph"]
        title = step_data["title"]
        description = step_data["description"]
        highlight_nodes = step_data["highlight_nodes"]
        highlight_edges = step_data["highlight_edges"]
        operation = step_data["operation"]
        step_num = step_data["step_number"]

        # Get positions
        pos = self._get_positions(graph)

        # Clear axes
        ax.clear()
        ax.set_aspect('equal')
        ax.axis('off')

        # Add subtle gradient background
        ax.set_facecolor('#FAFBFC')

        # Draw edges with shadows for depth
        self._draw_edges_with_shadows(ax, graph, pos, highlight_edges)

        # Draw nodes with gradient effect
        self._draw_nodes_with_gradients(ax, graph, pos, highlight_nodes)

        # Draw edge labels with beautiful styling
        edge_labels = {}
        for u, v, key, data in graph.edges(keys=True, data=True):
            label = data.get("label", "")
            edge_labels[(u, v, key)] = str(label)

        self._draw_edge_labels(ax, graph, pos, edge_labels)

        # Draw node labels
        nx.draw_networkx_labels(
            graph, pos, ax=ax,
            font_size=self.font_size + 1,
            font_weight='bold',
            font_color='white',
            font_family='sans-serif'
        )

        # Add title and description
        title_text = f"Step {step_num}: {title}"
        ax.set_title(title_text, fontsize=self.title_font_size, fontweight='bold', pad=20)

        # Add description box
        if description:
            desc_box = mpatches.FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.12,
                boxstyle="round,pad=0.01",
                transform=ax.transAxes,
                facecolor='#ecf0f1',
                edgecolor='#34495e',
                linewidth=2,
                alpha=0.9
            )
            ax.add_patch(desc_box)

            ax.text(
                0.5, 0.08, description,
                transform=ax.transAxes,
                fontsize=11,
                ha='center', va='center',
                wrap=True,
                bbox=dict(boxstyle='round', facecolor='none', edgecolor='none')
            )

        # Add statistics in corner
        stats_text = f"Nodes: {graph.number_of_nodes()}  |  Edges: {graph.number_of_edges()}"
        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
        )

        # Add operation indicator
        if operation:
            op_colors = {
                "f-move": "#e74c3c",
                "triangle": "#2ecc71",
                "glue": "#3498db",
                "initial": "#95a5a6"
            }
            op_color = op_colors.get(operation, "#34495e")

            ax.text(
                0.02, 0.98, operation.upper(),
                transform=ax.transAxes,
                fontsize=10,
                ha='left', va='top',
                fontweight='bold',
                color=op_color,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=op_color, linewidth=2, alpha=0.9)
            )

    def _draw_edges_with_shadows(self, ax, graph, pos, highlight_edges):
        """Draw edges with shadow effect for depth."""
        # First pass: draw shadows (offset slightly)
        edge_shadow_colors = []
        edge_shadow_widths = []

        for u, v, key, data in graph.edges(keys=True, data=True):
            is_highlighted = (u, v, key) in highlight_edges or (v, u, key) in highlight_edges
            edge_shadow_colors.append('#00000020')  # Semi-transparent black
            edge_shadow_widths.append(self.edge_width * 1.2 if is_highlighted else self.edge_width)

        # Draw shadows with offset
        offset_pos = {node: (x + 0.01, y - 0.01) for node, (x, y) in pos.items()}
        nx.draw_networkx_edges(
            graph, offset_pos, ax=ax,
            edge_color=edge_shadow_colors,
            width=edge_shadow_widths,
            arrows=False,
            alpha=0.3
        )

        # Second pass: draw actual edges
        edge_colors = []
        edge_widths = []

        for u, v, key, data in graph.edges(keys=True, data=True):
            is_highlighted = (u, v, key) in highlight_edges or (v, u, key) in highlight_edges
            if is_highlighted:
                edge_colors.append(self.colors["highlighted"])
                edge_widths.append(self.edge_width * 1.4)
            else:
                edge_colors.append(self.edge_color)
                edge_widths.append(self.edge_width)

        nx.draw_networkx_edges(
            graph, pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=False,
            alpha=0.85,
            style='solid'
        )

    def _draw_nodes_with_gradients(self, ax, graph, pos, highlight_nodes):
        """Draw nodes with gradient effect and shadows."""
        # Draw node shadows first
        nx.draw_networkx_nodes(
            graph, {node: (x + 0.015, y - 0.015) for node, (x, y) in pos.items()},
            ax=ax,
            node_color='#00000030',
            node_size=self.node_size,
            alpha=0.4,
            edgecolors='none'
        )

        # Draw nodes with beautiful styling
        node_colors = []
        edge_colors = []
        for node in graph.nodes():
            if node in highlight_nodes:
                node_colors.append(self.colors["new"])
                edge_colors.append('#C0392B')  # Darker red border
            else:
                node_colors.append(self.node_color)
                edge_colors.append('#2980B9')  # Darker blue border

        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_color=node_colors,
            node_size=self.node_size,
            alpha=0.95,
            edgecolors=edge_colors,
            linewidths=3.5
        )

    def _get_positions(self, graph: nx.MultiGraph) -> Dict:
        """
        Get beautiful node positions using multiple layout algorithms.

        Uses a hybrid approach:
        1. Try stored positions first (from graph editor)
        2. For planar graphs, use planar layout
        3. Otherwise use Kamada-Kawai (force-directed with good spacing)
        4. Apply smoothing and centering
        """
        pos = {}

        # Try to use stored positions first
        has_stored = all("pos" in graph.nodes[node] for node in graph.nodes())

        if has_stored:
            for node in graph.nodes():
                pos[node] = graph.nodes[node]["pos"]
        else:
            # Check if graph is planar
            is_planar, embedding = nx.check_planarity(graph)

            if is_planar and graph.number_of_nodes() >= 3:
                # Use planar layout for planar graphs (most aesthetic)
                try:
                    pos = nx.planar_layout(graph, scale=2.0)
                except:
                    # Fallback if planar layout fails
                    pos = nx.kamada_kawai_layout(graph, scale=2.0)
            else:
                # Use Kamada-Kawai for non-planar (better than spring for small graphs)
                try:
                    pos = nx.kamada_kawai_layout(graph, scale=2.0)
                except:
                    # Final fallback to spring layout
                    pos = nx.spring_layout(graph, k=2.0, iterations=100, seed=42)

            # Apply smoothing and centering
            pos = self._smooth_positions(pos)

            # Store positions back in graph for consistency
            for node, position in pos.items():
                graph.nodes[node]["pos"] = position

        # Ensure positions are well-scaled and centered
        return self._normalize_positions(pos)

    def _smooth_positions(self, pos: Dict) -> Dict:
        """Apply smoothing to positions to avoid overlaps."""
        if len(pos) <= 1:
            return pos

        # Convert to numpy for easier manipulation
        positions = np.array(list(pos.values()))

        # Center positions
        center = positions.mean(axis=0)
        positions -= center

        # Scale to nice range
        max_dist = np.max(np.abs(positions))
        if max_dist > 0:
            positions = positions / max_dist * 1.5

        # Rebuild dict
        return {node: positions[i] for i, node in enumerate(pos.keys())}

    def _normalize_positions(self, pos: Dict) -> Dict:
        """Normalize positions to fit nicely in the plot."""
        if not pos:
            return pos

        positions = np.array(list(pos.values()))

        # Center at origin
        center = positions.mean(axis=0)
        positions -= center

        # Scale to [-1, 1] range with some padding
        max_extent = np.max(np.abs(positions)) if len(positions) > 0 else 1
        if max_extent > 0:
            positions = positions / max_extent * 0.85

        return {node: positions[i] for i, node in enumerate(pos.keys())}

    def _draw_edge_labels(self, ax, graph, pos, edge_labels):
        """Draw edge labels with beautiful styling for multigraphs."""
        for (u, v, key), label in edge_labels.items():
            # Get edge positions
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            # Calculate midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2

            # For parallel edges, offset the label
            parallel_edges = [k for (u2, v2, k) in graph.edges(keys=True)
                            if (u2 == u and v2 == v) or (u2 == v and v2 == u)]

            if len(parallel_edges) > 1:
                # Calculate perpendicular offset
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    offset_x = -dy / length * 0.12 * (parallel_edges.index(key) - len(parallel_edges)/2)
                    offset_y = dx / length * 0.12 * (parallel_edges.index(key) - len(parallel_edges)/2)
                    mx += offset_x
                    my += offset_y

            # Draw shadow for label
            ax.text(
                mx + 0.01, my - 0.01, label,
                fontsize=self.font_size,
                fontweight='bold',
                ha='center', va='center',
                color='#00000030',
                zorder=10
            )

            # Draw label with beautiful background
            ax.text(
                mx, my, label,
                fontsize=self.font_size,
                fontweight='bold',
                ha='center', va='center',
                color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='#BDC3C7', linewidth=2, alpha=0.95),
                zorder=11
            )

    def save_gif(self, filename: str = "reduction.gif", duration: float = 2.0, loop: int = 0):
        """
        Save the reduction sequence as an animated GIF.

        Parameters:
        -----------
        filename : str
            Output filename
        duration : float
            Duration per frame in seconds
        loop : int
            Number of loops (0 = infinite)
        """
        if not self.steps:
            print("⚠️  No steps to animate!")
            return

        print(f"Creating animated GIF with {len(self.steps)} frames...")

        fig, ax = plt.subplots(figsize=self.fig_size, facecolor='white')

        def update(frame_num):
            self._draw_step(ax, self.steps[frame_num])
            return ax,

        anim = FuncAnimation(
            fig, update,
            frames=len(self.steps),
            interval=duration * 1000,
            repeat=True,
            blit=False
        )

        # Save as GIF
        writer = PillowWriter(fps=1/duration)
        anim.save(filename, writer=writer)
        plt.close(fig)

        print(f"  ✓ Saved: {filename}")

    def save_slides_pdf(self, filename: str = "reduction_slides.pdf"):
        """
        Save all steps as a multi-page PDF slideshow.

        Parameters:
        -----------
        filename : str
            Output PDF filename
        """
        if not self.steps:
            print("⚠️  No steps to save!")
            return

        print(f"Creating PDF slideshow with {len(self.steps)} pages...")

        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(filename) as pdf:
            for step_data in self.steps:
                fig, ax = plt.subplots(figsize=self.fig_size, facecolor='white')
                self._draw_step(ax, step_data)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"  ✓ Saved: {filename}")

    def summary(self):
        """Print a summary of captured steps."""
        print("\n" + "="*70)
        print("REDUCTION ANIMATION SUMMARY")
        print("="*70)
        print(f"Total steps captured: {len(self.steps)}")
        print(f"Output directory: {self.output_dir}")
        print("\nSteps:")
        for i, step in enumerate(self.steps, 1):
            print(f"  {i}. {step['title']}")
        print("="*70)
