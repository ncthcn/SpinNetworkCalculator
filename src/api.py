"""
SpinNetworkCalculator – Public Library API
==========================================
This module is the main entry point for using SpinNetworkCalculator as a
Python library (e.g. in a Jupyter Notebook).  In the old workflow, every
step was a separate script that wrote files to disk.  This module wraps the
same mathematical pipeline in a clean in-memory API.

Quick-start
-----------
    from src.api import new_network, load_network

    snet = new_network()                   # opens the drawing GUI
    args = snet.get_args()                 # list of free (symbolic) spin labels
    args[0].value = 1.5                    # assign a concrete spin value
    snet.set_args(args)                    # apply the assignment to the graph

    formula = snet.evaluate_symbolic()     # full graph reduction (expensive)
    formula.save("result.pdf", "pdf")      # save LaTeX expression
    result  = formula.evaluate_numeric()   # numerical value

    # Or load an existing graph from disk:
    snet = load_network("drawn_graph.graphml")

Classes
-------
SpinArg     – one free spin variable (label + assigned value)
Graph       – trivalent spin network graph with evaluation methods
SpinNetwork – user-facing wrapper around Graph (extend with future methods here)
Formula     – symbolic norm expression; supports numeric evaluation and saving

Jupyter / GUI note
------------------
The display() and modify() methods open Tkinter windows.  In Jupyter you
must first run  %gui tk  in a cell to hand the event loop to Tkinter.
In a plain Python script no special setup is needed.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Literal, Optional

import networkx as nx

# ---------------------------------------------------------------------------
# Path setup – makes "from src.X import ..." work from any working directory
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ===========================================================================
# SpinArg  –  one free spin variable
# ===========================================================================

@dataclass
class SpinArg:
    """
    One free spin variable in a spin network.

    When a graph edge carries a *symbolic* label (e.g. "j_3") instead of a
    concrete number (e.g. 1.5), that label becomes a free variable represented
    by a SpinArg.  The user must assign a numeric value to every SpinArg
    before the norm can be evaluated numerically.

    Attributes
    ----------
    label : str
        The edge-label name as it appears in the graph (e.g. "j_3", "a").
    value : float or sympy expression
        Starts as a sympy.Symbol (symbolic); becomes a float once assigned.

    --- Python note for C++ readers ---
    @dataclass is roughly equivalent to a C++ struct with an auto-generated
    constructor.  It creates __init__, __repr__, and __eq__ automatically.
    Attributes are accessed and modified directly (arg.value = 1.5) without
    getter/setter methods.

    Example
    -------
        args = snet.get_args()
        print(args[0].label)     # "j_3"
        args[0].value = 1.5      # direct assignment – no setter method needed
        snet.set_args(args)
    """

    label: str
    value: object  # float when assigned, sympy.Basic (a sympy Symbol) when free

    @property
    def is_numeric(self) -> bool:
        """
        True when this variable holds a concrete number (int or float).

        --- Python note ---
        @property turns a method into a field-like readable attribute.
        You write  arg.is_numeric  (no parentheses), not  arg.is_numeric().
        This replaces the C++ pattern  bool isNumeric() const { ... }.
        """
        return isinstance(self.value, (int, float))

    def __repr__(self) -> str:
        status = "numeric" if self.is_numeric else "symbolic"
        return f"SpinArg(label={self.label!r}, value={self.value!r}, [{status}])"


# ===========================================================================
# Internal helpers  –  GraphML I/O and formula-string parsing
# ===========================================================================

def _load_graphml(path: str) -> nx.MultiGraph:
    """
    Load a .graphml file and return a NetworkX MultiGraph.

    Edge labels are parsed from their stored string form to the correct
    Python type (float, sympy.Symbol, or sympy expression) using the same
    logic as compute_norm.py.  Node positions stored as x/y attributes are
    converted to the 'pos' tuple attribute expected by the rest of the code.

    Parameters
    ----------
    path : str
        Path to a .graphml file saved by the drawing GUI or Graph.save().
    """
    from src.utils import parse_spin_label  # lazy import avoids circular deps

    graph = nx.read_graphml(path, force_multigraph=True)

    # Convert each edge label from string to its proper Python type
    for u, v, data in graph.edges(data=True):
        if "label" in data:
            data["label"] = parse_spin_label(str(data["label"]))

    # Restore node positions; fall back to layout algorithm if not stored
    fallback_pos = nx.kamada_kawai_layout(graph)
    for node in graph.nodes:
        attrs = graph.nodes[node]
        if "x" in attrs and "y" in attrs:
            attrs["pos"] = (float(attrs["x"]), float(attrs["y"]))
        else:
            attrs["pos"] = fallback_pos[node]

    return graph


def _save_graphml(graph: nx.MultiGraph, path: str) -> None:
    """
    Save a NetworkX MultiGraph to a .graphml file.

    GraphML does not support Python tuples or sympy expressions as attribute
    values, so they are converted to strings before writing.

    Parameters
    ----------
    graph : nx.MultiGraph
        The graph to save.
    path : str
        Destination file path (should end in .graphml).
    """
    graph_copy = graph.copy()

    # Graph-level attributes: complex phase → string
    if "phase" in graph_copy.graph:
        graph_copy.graph["phase"] = str(graph_copy.graph["phase"])

    # Node attributes: tuples and lists → string
    for node, attrs in graph_copy.nodes(data=True):
        for key, value in list(attrs.items()):
            if isinstance(value, (tuple, list)):
                graph_copy.nodes[node][key] = str(value)

    # Edge attributes: tuples and sympy expressions → string
    for u, v, k, attrs in graph_copy.edges(keys=True, data=True):
        for key, value in list(attrs.items()):
            if isinstance(value, tuple):
                graph_copy.edges[u, v, k][key] = str(value)
            elif hasattr(value, "free_symbols"):  # sympy expression
                graph_copy.edges[u, v, k][key] = str(value)

    nx.write_graphml(graph_copy, path)


def _extract_free_variables(formula_string: str) -> List[str]:
    """
    Return all *free* variable names found in a formula string.

    A variable is 'free' (i.e. a user-supplied spin value) when it:
      - appears as an identifier in the formula string, AND
      - is NOT a built-in function name (theta, delta, W6j, …), AND
      - is NOT a lambda-bound sum variable (e.g. 'lambda F_1:' makes F_1 bound).

    Parameters
    ----------
    formula_string : str
        A formula string produced by terms_to_formula_string() or loaded
        from a .txt file.

    Returns
    -------
    list[str]
        Sorted list of free variable names (e.g. ["a", "j_1", "j_2"]).
    """
    # These identifiers are internal function/keyword names, not spin variables
    BUILTINS = {
        "theta", "delta", "W6j", "Sum", "lambda",
        "round", "abs", "max", "min", "int", "float", "str", "bool",
    }

    # Lambda-bound variables are NOT free: "lambda F_1:" means F_1 is bound
    # re.findall returns all matches of the capturing group
    bound_vars: set = set(re.findall(r"\blambda\s+(\w+)\s*:", formula_string))

    # All identifiers that appear anywhere in the formula
    all_identifiers: set = set(
        re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", formula_string)
    )

    free_vars = all_identifiers - BUILTINS - bound_vars
    return sorted(free_vars)


# ===========================================================================
# Formula  –  symbolic norm expression
# ===========================================================================

class Formula:
    """
    The symbolic norm expression for a spin network.

    A Formula is always created by Graph.evaluate_symbolic() and should
    never be instantiated directly by the user.  It holds the canonical
    list of reduction terms (Wigner 6j symbols, theta symbols, delta
    symbols, sign factors, …) and can evaluate them numerically once all
    free spin variables have been assigned numeric values.

    Workflow
    --------
        formula = graph.evaluate_symbolic()    # expensive; graph caches it
        args = formula.get_args()              # free spin variables
        args[0].value = 1.5
        formula.set_args(args)
        result = formula.evaluate_numeric()    # fast once args are assigned

    Persistence
    -----------
        formula.save("result.txt", "txt")      # plain-text Python expression
        formula.save("result.pdf", "pdf")      # LaTeX-rendered PDF
        f2 = Formula.load("result.txt")        # reload from text (eval only)

    Note: a Formula loaded from .txt supports evaluate_numeric() but NOT
    save(..., 'pdf'), because the original coefficient terms are not stored
    in the text file.
    """

    def __init__(
        self,
        terms: list,
        free_arg_labels: List[str],
    ) -> None:
        """
        Internal constructor.  Use Graph.evaluate_symbolic() instead.

        Parameters
        ----------
        terms : list
            Canonical coefficient terms from norm_reducer.canonicalise_terms().
        free_arg_labels : list[str]
            Names of symbolic edge labels that need numeric assignment.
        """
        import sympy

        # The canonical coefficient terms (list of dicts).
        # None when this Formula was loaded from a .txt file.
        self._terms: Optional[list] = terms

        # Pre-compute the formula string (Python expression used for evaluation
        # and for saving to .txt).
        if terms is not None:
            from src.LaTeX_rendering import terms_to_formula_string
            self._formula_string: str = terms_to_formula_string(terms)
        else:
            self._formula_string = ""  # will be set by load()

        # One SpinArg per free variable, initially holding a sympy.Symbol
        # sorted() ensures a deterministic order regardless of insertion order
        self._args: List[SpinArg] = [
            SpinArg(label=name, value=sympy.Symbol(name))
            for name in sorted(free_arg_labels)
        ]

    # ------------------------------------------------------------------
    # Argument management
    # ------------------------------------------------------------------

    def get_args(self) -> List[SpinArg]:
        """
        Return the list of free spin variables in this formula.

        Returns a *shallow copy* of the internal list, so modifying the
        returned list does not affect the Formula.  Use set_args() to push
        changes back.

        Returns
        -------
        list[SpinArg]
        """
        return list(self._args)

    def set_args(self, args: List[SpinArg]) -> None:
        """
        Assign numeric spin values to free variables in the formula.

        Only SpinArgs with is_numeric == True are applied.  Symbolic ones
        are silently skipped (the variable stays free).

        Parameters
        ----------
        args : list[SpinArg]
            The list returned by get_args(), with some values updated.

        Raises
        ------
        ValueError
            If len(args) > len(self.get_args()).
        ValueError
            If args contains a label not present in this formula.
        """
        if len(args) > len(self._args):
            raise ValueError(
                f"Too many arguments: received {len(args)}, "
                f"formula has {len(self._args)} free variables."
            )

        for incoming in args:
            if not incoming.is_numeric:
                continue  # skip symbolic values

            # Find the matching arg and update its value.
            # The for/else construct in Python: the else block runs only
            # when the loop ends WITHOUT hitting a break statement — i.e.
            # when no match was found.  This replaces a C++ bool-flag pattern.
            for existing in self._args:
                if existing.label == incoming.label:
                    existing.value = incoming.value
                    break
            else:
                raise ValueError(
                    f"Unknown variable '{incoming.label}'. "
                    f"Known variables: {[a.label for a in self._args]}"
                )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_numeric(
        self,
        args: Optional[List[SpinArg]] = None,
        backend: str = "auto",
        max_two_j: int = 200,
    ) -> float:
        """
        Evaluate the formula numerically.

        Parameters
        ----------
        args : list[SpinArg], optional
            Additional or overriding spin assignments.  If omitted, uses
            the values stored by previous set_args() calls.
        backend : {'auto', 'jax', 'multiprocessing', 'serial'}, optional
            Computation backend for the summation over F-variables.

            - 'auto'           picks the fastest available option (default)
            - 'jax'            uses JAX/GPU; requires  pip install jax  (and
                               pip install jax-metal  on Apple Silicon)
            - 'multiprocessing' parallelises over CPU cores using Python's
                               multiprocessing module
            - 'serial'         single-threaded; easiest to debug

        max_two_j : int, optional
            Pre-allocates wigxjpf tables up to spin j = max_two_j / 2.
            Increase this if your network contains spins larger than 100.
            Default: 200  (supports j up to 100).

        Returns
        -------
        float
            The numerical value of the spin network norm.

        Raises
        ------
        ValueError
            If any free variable is still unassigned (symbolic).
        """
        from src.spin_evaluator import FormulaEvaluator

        # Build the variable substitution dictionary.
        # dict[str, float]  maps variable name → numeric spin value.
        variables: dict = {}

        # First apply values stored from previous set_args() calls
        for arg in self._args:
            if arg.is_numeric:
                variables[arg.label] = float(arg.value)

        # Then apply the overrides passed directly to this call
        if args:
            for arg in args:
                if arg.is_numeric:
                    variables[arg.label] = float(arg.value)

        # Check that every free variable now has a numeric assignment
        unassigned = [a.label for a in self._args if a.label not in variables]
        if unassigned:
            raise ValueError(
                f"The following spin variables are still unassigned: {unassigned}.\n"
                f"Assign them with formula.set_args([SpinArg('name', value), ...])\n"
                f"or pass them directly: formula.evaluate_numeric([SpinArg(...)])."
            )

        evaluator = FormulaEvaluator(max_two_j=max_two_j, backend=backend)
        try:
            result = evaluator.evaluate(
                self._formula_string,
                # Pass None (not an empty dict) when there are no substitutions;
                # FormulaEvaluator treats None and {} differently
                variables=variables if variables else None,
            )
        finally:
            evaluator.cleanup()  # always release the underlying wigxjpf C++ resources

        return result

    def evaluate_batch(
        self,
        args_list: List[List[SpinArg]],
        backend: str = "auto",
        max_two_j: int = 200,
    ) -> List[float]:
        """
        Evaluate the formula for multiple sets of spin values in one call.

        The evaluator is initialised once and reused for every entry in
        args_list, which is more efficient than calling evaluate_numeric()
        in a Python loop for large batches.

        Parameters
        ----------
        args_list : list[list[SpinArg]]
            Each inner list is one complete set of spin assignments, in
            the same format as the args parameter of evaluate_numeric().
        backend : {'auto', 'jax', 'multiprocessing', 'serial'}, optional
            See evaluate_numeric() for details.
        max_two_j : int, optional
            See evaluate_numeric() for details.

        Returns
        -------
        list[float]
            One float per element of args_list, in the same order.

        Example
        -------
            from src.api import SpinArg

            formula = snet.evaluate_symbolic()
            # Evaluate for j_1 in {0.5, 1.0, 1.5, 2.0}
            args_list = [
                [SpinArg("j_1", 0.5)],
                [SpinArg("j_1", 1.0)],
                [SpinArg("j_1", 1.5)],
                [SpinArg("j_1", 2.0)],
            ]
            results = formula.evaluate_batch(args_list, backend="multiprocessing")
        """
        from src.spin_evaluator import FormulaEvaluator

        # Initialise once and reuse across all entries — avoids repeated
        # wigxjpf table allocation, which is the expensive part
        evaluator = FormulaEvaluator(max_two_j=max_two_j, backend=backend)
        try:
            results = []
            for single_args in args_list:
                # Build variable dict for this entry (same logic as evaluate_numeric)
                variables: dict = {}
                for arg in self._args:
                    if arg.is_numeric:
                        variables[arg.label] = float(arg.value)
                if single_args:
                    for arg in single_args:
                        if arg.is_numeric:
                            variables[arg.label] = float(arg.value)

                unassigned = [a.label for a in self._args if a.label not in variables]
                if unassigned:
                    raise ValueError(
                        f"Unassigned variables in batch entry: {unassigned}"
                    )

                results.append(
                    evaluator.evaluate(
                        self._formula_string,
                        variables=variables if variables else None,
                    )
                )
        finally:
            evaluator.cleanup()

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, format: str = "txt") -> None:
        """
        Save the formula to a file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. "result.txt" or "result.pdf").
        format : {'txt', 'pdf'}
            'txt' saves a plain-text Python expression compatible with
            scripts/evaluate_formula.py.
            'pdf' saves a LaTeX-rendered PDF using matplotlib (requires
            a working LaTeX installation for best results).

        Raises
        ------
        ValueError
            If format is 'pdf' and this Formula was loaded from a .txt file
            (the original coefficient terms are not available).
        ValueError
            If an unrecognised format string is given.
        """
        if format not in ("txt", "pdf"):
            raise ValueError(
                f"Unknown format {format!r}. Use 'txt' or 'pdf'."
            )
        if format == "pdf" and self._terms is None:
            raise ValueError(
                "Cannot save as PDF: this Formula was loaded from a .txt file "
                "and the original coefficient terms are not available.\n"
                "Regenerate it via graph.evaluate_symbolic() if you need a PDF."
            )

        if format == "txt":
            from src.LaTeX_rendering import save_formula_txt
            save_formula_txt(self._terms, path)
        else:
            from src.LaTeX_rendering import save_latex_pdf
            save_latex_pdf(self._terms, path)

    @classmethod
    def load(cls, path: str) -> "Formula":
        """
        Load a Formula from a plain-text .txt file.

        The loaded Formula supports evaluate_numeric() and evaluate_batch()
        but NOT save(..., 'pdf'), because the original coefficient terms
        are not stored in the text file format.

        Parameters
        ----------
        path : str
            Path to a .txt file produced by Formula.save(..., 'txt').

        Returns
        -------
        Formula

        Raises
        ------
        ValueError
            If the file contains no formula expression.

        --- Python note ---
        @classmethod is like a static factory method in C++.  The first
        parameter is cls (the class itself) instead of self (an instance).
        You call it as  Formula.load("file.txt"),  not on an instance.
        """
        import sympy

        with open(path, "r") as f:
            # Skip comment lines starting with '#'
            lines = [line for line in f if not line.startswith("#")]
        formula_string = "".join(lines).strip()

        if not formula_string:
            raise ValueError(
                f"File {path!r} contains no formula (only comments or empty lines)."
            )

        # Build the object without calling __init__ by using object.__new__.
        # object.__new__(cls) allocates memory for the object but skips __init__.
        # This is a standard Python pattern when you need a factory method that
        # initialises the object differently from the normal constructor.
        formula = object.__new__(cls)
        formula._terms = None               # no coefficient terms when loaded
        formula._formula_string = formula_string

        # Detect free variables from the formula string
        free_var_names = _extract_free_variables(formula_string)
        formula._args = [
            SpinArg(label=name, value=sympy.Symbol(name))
            for name in free_var_names
        ]

        return formula

    def __repr__(self) -> str:
        source = "loaded from file" if self._terms is None else "from graph"
        free = [a.label for a in self._args if not a.is_numeric]
        assigned = [a.label for a in self._args if a.is_numeric]
        n_terms = "?" if self._terms is None else len(self._terms)
        return (
            f"Formula({source}, terms={n_terms}, "
            f"free={free}, assigned={assigned})"
        )


# ===========================================================================
# Graph  –  trivalent spin network graph
# ===========================================================================

class Graph:
    """
    A trivalent spin network graph with symbolic and numeric evaluation.

    Wraps a NetworkX MultiGraph and provides:
      - display() / modify()      – GUI windows for inspection and editing
      - get_args() / set_args()   – manage free (symbolic) spin variables
      - save()                    – persist to .graphml
      - evaluate_symbolic()       – full reduction pipeline, result is cached
      - evaluate_numeric()        – shortcut for evaluate_symbolic().evaluate_numeric()

    State management and caching
    ----------------------------
    evaluate_symbolic() is expensive (F-moves, triangle reductions, …).
    Its result is cached in self._formula.  A boolean flag self._dirty
    controls whether the cache is valid:

      - _dirty starts as True (nothing computed yet)
      - evaluate_symbolic() sets _dirty = False after computing
      - modify() and set_args() set _dirty = True, invalidating the cache

    This ensures that in a Jupyter notebook, re-running a cell that calls
    evaluate_symbolic() does not redo the expensive computation unless the
    graph actually changed.

    --- Python note for C++ readers ---
    The underscore prefix _ marks members as 'internal'.  Python does not
    enforce this at the language level, but it is the universal convention
    that says "do not use this from outside the class."
    """

    def __init__(self, nx_graph: nx.MultiGraph) -> None:
        """
        Wrap an existing NetworkX MultiGraph.

        Parameters
        ----------
        nx_graph : nx.MultiGraph
            The underlying graph.  Must be trivalent for evaluate_symbolic()
            to succeed (every internal node has exactly 3 edges).
        """
        self._nx_graph: nx.MultiGraph = nx_graph

        # Cached symbolic result – None until evaluate_symbolic() is first called
        self._formula: Optional[Formula] = None

        # True means the graph changed since the last evaluate_symbolic() call
        self._dirty: bool = True

        # Populated by _update_args(); one entry per symbolic edge label
        self._args: List[SpinArg] = []
        self._update_args()

    # ------------------------------------------------------------------
    # Internal: argument bookkeeping
    # ------------------------------------------------------------------

    def _update_args(self) -> None:
        """
        Rescan all edge labels and rebuild self._args.

        Called automatically by __init__ and after any mutation
        (modify(), set_args()).  Symbolic labels produce SpinArg entries;
        numeric labels are silently skipped.  Duplicate labels across
        multiple edges produce only one SpinArg entry.
        """
        import sympy
        from src.utils import is_numeric_label

        seen: set = set()
        args: List[SpinArg] = []

        for u, v, key, data in self._nx_graph.edges(keys=True, data=True):
            label = data.get("label")
            if label is None:
                continue
            if is_numeric_label(label):
                continue  # numeric label → not a free variable

            label_str = str(label)
            if label_str in seen:
                continue  # same symbol appears on multiple edges → one entry

            seen.add(label_str)

            # Preserve the label as a sympy expression if it already is one;
            # otherwise create a new sympy.Symbol from its string name
            sym_value = label if hasattr(label, "free_symbols") else sympy.Symbol(label_str)
            args.append(SpinArg(label=label_str, value=sym_value))

        self._args = args

    def _triangular_ok(self, label_str: str, new_value: float) -> bool:
        """
        Return True if assigning new_value to every edge labelled label_str
        still satisfies the triangular inequality at all affected vertices.

        Vertices where one or more edges still carry a symbolic label are
        skipped (they will be re-checked when the last symbol is assigned).

        Parameters
        ----------
        label_str : str
            The edge label being assigned (e.g. "j_3").
        new_value : float
            The proposed numeric spin value.
        """
        from src.utils import vertex_satisfies_triangular_conditions

        for u, v, k, data in self._nx_graph.edges(keys=True, data=True):
            if str(data.get("label")) != label_str:
                continue

            # Both endpoints are affected by this edge's label
            for node in (u, v):
                labels_at_node = []
                all_numeric = True

                for nbr, edge_dict in self._nx_graph[node].items():
                    for ek, edata in edge_dict.items():
                        lbl = edata.get("label")
                        if str(lbl) == label_str:
                            labels_at_node.append(new_value)
                        elif isinstance(lbl, (int, float)):
                            labels_at_node.append(lbl)
                        else:
                            all_numeric = False

                # Only validate when every edge at this vertex is now numeric
                if all_numeric and len(labels_at_node) == 3:
                    if not vertex_satisfies_triangular_conditions(labels_at_node):
                        return False

        return True

    # ------------------------------------------------------------------
    # Argument management (public)
    # ------------------------------------------------------------------

    def get_args(self) -> List[SpinArg]:
        """
        Return the list of free (symbolic) spin variables in this graph.

        Returns a shallow copy so modifying the returned list does not
        affect the Graph.  Use set_args() to push changes back.

        Returns
        -------
        list[SpinArg]
            One SpinArg per unique symbolic edge label.  Empty if the graph
            has only numeric labels.
        """
        return list(self._args)

    def set_args(self, args: List[SpinArg]) -> None:
        """
        Assign numeric spin values to symbolic edge labels.

        For each SpinArg with is_numeric == True:
          1. Validates that the assignment satisfies the triangular inequality
             at all vertices where all three edges are now numeric.
          2. Updates every edge in the graph that carries that label.

        Symbolic SpinArgs (is_numeric == False) are silently skipped.
        After the call, _update_args() is called and the formula cache is
        invalidated.

        Parameters
        ----------
        args : list[SpinArg]
            The list returned by get_args(), with some values updated.

        Raises
        ------
        ValueError
            If len(args) > len(self.get_args()) ("too many arguments").
        ValueError
            If an assignment would violate the triangular inequality.
        """
        if len(args) > len(self._args):
            raise ValueError(
                f"Too many arguments: received {len(args)}, "
                f"graph has {len(self._args)} free variables."
            )

        for arg in args:
            if not arg.is_numeric:
                continue  # only apply concrete numeric assignments

            if not self._triangular_ok(arg.label, float(arg.value)):
                raise ValueError(
                    f"Assigning {arg.value} to edge label '{arg.label}' would "
                    f"violate the triangular inequality |j1-j2| ≤ j3 ≤ j1+j2 "
                    f"at one or more vertices."
                )

            # Apply to every edge that carries this label
            for u, v, k, data in self._nx_graph.edges(keys=True, data=True):
                if str(data.get("label")) == arg.label:
                    self._nx_graph.edges[u, v, k]["label"] = float(arg.value)

        self._update_args()  # rebuild _args; newly numeric labels disappear from the list
        self._dirty = True
        self._formula = None  # invalidate the cached formula

    # ------------------------------------------------------------------
    # GUI methods
    # ------------------------------------------------------------------

    def display(self) -> None:
        """
        Open a read-only visual inspector for this graph.

        Launches the GraphInspector Tkinter GUI.  The call blocks until the
        user closes the inspector window.  The graph is NOT modified.

        Note: In Jupyter, run  %gui tk  in a cell before calling this.
        """
        import tkinter as tk
        from scripts.inspect_graph import GraphInspector  # type: ignore

        # Save to a temp file so GraphInspector (which expects a path) can load it
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            temp_path = f.name
        try:
            _save_graphml(self._nx_graph, temp_path)
            root = tk.Tk()
            GraphInspector(root, temp_path)
            root.mainloop()
        finally:
            os.unlink(temp_path)  # always clean up, even if an exception occurs

    def modify(self) -> None:
        """
        Open an interactive editor for this graph.

        Launches the GraphModifier Tkinter GUI.  The call blocks until the
        user saves and closes the editor.  The graph is updated in-place
        with any changes the user made, and the formula cache is invalidated.

        Note: In Jupyter, run  %gui tk  in a cell before calling this.
        """
        import tkinter as tk
        from scripts.modify_graph import GraphModifier  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            temp_path = f.name
        try:
            _save_graphml(self._nx_graph, temp_path)
            root = tk.Tk()
            GraphModifier(root, temp_path)
            root.mainloop()
            # Reload the (potentially modified) graph from the temp file
            self._nx_graph = _load_graphml(temp_path)
        finally:
            os.unlink(temp_path)

        # The graph may have changed; reset all derived state
        self._update_args()
        self._dirty = True
        self._formula = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save this graph to a .graphml file.

        Parameters
        ----------
        path : str
            Destination path, e.g. "my_network.graphml".
        """
        _save_graphml(self._nx_graph, path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_symbolic(self) -> Formula:
        """
        Run the full symbolic reduction pipeline and return a Formula.

        This is the most expensive operation in the library.  It performs:
          1. Triangular inequality validation
          2. Open-edge gluing (creating the theta/closed graph)
          3. F-moves and triangle reductions  (graph_reducer.py)
          4. Kronecker delta simplifications   (norm_reducer.py)
          5. 6j → Wigner 6j expansion
          6. Canonicalisation (Regge symmetries, sorting)

        The result is cached.  Calling this again on an unchanged graph
        returns the cached Formula instantly.  The cache is invalidated
        by modify() and set_args().

        Returns
        -------
        Formula
            The symbolic norm expression.

        Raises
        ------
        ValueError
            If the graph violates the triangular inequality.
        """
        if not self._dirty and self._formula is not None:
            return self._formula  # return the cached result

        # Lazy imports keep module load time short
        from src.gluer import glue_open_edges
        from src.graph_reducer import reduce_all_cycles
        from src.norm_reducer import (
            apply_kroneckers,
            canonicalise_terms,
            expand_6j_symbolic,
        )
        from src.utils import check_triangular_condition

        # Step 1: validate the graph
        check_triangular_condition(self._nx_graph)

        # Step 2: glue open edges to form the closed theta graph
        glued = glue_open_edges(self._nx_graph)

        # Step 3: apply F-moves and triangle reductions.
        # Returns a list of "term" dicts, each with keys "graph" and "coeffs".
        terms = reduce_all_cycles(glued)

        # Step 4: apply Kronecker delta substitutions.
        # Terms that reduce to zero return None and are discarded.
        clean_terms = []
        for t in terms:
            result = apply_kroneckers(t)
            if result is not None:
                clean_terms.append(result)

        # Step 5: expand compact 6j symbols to full Wigner 6j form
        for term in clean_terms:
            expanded_coeffs = []
            for coeff in term["coeffs"]:
                if isinstance(coeff, dict) and coeff.get("type") == "6j":
                    expanded_coeffs.extend(expand_6j_symbolic(coeff))
                else:
                    expanded_coeffs.append(coeff)
            term["coeffs"] = expanded_coeffs

        # Step 6: canonicalise (apply Regge symmetries, sort arguments)
        canon_terms = canonicalise_terms(clean_terms)

        # Build the Formula, passing the names of the graph's free variables
        free_labels = [a.label for a in self._args]
        self._formula = Formula(canon_terms, free_labels)
        self._dirty = False

        return self._formula

    def evaluate_numeric(self, args: Optional[List[SpinArg]] = None) -> float:
        """
        Compute the numerical value of the spin network norm.

        Shortcut for  evaluate_symbolic().evaluate_numeric(args).

        Parameters
        ----------
        args : list[SpinArg], optional
            Spin value assignments.  Required unless all edge labels are
            already numeric or have been assigned via set_args().

        Returns
        -------
        float
        """
        return self.evaluate_symbolic().evaluate_numeric(args)

    def __repr__(self) -> str:
        n_nodes = self._nx_graph.number_of_nodes()
        n_edges = self._nx_graph.number_of_edges()
        n_free = len(self._args)
        cache_status = (
            "formula cached" if (self._formula is not None and not self._dirty)
            else "no cached formula"
        )
        return (
            f"Graph(nodes={n_nodes}, edges={n_edges}, "
            f"free_args={n_free}, {cache_status})"
        )


# ===========================================================================
# SpinNetwork  –  user-facing wrapper
# ===========================================================================

class SpinNetwork:
    """
    A spin network: the main object users interact with.

    This class is a thin wrapper around Graph.  It exists as a separate
    layer so that additional physics-level members and methods (reconnection
    probabilities, network comparisons, amplitudes, …) can be added here in
    the future without modifying the lower-level Graph API.

    All current methods delegate directly to the underlying Graph object.
    See the Graph class for full documentation on each method.

    Example
    -------
        from src.api import new_network

        snet = new_network()                   # draw a graph interactively
        snet.display()                         # inspect it visually
        snet.modify()                          # edit it

        args = snet.get_args()                 # [SpinArg("j_1", Symbol("j_1")), ...]
        args[0].value = 1.5
        snet.set_args(args)

        formula = snet.evaluate_symbolic()
        formula.save("norm.pdf", "pdf")

        result = formula.evaluate_numeric()
        print(result)
    """

    def __init__(self, graph: Graph) -> None:
        """
        Wrap an existing Graph object.

        Parameters
        ----------
        graph : Graph
            The underlying trivalent spin network graph.
        """
        self._graph: Graph = graph

    # ------------------------------------------------------------------
    # Delegation methods
    # ------------------------------------------------------------------
    # Python note: these one-liner methods are the "delegation" pattern.
    # In C++ you would do the same by holding a member object and calling
    # its methods.  The explicit delegation here (rather than inheritance)
    # is intentional: SpinNetwork IS NOT a Graph, it HAS a Graph.

    def display(self) -> None:
        """Open the read-only inspector window.  See Graph.display()."""
        self._graph.display()

    def modify(self) -> None:
        """Open the interactive editor window.  See Graph.modify()."""
        self._graph.modify()

    def get_args(self) -> List[SpinArg]:
        """Return free spin variables.  See Graph.get_args()."""
        return self._graph.get_args()

    def set_args(self, args: List[SpinArg]) -> None:
        """Assign numeric spin values.  See Graph.set_args()."""
        self._graph.set_args(args)

    def save(self, path: str) -> None:
        """Save to a .graphml file.  See Graph.save()."""
        self._graph.save(path)

    def evaluate_symbolic(self) -> Formula:
        """Run the symbolic reduction pipeline.  See Graph.evaluate_symbolic()."""
        return self._graph.evaluate_symbolic()

    def evaluate_numeric(self, args: Optional[List[SpinArg]] = None) -> float:
        """Evaluate numerically.  See Graph.evaluate_numeric()."""
        return self._graph.evaluate_numeric(args)

    def __repr__(self) -> str:
        return f"SpinNetwork({self._graph!r})"


# ===========================================================================
# Factory functions  –  create or load a SpinNetwork
# ===========================================================================

def new_network() -> SpinNetwork:
    """
    Draw a new spin network interactively and return it as a SpinNetwork.

    Opens the graph-drawing GUI (Tkinter).  Keyboard shortcuts:
      N – add node    E – add edge    M – move node
      D – delete node X – delete edge Z – undo
    Click "Save & Exit" or press S when finished.

    The GUI blocks until the window is closed.  The graph is captured
    in memory (no temp files required) and wrapped in a SpinNetwork.

    Returns
    -------
    SpinNetwork

    Raises
    ------
    RuntimeError
        If the window was closed without saving (via the OS close button).

    Note: In Jupyter, run  %gui tk  in a cell before calling this.

    Example
    -------
        from src.api import new_network

        snet = new_network()     # blocks until you close the GUI
        print(snet)
    """
    import tkinter as tk
    from scripts.graph import GraphEditor  # type: ignore

    # captured is a one-element list used as a mutable cell that can be
    # written to from inside the nested class below.
    # (In Python, a plain variable in an outer function cannot be reassigned
    # from an inner function/class without the 'nonlocal' keyword.  Using a
    # list avoids that restriction and works in all Python versions.)
    captured: List[Optional[nx.MultiGraph]] = [None]

    class _CapturingEditor(GraphEditor):
        """GraphEditor subclass that saves the graph object before closing."""

        def save_graph(self) -> None:
            # Capture the graph BEFORE calling super() which destroys the window.
            # self.graph is the nx.MultiGraph held by the editor.
            captured[0] = self.graph.copy()
            super().save_graph()  # writes drawn_graph.graphml and closes the window

    root = tk.Tk()
    _CapturingEditor(root)
    root.mainloop()  # blocks until the Tkinter window is destroyed

    if captured[0] is None:
        raise RuntimeError(
            "No graph was saved.  Please click 'Save & Exit' (or press S) "
            "to return the graph.  Closing the window with the X button "
            "does not save."
        )

    return SpinNetwork(Graph(captured[0]))


def load_network(path: str) -> SpinNetwork:
    """
    Load a spin network from a .graphml file.

    Parameters
    ----------
    path : str
        Path to a .graphml file (e.g. "drawn_graph.graphml").

    Returns
    -------
    SpinNetwork

    Example
    -------
        from src.api import load_network

        snet = load_network("drawn_graph.graphml")
        formula = snet.evaluate_symbolic()
        print(formula.evaluate_numeric())
    """
    nx_graph = _load_graphml(path)
    return SpinNetwork(Graph(nx_graph))
