Let us refactor this entire codebase.


**Basic methods and members**

The goal is to make this code usable in a Jupyter Notebook, or more generally to be able to use this code as a library. Here is an example of how one could use it:

```python
snet = new_network()
```
would call the methods in `graph.py` and open the interactive graw-drawing console. Once the graph is drawn, it is returned by `new_network()` ending, and is assigned to `snet`: a `SpinNetwork` class that holds a `Graph` class as one of its members.

The `Graph` class is a graph like we built them so far, but which now has a `print()` method that calls what we wrote in `inspect_graph.py` and a `modify()` method that calls what we wrote in `modify_graph.py`.
Moreover, the `Graph` class has a new member `_args`, as well one private and two public methods:
```python
# private
_update_args()
# public
get_args()
set_args(args)
```
Here, `_update_args()`is called in the constructor, and whenever the graph is modified. It iterates over all edge labels of the graph and checks if each is numerical or symbolic. It then returns some sort of list with components (edge label, symbol), which itself has a `set_value()` method to change the symbol associated to an edge label, a `get_value()` one to get the symbol and a `get_label()` to get the edge label. This list is then precisely `_args`, as is defined already in the code (I think).
`get_args()` just returns `_args`, and `set_args(args)` is defined as
```python
def set_args(args):
    if args.size() <= _args.size():
        for a in args:
            if a.get_value().is_numeric():
                # if (the triangular conditions are satisfied at all vertices touched by a.get_label() when substituting its symbolic value by a.get_value()):
                    # Assign a.get_value() to the edge a.get_label()
        _update_args()
    else:
        # Error "Too many arguments"
```
Finally, the `Graph` class has a `save()` method that saves the specific instantiation of this class as a `.graphml` file in some directory.

We then built a spin network, which can use the methods:
```python
snet.print()
snet.modify()
snet.get_args()
snet.set_args()
snet.save()
```
where `print()`, `modify()`, `get_args()`, `set_args(args)`, and `save()` call the `print()`, `modify()`, `get_args()`, `set_args(args)`,  and `save()` methods of the graph class it holds.

One can then get this list, or assign a new list to the `SpinNetwork` (or `Graph`) using
```python
list = snet.get_args()
list
```
to see the arguments, and for example
```python
list[3].set_value(7.5)
snet.set_args(list)
```
to set the third argument of `_args`to a numerical value, reducing its size by one.


**Evaluations**

The `Graph` (and by extension the `SpinNetwork`) class has three evaluation methods. The first one is a general that calls the two others:
```python
def evaluate(self, type, args):
    if type == "symbolic":
        _evaluate_symbolically(self._graph, self._args)
    elif type == "numerical":
        _evaluate_numerically(self._graph, args)
    else:
        # Error "Please choose either symbolically or numerically"
```
where `_evaluate_symbolically()` calls the methods in `compute_norm.py` and returns a `Formula` class. 
This class holds as a member `_coeffs`, that is an object that is the same type as the output of the builder methods in `graph_reducer.py` and the input of `norm_reducer.py` and `spin_evaluator.py`. It also holds an `_args` member that has is a list with components (label, symbol) where for example a component could be `(j_3, j_3)`. One can then call 
```python
list = form.get_args()
list[3].set_value(7.5)
form.set_args(list)
```
so that now the same component reads (before the call of `_update_args()` inside `set_args(args)`) `(j_3, 7.5)`. We can now define the `Formula` method
```python
def evaluate(args):
    if (args.is_empty() && self._args.is_empty()) || (args.is_numeric() && (args.size() == self._args.size()) && is_valid(self._graph, args)):
        assigned_coeffs = assign(self._coeffs, self._args)
        # Call methods in `evaluate_formula.py` with `assigned_coeff`
    else:
        # Error 'Invalid arguments'
```
where `is_valid(graph, args)` is a global method that checks if assigning all the labels `args.get_value()` to the edges `args.get_label()` satisfies the triangular conditions. Moreovern `assign(coeffs, args)` is a global method that returns a `Coeffs` class with the all the arguments replaced by `args`.

With this we can define `_evaluate_numerically(self, args)` from the `Graph` (and by extension the `SpinNetwork`) class:
```python
def _evaluate_numerically(self, args):
    form = _evaluate_symbolically(self._graph, self._args)
    form.evaluate(args)
```
