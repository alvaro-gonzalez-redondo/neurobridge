from __future__ import annotations

from typing import Optional, List

import contextlib
import contextvars

import torch


class _ParentStack(contextlib.AbstractContextManager):
    """Context manager that tracks the current parent node in a hierarchical structure.

    This class manages a stack of parent nodes during the construction of a node
    hierarchy, allowing child nodes to automatically register with their parents
    when created within a context block.

    Examples
    --------
    >>> with _ParentStack(parent_node):
    ...     # Any Node created in this context will be added as a child to parent_node
    ...     child_node = Node()  # Automatically added as a child to parent_node
    """

    _stack_var = contextvars.ContextVar("_stack", default=[])

    def __init__(self, parent: Node):
        """Initialize the ParentStack with a parent node.

        Parameters
        ----------
        parent : Node
            The node to be set as the current parent within this context.
        """
        self._parent = parent
        self._token = None

    def __enter__(self):
        """Enter the context and set the provided node as the current parent.

        Returns
        -------
        Node
            The parent node, allowing for method chaining.
        """
        stack = self._stack_var.get()
        self._token = self._stack_var.set(stack + [self._parent])
        return self._parent

    def __exit__(self, exc_type, exc, tb):
        """Exit the context and restore the previous parent in the stack.

        Parameters
        ----------
        exc_type : type
            The type of exception raised, if any.
        exc : Exception
            The exception raised, if any.
        tb : traceback
            The traceback, if an exception was raised.

        Returns
        -------
        bool
            Always returns False, meaning any exceptions will be propagated.
        """
        self._stack_var.reset(self._token)
        return False  # no suprime excepciones

    @staticmethod
    def current_parent() -> Optional[Node]:
        """Get the current parent node from the stack.

        Returns
        -------
        _Node or None
            The current parent node, or None if the stack is empty.
        """
        stack = _ParentStack._stack_var.get()
        return stack[-1] if stack else None


class Node:
    """Base class for all nodes in the neural network hierarchy.

    The _Node class implements a tree structure where each node can have a parent and
    multiple children. _Nodes automatically register with their parent when created
    within a _ParentStack context.

    This class provides the foundation for the structural organization of the neural
    network components and implements a visitor pattern through the _process() and
    _ready() methods, which are called recursively on the tree.

    Attributes
    ----------
    children : List[_Node]
        Direct child nodes of this node.
    parent : Optional[_Node]
        Parent node, or None if this is a root node.
    """

    children: List[Node]
    parent: Optional[Node]

    def __init__(self):
        """Initialize a new _Node.

        If created within a _ParentStack context, the node will automatically
        register itself with the current parent.
        """
        self.children = []
        self.parent = None

        parent = _ParentStack.current_parent()
        if parent is not None:
            parent.add_child(self)

    def add_child(self, node: Node) -> None:
        """Add a child node to this node.

        If the node already has a parent, it will be removed from its
        previous parent's children list before being added to this node.

        Parameters
        ----------
        node : _Node
            The node to add as a child.
        """
        # Nodes should be unique in the scene tree
        if node.parent is not None:
            node.parent.remove_child(node)

        self.children.append(node)
        node.parent = self

    def remove_child(self, node: Node):
        """Remove a child node from this node.

        Parameters
        ----------
        node : _Node
            The node to remove.
        """
        self.children.remove(node)

    def _call_ready(self) -> None:
        """Recursively call _ready() on all children, then on self.

        This method should not be overridden. Override _ready() instead.
        """
        for child in self.children:
            child._call_ready()
        self._ready()

    def _ready(self) -> None:
        """Initialize the node after all children are ready.

        This method is called after all children's _ready() methods have been called.
        Override this method to perform initialization that depends on children being
        fully set up.
        """
        pass

    def _call_process(self) -> None:
        """Recursively call _process() on all children, then on self.

        This method should not be overridden. Override _process() instead.
        """
        for child in self.children:
            child._call_process()
        self._process()

    def _process(self) -> None:
        """Define the node's behavior during each simulation step.

        Override this method to implement the node's processing logic that
        should be executed during each step of the simulation.
        """
        pass


# core.py
class GPUNode(Node):
    """A node that is associated with a specific GPU device.

    This class extends the base Node class to add GPU device information,
    making it the base class for all components that perform computations on GPU.

    Attributes
    ----------
    device : torch.device
        The GPU device this node is associated with.
    """

    device: torch.device

    def __init__(self, device: str):
        """Initialize a new _GPUNode.

        Parameters
        ----------
        device : str
            String representation of the GPU device (e.g., 'cuda:0').
            This will be converted to a torch.device.
        """
        super().__init__()
        self.device = torch.device(device)
