from __future__ import annotations

from typing import Optional, List

import contextlib
import contextvars

import torch


class _ParentStack(contextlib.AbstractContextManager):
    _stack_var = contextvars.ContextVar("_stack", default=[])

    def __init__(self, parent: _Node):
        self._parent = parent
        self._token  = None

    def __enter__(self):
        stack = self._stack_var.get()
        self._token = self._stack_var.set(stack + [self._parent])
        return self._parent

    def __exit__(self, exc_type, exc, tb):
        self._stack_var.reset(self._token)
        return False          # no suprime excepciones

    @staticmethod
    def current_parent() -> Optional[_Node]:
        stack = _ParentStack._stack_var.get()
        return stack[-1] if stack else None


# core.py
class _Node:
    children: List[_Node]
    parent: Optional[_Node]

    def __init__(self):
        self.children = []
        self.parent = None

        parent = _ParentStack.current_parent()
        if parent is not None:
            parent.add_child(self)


    def add_child(self, node:_Node) -> None:
        # Nodes should be unique in the scene tree
        if node.parent is not None:
            node.parent.remove_child(node)
        
        self.children.append(node)
        node.parent = self
    
    def remove_child(self, node:_Node):
        self.children.remove(node)

    def _call_ready(self) -> None:
        for child in self.children:
            child._call_ready()
        self._ready()

    def _ready(self) -> None:
        """Override this to set up the node after all children are ready."""
        pass

    def _call_process(self) -> None:
        for child in self.children:
            child._call_process()
        self._process()

    def _process(self) -> None:
        """Override this to define what the node does each step."""
        pass


# core.py
class _GPUNode(_Node):
    """A node attached to a GPU."""
    device: torch.device

    def __init__(self, device:str):
        super().__init__()
        self.device = torch.device(device)