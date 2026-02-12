import math

class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self):
        self.grad = self._recursive_op(self.data, None, lambda x, y: 0.0)

    def _recursive_op(self, d1, d2, op):
        """Helper to perform math (like add/zero) on nested lists of any depth."""
        if not isinstance(d1, list):
            return op(d1, d2)
        return [self._recursive_op(item1, item2 if d2 else None, op) 
                for item1, item2 in zip(d1, d2 if d2 else d1)]

    def backward(self):
        # Build topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Start gradient at 1.0 (for scalar loss)
        if self.grad is None:
            self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

    # We need this to handle weight updates in the optimizer later
    def __sub__(self, other):
        # Implementation for: tensor - other
        return self + (other * -1)

    # Simplified addition for the framework
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self._recursive_op(self.data, other.data, lambda x, y: x + y)
        out = Tensor(out_data, (self, other), '+')

        def _backward():
            if self.requires_grad:
                # Add out.grad to self.grad element-wise
                self.grad = self._recursive_op(self.grad, out.grad, lambda x, y: x + y)
            if other.requires_grad:
                other.grad = self._recursive_op(other.grad, out.grad, lambda x, y: x + y)
        out._backward = _backward
        return out