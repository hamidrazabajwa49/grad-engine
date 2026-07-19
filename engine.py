"""
engine.py - Scalar autograd engine with enhanced functionality

This module implements a scalar-valued automatic differentiation engine
with support for additional operations, optimizations, and utilities
beyond the original micrograd implementation.
"""

import math
import random
from typing import List, Union, Set, Tuple, Optional, Callable
from functools import lru_cache

class Value:
    """
    A scalar value with automatic differentiation capabilities.
    
    Attributes:
        data (float): The scalar value
        grad (float): Gradient of the value with respect to the loss
        _prev (set): Parent nodes in the computation graph
        _op (str): Operation that created this node
        _backward (callable): Custom backward function for this node
    """
    
    def __init__(self, data: float, _children: tuple = (), _op: str = '', 
                _backward: Optional[Callable] = None):
        """
        Initialize a Value object.
        
        Args:
            data: Scalar value
            _children: Parent Value objects
            _op: Operation label
            _backward: Custom gradient function for this node
        """
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = _backward
        
        # Enhanced features
        self._requires_grad = True
        self._retain_grad = False  # For keeping intermediate gradients
        self._grad_fn = None  # Reference to gradient function for debugging
        
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    # Core Arithmetic Operations 
    
    def __add__(self, other):
        """Addition with automatic broadcasting."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            if self._requires_grad:
                self.grad += out.grad * 1.0
            if other._requires_grad:
                other.grad += out.grad * 1.0
        out._backward = _backward
        out._grad_fn = '_backward_add'
        return out
    
    def __mul__(self, other):
        """Multiplication with automatic broadcasting."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            if self._requires_grad:
                self.grad += other.data * out.grad
            if other._requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_mul'
        return out
    
    def __pow__(self, other):
        """Power operation (constant exponent only)."""
        assert isinstance(other, (int, float)), "Only supports numeric exponents"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            if self._requires_grad:
                self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_pow'
        return out
    
    def __truediv__(self, other):
        """Division implemented as multiplication by reciprocal."""
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)
    
    def __neg__(self):
        """Negation."""
        return self * -1
    
    def __sub__(self, other):
        """Subtraction."""
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    
    def __radd__(self, other):
        """Right addition for scalar + Value."""
        return self + other
    
    def __rmul__(self, other):
        """Right multiplication for scalar * Value."""
        return self * other
    
    def __rsub__(self, other):
        """Right subtraction for scalar - Value."""
        return other + (-self)
    
    def __rtruediv__(self, other):
        """Right division for scalar / Value."""
        return other * (self ** -1)
