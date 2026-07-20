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


    # Activation Functions 
    
    def tanh(self):
        """Hyperbolic tangent activation with caching."""
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            if self._requires_grad:
                self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_tanh'
        return out
    
    def relu(self):
        """Rectified Linear Unit activation."""
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'relu')
        
        def _backward():
            if self._requires_grad:
                self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_relu'
        return out
    
    def leaky_relu(self, alpha: float = 0.01):
        """Leaky ReLU activation with configurable slope."""
        out = Value(self.data if self.data > 0 else alpha * self.data, (self,), 'leaky_relu')
        
        def _backward():
            if self._requires_grad:
                grad_input = 1.0 if self.data > 0 else alpha
                self.grad += grad_input * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_leaky_relu'
        return out
    
    def sigmoid(self):
        """Sigmoid activation function."""
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')
        
        def _backward():
            if self._requires_grad:
                self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_sigmoid'
        return out
    
    def elu(self, alpha: float = 1.0):
        """Exponential Linear Unit activation."""
        x = self.data
        out_val = x if x >= 0 else alpha * (math.exp(x) - 1)
        out = Value(out_val, (self,), 'elu')
        
        def _backward():
            if self._requires_grad:
                grad_input = 1.0 if x >= 0 else out_val + alpha
                self.grad += grad_input * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_elu'
        return out
    
    def gelu(self):
        """Gaussian Error Linear Unit (approximation)."""
        x = self.data
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        c = math.sqrt(2.0 / math.pi)
        tanh_arg = c * (x + 0.044715 * x**3)
        gelu_val = 0.5 * x * (1 + math.tanh(tanh_arg))
        out = Value(gelu_val, (self,), 'gelu')
        
        def _backward():
            if self._requires_grad:
                # Derivative approximation
                tanh_val = math.tanh(tanh_arg)
                sech2 = 1 - tanh_val**2
                d_tanh = c * (1 + 3 * 0.044715 * x**2)
                d_gelu = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * d_tanh
                self.grad += d_gelu * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_gelu'
        return out


    # Loss Functions 
    
    def mse_loss(self, target):
        """Mean squared error loss."""
        diff = self - target
        return diff ** 2
    
    def cross_entropy(self, target, epsilon: float = 1e-7):
        """Cross entropy loss with numerical stability."""
        p = self.sigmoid()
        p_clipped = p.clamp(epsilon, 1 - epsilon)
        loss = -(target * p_clipped.log() + (1 - target) * (1 - p_clipped).log())
        return loss
    
    def hinge_loss(self, target, margin: float = 1.0):
        """Hinge loss for SVM."""
        return (margin - target * self).relu()
    
    def huber_loss(self, target, delta: float = 1.0):
        """Huber loss robust to outliers."""
        diff = self - target
        abs_diff = diff.abs()
        if abs_diff.data <= delta:
            return 0.5 * diff ** 2
        else:
            return delta * (abs_diff - 0.5 * delta)


    # Utility Methods 
    
    def abs(self):
        """Absolute value."""
        out = Value(abs(self.data), (self,), 'abs')
        
        def _backward():
            if self._requires_grad:
                self.grad += (1.0 if self.data > 0 else -1.0) * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_abs'
        return out
    
    def log(self):
        """Natural logarithm."""
        assert self.data > 0, "Log defined only for positive values"
        out = Value(math.log(self.data), (self,), 'log')
        
        def _backward():
            if self._requires_grad:
                self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_log'
        return out
    
    def exp(self):
        """Exponential function."""
        x = math.exp(self.data)
        out = Value(x, (self,), 'exp')
        
        def _backward():
            if self._requires_grad:
                self.grad += x * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_exp'
        return out
    
    def sqrt(self):
        """Square root."""
        assert self.data >= 0, "Square root defined for non-negative values"
        out = Value(math.sqrt(self.data), (self,), 'sqrt')
        
        def _backward():
            if self._requires_grad:
                self.grad += (0.5 / out.data) * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_sqrt'
        return out
    
    def clamp(self, min_val: float = None, max_val: float = None):
        """Clamp value between min and max."""
        clamped = self.data
        if min_val is not None and clamped < min_val:
            clamped = min_val
        if max_val is not None and clamped > max_val:
            clamped = max_val
        out = Value(clamped, (self,), 'clamp')
        
        def _backward():
            if self._requires_grad:
                grad_input = 0.0
                if (min_val is None or self.data > min_val) and \
                    (max_val is None or self.data < max_val):
                    grad_input = 1.0
                self.grad += grad_input * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_clamp'
        return out
    
    def dropout(self, p: float = 0.5, training: bool = True):
        """Dropout regularization."""
        if not training:
            return self
        
        mask = 1.0 if random.random() > p else 0.0
        scale = 1.0 / (1.0 - p) if mask > 0 else 0.0
        out = Value(self.data * mask * scale, (self,), 'dropout')
        
        def _backward():
            if self._requires_grad:
                self.grad += mask * scale * out.grad
        out._backward = _backward
        out._grad_fn = '_backward_dropout'
        return out
    
    def batch_norm(self, mean, var, epsilon: float = 1e-5):
        """Batch normalization operation."""
        x_hat = (self - mean) / (var + epsilon).sqrt()
        out = Value(x_hat.data, (self,), 'batch_norm')
        
        def _backward():
            if self._requires_grad:
                # Simplified backward (full batch norm gradient is complex)
                self.grad += out.grad / (var.data + epsilon).sqrt()
        out._backward = _backward
        out._grad_fn = '_backward_batch_norm'
        return out


    def detach(self):
        """Detach value from computation graph."""
        out = Value(self.data, (), 'detach')
        out._requires_grad = False
        out._grad_fn = 'detached'
        return out
    
    def retain_grad(self):
        """Force retention of gradient for this node."""
        self._retain_grad = True
        return self
    
    def zero_grad(self):
        """Reset gradient of this node and all descendants."""
        visited = set()
        
        def _zero_grad(v):
            if v not in visited:
                visited.add(v)
                v.grad = 0.0
                for child in v._prev:
                    _zero_grad(child)
        
        _zero_grad(self)
    
    def backward(self):
        """Perform reverse-mode automatic differentiation."""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for v in reversed(topo):
            if v._backward is not None:
                v._backward()
            # Clear gradients if not retained
            if not v._retain_grad and v is not self:
                # clear to save memory
                pass
