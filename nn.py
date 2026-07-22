"""
nn.py - Neural Network library built on top of the autograd engine

This module implements neural network components including:
- Neuron: A single artificial neuron with weights and bias
- Layer: A collection of neurons
- MLP: Multi-Layer Perceptron with configurable architecture

All parameters are Value objects from the engine module, enabling
automatic differentiation for training.
"""

import random
from typing import List, Union, Optional
from engine import Value


class Module:
    """
    Base class for all neural network modules.
    
    Provides common functionality like parameter management and
    training/evaluation mode switching.
    """
    
    def __init__(self):
        self.training = True
    
    def parameters(self) -> List[Value]:
        """
        Return all trainable parameters of this module.
        
        Returns:
            List of Value objects that require gradients
        """
        return []
    
    def zero_grad(self):
        """
        Reset gradients of all parameters to zero.
        """
        for p in self.parameters():
            p.grad = 0.0
    
    def train(self):
        """
        Set module to training mode.
        
        Affects layers with different behavior during training vs evaluation
        (e.g., dropout, batch norm).
        """
        self.training = True
        for child in self._get_children():
            if hasattr(child, 'train'):
                child.train()
    
    def eval(self):
        """
        Set module to evaluation mode.
        
        Disables dropout, uses running statistics for batch norm, etc.
        """
        self.training = False
        for child in self._get_children():
            if hasattr(child, 'eval'):
                child.eval()
    
    def _get_children(self) -> List['Module']:
        """
        Get all child modules.
        
        Override this if your module contains submodules.
        """
        return []


class Neuron(Module):
    """
    A single artificial neuron.
    Implements: output = activation(weighted_sum(inputs) + bias)
    
    Args:
        nin: Number of input connections
        activation: Activation function to use ('tanh', 'relu', 'sigmoid', etc.)
        use_bias: Whether to include a bias term
        weight_init: Initialization strategy ('uniform', 'normal', 'xavier')
    """
    
    def __init__(self, nin: int, activation: str = 'tanh', 
                use_bias: bool = True, weight_init: str = 'uniform'):
        super().__init__()
        
        self.nin = nin
        self.activation_name = activation
        self.use_bias = use_bias
        
        # Initialize weights
        self.w = [self._init_weight(weight_init) for _ in range(nin)]
        
        # Initialize bias
        self.b = Value(0.0) if use_bias else None
    
    def _init_weight(self, init_type: str) -> Value:
        """
        Initialize a single weight based on the specified strategy.
        """
        if init_type == 'uniform':
            # Standard uniform initialization for small networks
            return Value(random.uniform(-1.0, 1.0))
        
        elif init_type == 'normal':
            # Normal initialization with small variance
            return Value(random.gauss(0.0, 0.1))
        
        elif init_type == 'xavier':
            # Xavier/Glorot initialization (good for tanh/sigmoid)
            # Variance = 1/sqrt(nin)
            std = 1.0 / (self.nin ** 0.5)
            return Value(random.gauss(0.0, std))
        
        else:
            raise ValueError(f"Unknown initialization: {init_type}")
    
    def _get_activation(self, x: Value) -> Value:
        """
        Apply the activation function to the input.
        """
        if self.activation_name == 'tanh':
            return x.tanh()
        elif self.activation_name == 'relu':
            return x.relu()
        elif self.activation_name == 'leaky_relu':
            return x.leaky_relu(0.01)
        elif self.activation_name == 'sigmoid':
            return x.sigmoid()
        elif self.activation_name == 'gelu':
            return x.gelu()
        elif self.activation_name == 'elu':
            return x.elu()
        elif self.activation_name == 'linear' or self.activation_name == 'none':
            return x
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def __call__(self, x: List[Value]) -> Value:
        """
        Forward pass through the neuron.
        """
        if len(x) != self.nin:
            raise ValueError(f"Expected {self.nin} inputs, got {len(x)}")
        
        # Compute weighted sum: sum(w_i * x_i)
        weighted_sum = sum((wi * xi for wi, xi in zip(self.w, x)), Value(0.0))
        
        # Add bias
        if self.use_bias:
            weighted_sum = weighted_sum + self.b
        
        # Apply activation
        return self._get_activation(weighted_sum)
