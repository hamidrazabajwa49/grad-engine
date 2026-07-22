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
