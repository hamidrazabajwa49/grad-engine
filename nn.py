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

        def parameters(self) -> List[Value]:
        """
        Get all trainable parameters.
        
        Returns:
            List containing weights and bias (if used)
        """
        params = self.w.copy()
        if self.use_bias:
            params.append(self.b)
        return params
    
    def __repr__(self) -> str:
        return f"Neuron(nin={self.nin}, activation={self.activation_name})"



class Layer(Module):
    """
    A layer of neurons.
    
    A layer applies the same transformation to all inputs using a collection
    of neurons, each with its own set of weights and bias.
    
    Args:
        nin: Number of input connections
        nout: Number of neurons in this layer
        activation: Activation function to use
        use_bias: Whether to include bias terms
        weight_init: Weight initialization strategy
    """
    
    def __init__(self, nin: int, nout: int, activation: str = 'tanh',
                use_bias: bool = True, weight_init: str = 'uniform'):
        super().__init__()
        
        self.nin = nin
        self.nout = nout
        self.activation = activation
        
        # Create neurons
        self.neurons = [
            Neuron(nin, activation=activation, use_bias=use_bias, 
                    weight_init=weight_init)
            for _ in range(nout)
        ]

    def __call__(self, x: List[Value]) -> List[Value]:
        """
        Forward pass through the layer.
        
        Args:
            x: List of input Values
        """
        # Pass input through each neuron
        return [neuron(x) for neuron in self.neurons]
    
    def parameters(self) -> List[Value]:
        """
        Get all trainable parameters from all neurons.
        """
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
    
    def _get_children(self) -> List[Module]:
        """
        Get child modules (neurons in this layer).
        """
        return self.neurons
    
    def __repr__(self) -> str:
        return f"Layer(nin={self.nin}, nout={self.nout}, activation={self.activation})"


class MLP(Module):
    """
    Multi-Layer Perceptron (fully connected neural network).
    
    An MLP consists of multiple layers stacked sequentially.
    Input passes through each layer in order.
    
    Args:
        nin: Number of input features
        nouts: List of output sizes for each layer
        (e.g., [16, 16, 1] creates 3 layers)
        activations: Activation function(s) for each layer
        Can be a single string or a list per layer
        use_bias: Whether to include bias terms
        weight_init: Weight initialization strategy
        dropout_rate: Dropout probability (0.0 = no dropout)
    """
    
    def __init__(self, nin: int, nouts: List[int], 
                activation: Union[str, List[str]] = 'tanh',
                use_bias: bool = True, 
                weight_init: str = 'uniform',
                dropout_rate: float = 0.0):
        super().__init__()
        
        self.nin = nin
        self.nouts = nouts
        self.dropout_rate = dropout_rate
        
        # Handle activation specification
        if isinstance(activation, str):
            activations = [activation] * len(nouts)
        elif isinstance(activation, list):
            if len(activation) != len(nouts):
                raise ValueError("Number of activations must match number of layers")
            activations = activation
        else:
            raise TypeError("activation must be str or list of str")
        
        # Build layers
        layer_sizes = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            layer = Layer(
                nin=layer_sizes[i],
                nout=layer_sizes[i+1],
                activation=activations[i],
                use_bias=use_bias,
                weight_init=weight_init
            )
            self.layers.append(layer)
    
    def __call__(self, x: List[Value]) -> Value:
        """
        Forward pass through the MLP.
        """
        # Pass through each layer
        current = x
        for i, layer in enumerate(self.layers):
            current = layer(current)
            
            # Apply dropout between layers (except after last layer)
            if self.dropout_rate > 0 and self.training and i < len(self.layers) - 1:
                current = [v.dropout(self.dropout_rate, training=self.training) 
                        for v in current]
        
        # For binary classification, return single Value
        # If multiple outputs, return list
        if len(current) == 1:
            return current[0]
        else:
            return current
    
    def parameters(self) -> List[Value]:
        """
        Get all trainable parameters from all layers.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def _get_children(self) -> List[Module]:
        """
        Get child modules (layers in this MLP).
        """
        return self.layers
    
    def layers_info(self) -> List[dict]:
        """
        Get information about each layer.
        """
        info = []
        for i, layer in enumerate(self.layers):
            info.append({
                'layer': i + 1,
                'type': 'Linear',
                'in_features': layer.nin,
                'out_features': layer.nout,
                'activation': layer.activation,
                'parameters': len(layer.parameters())
            })
        return info
    
    def __repr__(self) -> str:
        layer_reprs = [repr(layer) for layer in self.layers]
        layers_str = '\n  ' + '\n  '.join(layer_reprs)
        return f"MLP(\n  {layers_str}\n)"
    
    def summary(self) -> str:
        """
        Generate a detailed summary of the model architecture.
        """
        total_params = len(self.parameters())
        
        lines = [
            "=" * 60,
            f"MLP Architecture Summary",
            "=" * 60,
            f"Input features: {self.nin}",
            f"Total layers: {len(self.layers)}",
            f"Total parameters: {total_params:,}",
            "-" * 60,
            "Layer | In  | Out | Activation | Parameters"
        ]
        
        for i, layer in enumerate(self.layers):
            lines.append(
                f"  {i+1:2d}  | {layer.nin:3d} | {layer.nout:3d} | "
                f"{layer.activation:10s} | {len(layer.parameters()):8d}"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)


class Sequential(Module):
    """
    Sequential container for stacking modules.
    
    Similar to MLP but more flexible, allows arbitrary modules
    (not just linear layers).
    """
    
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules = list(modules)
    
    def __call__(self, x):
        """
        Forward pass through all modules sequentially.
        """
        current = x
        for module in self.modules:
            current = module(current)
        return current
    
    def parameters(self) -> List[Value]:
        """
        Get all parameters from all modules.
        """
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params
    
    def _get_children(self) -> List[Module]:
        """
        Get all child modules.
        """
        return self.modules
    
    def __repr__(self) -> str:
        module_reprs = [repr(m) for m in self.modules]
        modules_str = '\n  ' + '\n  '.join(module_reprs)
        return f"Sequential(\n  {modules_str}\n)"
    
    def add_module(self, module: Module):
        """
        Add a module to the end of the sequence.
        """
        self.modules.append(module)
        return self
