"""
TensorBoard Visualization Utilities

This module provides utilities for visualizing PyTorch models using TensorBoard,
including model structure visualization and computation graph logging.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union
import pathlib


def visualize_model_graph(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    log_dir: Union[str, pathlib.Path] = "runs/model_visualization",
    model_name: Optional[str] = None,
    verbose: bool = True
) -> SummaryWriter:
    """
    Visualize a PyTorch model's computation graph using TensorBoard.
    
    This function creates a TensorBoard log with the model's computation graph,
    showing all layers, operations, and data flow. The graph can be viewed
    in TensorBoard by running: tensorboard --logdir=<log_dir>
    
    Args:
        model: PyTorch model to visualize (should be in eval mode)
        sample_input: Sample input tensor matching the model's expected input shape
        log_dir: Directory where TensorBoard logs will be saved
        model_name: Optional name for the model (used in log directory naming)
        verbose: Whether to print status messages
        
    Returns:
        SummaryWriter instance that can be used for additional logging
        
    Example:
        >>> model = CRATEClassification(...)
        >>> model.eval()
        >>> sample_input = torch.randn(1, 1, 28, 28)
        >>> writer = visualize_model_graph(model, sample_input)
        >>> # View in TensorBoard: tensorboard --logdir=runs/
    """
    # Ensure model is in eval mode for graph tracing
    model.eval()
    
    # Create log directory path
    log_path = pathlib.Path(log_dir)
    if model_name:
        log_path = log_path / model_name
    
    # Create SummaryWriter
    writer = SummaryWriter(log_dir=str(log_path))
    
    if verbose:
        print(f"Visualizing model structure...")
        print(f"Model: {model.__class__.__name__}")
        print(f"Sample input shape: {sample_input.shape}")
        print(f"Log directory: {log_path.absolute()}")
    
    # Add graph to TensorBoard
    # Note: sample_input should be on the same device as the model
    try:
        with torch.no_grad():
            writer.add_graph(model, sample_input)
        
        if verbose:
            print(f"\n✓ Model graph successfully logged to TensorBoard!")
            print(f"\nTo view the visualization, run:")
            print(f"  tensorboard --logdir={log_path.parent.absolute()}")
            print(f"\nThen open your browser and navigate to: http://localhost:6006")
            print(f"Click on the 'GRAPHS' tab to view the model structure.")
    
    except Exception as e:
        if verbose:
            print(f"\n✗ Error while creating graph: {e}")
            print(f"Make sure the sample_input is on the same device as the model.")
        raise
    
    return writer

