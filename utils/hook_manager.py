"""
CRATE Hook Manager

This module provides a hook manager class to extract intermediate outputs
and subspaces from CRATEEncoder layers using PyTorch forward hooks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from ..models.crate_encoder import MSSA, ISTA, EncoderBlock


def remove_all_hooks(model: nn.Module):
    """
    Remove all forward hooks, forward pre-hooks, and backward hooks from a model.
    
    This function recursively traverses all modules in the model and removes
    all registered hooks, regardless of how they were registered.
    
    Args:
        model: The PyTorch model to remove hooks from
    """
    for module in model.modules():
        # Remove forward hooks
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        
        # Remove forward pre-hooks
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()
        
        # Remove backward hooks
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()


class CRATEHookManager:
    """
    Manages forward hooks for extracting intermediate outputs and subspaces
    from CRATEEncoder layers.
    
    This class registers hooks on MSSA, ISTA, and EncoderBlock modules to
    capture intermediate outputs and extract subspaces (U_k from MSSA and D from ISTA).
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the hook manager.
        
        Args:
            model: The CRATEClassification model instance
        """
        self.model = model
        self.hooks = []
        self.intermediates = {
            'mssa_outputs': [],      # Outputs after each MSSA (before residual)
            'ista_outputs': [],      # Outputs after each ISTA (before residual)
            'block_outputs': [],     # Outputs after each EncoderBlock
            'U_k_list': [],          # List of lists: U_k per block per head
            'D_list': []             # List of D matrices per ISTA block
        }
        
    def _extract_U_k(self, mssa_module: MSSA) -> List[torch.Tensor]:
        """
        Extract U_k subspaces from MSSA module.
        
        The qkv weight matrix is [inner_dim, dim] = [heads * dim_head, dim].
        We reshape it to [dim, heads, dim_head], then extract each head's subspace.
        
        Args:
            mssa_module: MSSA module instance
            
        Returns:
            List of U_k matrices, each of shape [dim, dim_head]
        """
        qkv_weight = mssa_module.qkv.weight  # [inner_dim, dim] = [heads * dim_head, dim]
        inner_dim, dim = qkv_weight.shape
        heads = mssa_module.heads
        dim_head = inner_dim // heads
        
        # Reshape to [dim, heads, dim_head]
        qkv_weight_T = qkv_weight.T  # [dim, inner_dim]
        U_k_reshaped = qkv_weight_T.view(dim, heads, dim_head)
        
        # Split into list: each U_k is [dim, dim_head]
        U_k_list = [U_k_reshaped[:, k, :] for k in range(heads)]
        
        return U_k_list
    
    def _make_mssa_hook(self, block_idx: int):
        """Create a forward hook for MSSA module."""
        def hook(module: MSSA, input: Tuple[torch.Tensor], output: torch.Tensor):
            # Capture intermediate output before to_out projection (shape [b n (h*d)])
            # This allows us to reshape to separate heads for coding rate computation
            if hasattr(module, '_intermediate_output'):
                intermediate_output = module._intermediate_output.detach().clone()
            else:
                # Fallback: use the final output (shape [b n dim])
                intermediate_output = output.detach().clone()
            self.intermediates['mssa_outputs'].append(intermediate_output)
            
            # Extract U_k subspaces
            U_k_list = self._extract_U_k(module)
            self.intermediates['U_k_list'].append(U_k_list)
        
        return hook
    
    def _make_ista_hook(self, block_idx: int):
        """Create a forward hook for ISTA module."""
        def hook(module: ISTA, input: Tuple[torch.Tensor], output: torch.Tensor):
            # output is the ISTA output (before residual)
            self.intermediates['ista_outputs'].append(output.detach().clone())
            
            # Extract dictionary D
            D = module.weight.detach().clone()  # [dim, dim]
            self.intermediates['D_list'].append(D)
        
        return hook
    
    def _make_block_hook(self, block_idx: int):
        """Create a forward hook for EncoderBlock module."""
        def hook(module: EncoderBlock, input: Tuple[torch.Tensor], output: torch.Tensor):
            # output is the final block output (after both MSSA and ISTA with residuals)
            self.intermediates['block_outputs'].append(output.detach().clone())
        
        return hook
    
    def register_hooks(self):
        """
        Register forward hooks on all MSSA, ISTA, and EncoderBlock modules.
        
        This method traverses the model structure and registers hooks on:
        - Each MSSA module (to capture outputs and extract U_k)
        - Each ISTA module (to capture outputs and extract D)
        - Each EncoderBlock module (to capture final block outputs)
        """
        # Get the CRATEEncoder from the classification model
        crate_encoder = self.model.crate
        
        # Register hooks on each EncoderBlock and its submodules
        for block_idx, block in enumerate(crate_encoder.blocks):
            # Hook on the block itself (captures final output)
            block_hook = block.register_forward_hook(self._make_block_hook(block_idx))
            self.hooks.append(block_hook)
            
            # Hook on MSSA (captures MSSA output and extracts U_k)
            mssa_hook = block.mssa.register_forward_hook(self._make_mssa_hook(block_idx))
            self.hooks.append(mssa_hook)
            
            # Hook on ISTA (captures ISTA output and extracts D)
            ista_hook = block.ista.register_forward_hook(self._make_ista_hook(block_idx))
            self.hooks.append(ista_hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            try:
                hook.remove()
            except (AttributeError, RuntimeError):
                # Hook may have already been removed or is invalid
                pass
        self.hooks.clear()
        
        # Also ensure all hooks are removed from the model modules
        remove_all_hooks(self.model)
    
    def get_intermediates(self) -> Dict[str, List]:
        """
        Get collected intermediate outputs and subspaces.
        
        Returns:
            Dictionary containing:
            - 'mssa_outputs': List of MSSA outputs (before residual) per block
            - 'ista_outputs': List of ISTA outputs (before residual) per block
            - 'block_outputs': List of final block outputs (with residuals) per block
            - 'U_k_list': List of lists of U_k matrices per block per head
            - 'D_list': List of D matrices per ISTA block
        """
        return self.intermediates.copy()
    
    def clear_intermediates(self):
        """Clear all collected intermediate outputs."""
        for key in self.intermediates:
            self.intermediates[key] = []
    
    def __enter__(self):
        """Context manager entry: register hooks."""
        self.register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: remove hooks."""
        self.remove_hooks()

