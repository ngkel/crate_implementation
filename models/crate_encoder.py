"""
CRATE Encoder Implementation

This module implements the CRATE (whitebox transformer) encoder architecture
with modular components: LayerNorm, MSSA (Multi-Scale Self-Attention), and
ISTA (Iterative Shrinkage-Thresholding Algorithm).

The encoder accepts pre-embedded tokens/patches and processes them through
a series of encoder blocks, each containing LayerNorm -> MSSA -> LayerNorm -> ISTA.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
import torch.nn.functional as F
import torch.nn.init as init


class LayerNorm(nn.Module):
    """
    Layer Normalization module.
    
    Applies layer normalization to stabilize and normalize the input features.
    Uses PyTorch's built-in LayerNorm for efficiency.
    
    Args:
        dim: Feature dimension
        eps: Small value to avoid division by zero (default: 1e-6)
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            
        Returns:
            Normalized tensor of shape [batch, seq_len, dim]
        """
        return self.norm(x)


class MSSA(nn.Module):
    def __init__(
        self,
        dim, #
        heads=8,
        dim_head=64,
        dropout=0.,
    ):
        super().__init__()
        inner_dim = dim_head * heads # dim_head is the dimension of the U_k^TZ vectors for each head. It is also d in the rearrange operation.
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # projection matrix
        self.qkv = nn.Linear(dim, inner_dim, bias=False) 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        
        # project the input x to the K subspaces U_k
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        
        # auto-correlation matrix
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        # softmax to get the attention weights
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        # Store intermediate output before to_out for coding rate computation
        self._intermediate_output = out
        return self.to_out(out)


class ISTA(nn.Module):
    def __init__(self, dim, dropout=0., step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output


class EncoderBlock(nn.Module):
    """
    CRATE Encoder Block.
    
    A single encoder block following the architecture:
    LayerNorm -> MSSA -> LayerNorm -> ISTA
    
    Each sub-module is wrapped with a residual connection.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads for MSSA
        dim_head: Dimension per attention head
        dropout: Dropout probability (default: 0.0, matching CRATE paper)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads,
        dim_head: int,
        dropout: float = 0.0,
        step_size_ista: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = LayerNorm(dim)
        self.mssa = MSSA(dim, num_heads, dim_head, dropout)
        self.norm2 = LayerNorm(dim)
        self.ista = ISTA(dim, dropout, step_size_ista)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            
        Returns:
            Output tensor of shape [batch, seq_len, dim]
        """
        # LayerNorm -> MSSA with residual connection
        x = self.norm1(x) + self.mssa(self.norm1(x))
        
        # LayerNorm -> ISTA with residual connection
        x = self.norm2(x) + self.ista(self.norm2(x))
        
        return x


class CRATEEncoder(nn.Module):
    """
    CRATE Encoder (Transformer).
    
    Main encoder class that stacks multiple EncoderBlock instances.
    The encoder accepts pre-embedded tokens/patches and processes them
    through a series of encoder blocks, followed by a final layer normalization.
    
    Args:
        dim: Feature dimension
        num_blocks: Number of encoder blocks
        num_heads: Number of attention heads for MSSA
        dim_head: Dimension per attention head
        dropout: Dropout probability (default: 0.0, matching CRATE paper)
    """
    
    def __init__(
        self,
        dim: int, # dimension of token representation z, don't confuses it with patch_dim which is the raw
        num_blocks: int,
        num_heads: int,
        dim_head: int,
        dropout: float = 0.0,
        step_size_ista: float = 0.1
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, num_heads, dim_head, dropout, step_size_ista)
            for _ in range(num_blocks)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
               This should be pre-embedded tokens/patches (patch embedding
               and positional embedding should be applied before this encoder)
            
        Returns:
            Encoded tensor of shape [batch, seq_len, dim]
        """
        # Process through encoder blocks
        for block in self.blocks:
            x = block(x)
        
        return x

