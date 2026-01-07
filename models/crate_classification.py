"""
CRATE Classification Model

This module implements a classification model using the CRATE encoder.
The model takes raw images, applies patch embedding and positional embedding,
encodes them using CRATEEncoder, and applies a trainable classification head
for image classification.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union
from einops import repeat
from .crate_encoder import CRATEEncoder


def pair(t):
    """Convert a value to a tuple if it's not already a tuple."""
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""
    
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer to create patch embeddings
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
            
        Returns:
            Output tensor of shape [batch, n_patches, embed_dim]
        """
        # x: [batch, channels, height, width]
        # Output: [batch, n_patches, embed_dim]
        x = self.proj(x)  # [batch, embed_dim, H', W']
        x = x.flatten(2)  # [batch, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch, n_patches, embed_dim]
        return x


class CRATEClassification(nn.Module):
    """
    CRATE Classification Model.
    
    Complete CRATE model for image classification. Takes raw images as input,
    applies patch embedding, positional embedding, and CLS token, then encodes
    using CRATEEncoder and applies a classification head.
    
    Args:
        img_size: Image size (height and width, or tuple for different dimensions)
        patch_size: Patch size (height and width, or tuple for different dimensions)
        in_channels: Number of input image channels (default: 1)
        embed_dim: Embedding dimension (default: 128)
        num_classes: Number of output classes (default: 10)
        num_blocks: Number of encoder blocks in CRATEEncoder (default: 4)
        num_heads: Number of attention heads for MSSA (default: 8)
        dim_head: Dimension per attention head (default: 16)
        dropout: Dropout probability (default: 0.0, matching CRATE paper)
        pool: Pooling type, either 'cls' (CLS token) or 'mean' (mean pooling) (default: 'cls')
    """
    
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_channels=1,
        embed_dim=128,
        num_classes=10,
        num_blocks=4,
        num_heads=8,
        dim_head=16,
        dropout=0.0,
        pool='cls'
    ):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.pool = pool
        self.crate = CRATEEncoder(
            dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
            
        Returns:
            Classification logits of shape [batch, num_classes]
        """
        # x: [batch, channels, height, width]
        x = self.patch_embed(x)  # [batch, n_patches, embed_dim]
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]   # [batch, n_patches + 1, embed_dim]
        x = self.dropout(x)
        x = self.crate(x)        # [batch, n_patches + 1, embed_dim]

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # [batch, embed_dim]

        x = self.to_latent(x)
        feature_last = x
        return self.mlp_head(x)  # [batch, num_classes]
