"""
Spatial and relation embedding generators for text-guided detection.
Creates specialized embeddings for spatial and relational phrases.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import math


class SpatialEmbeddingEncoder:
    """
    Generates spatial embeddings for spatial phrases.
    Creates 64-dimensional spatial token embeddings based on spatial keywords.
    """
    
    SPATIAL_DIM = 64
    
    # Spatial token embeddings (learned or fixed)
    SPATIAL_VOCAB = {
        "left": 0,
        "right": 1,
        "top": 2,
        "bottom": 3,
        "center": 4,
        "above": 5,
        "below": 6,
        "beside": 7,
        "null": 8,  # For non-spatial phrases
    }
    
    @classmethod
    def encode(cls, phrase_text: str, phrase_type: str) -> torch.Tensor:
        """
        Create a spatial embedding for a phrase.
        
        Args:
            phrase_text: The phrase text
            phrase_type: One of 'spatial', 'relation', 'attribute', 'object'
            
        Returns:
            Tensor of shape [64] containing spatial encoding
        """
        embedding = torch.zeros(cls.SPATIAL_DIM, dtype=torch.float32)
        
        if phrase_type != "spatial":
            # For non-spatial phrases, return zeros (will be handled as padding)
            return embedding
        
        text_lower = phrase_text.lower()
        
        # Encode spatial directions using positional encodings
        # Use a spatial encoding inspired by sin/cos positional encoding
        
        # Left-Right axis (dimensions 0-15)
        if "left" in text_lower:
            embedding[0:8] = torch.sin(torch.arange(8, dtype=torch.float32) * (math.pi / 8))
        elif "right" in text_lower:
            embedding[0:8] = torch.cos(torch.arange(8, dtype=torch.float32) * (math.pi / 8))
        
        # Top-Bottom axis (dimensions 16-31)
        if "top" in text_lower or "upper" in text_lower:
            embedding[16:24] = torch.sin(torch.arange(8, dtype=torch.float32) * (math.pi / 8))
        elif "bottom" in text_lower or "lower" in text_lower:
            embedding[16:24] = torch.cos(torch.arange(8, dtype=torch.float32) * (math.pi / 8))
        
        # Center marker (dimension 32)
        if "middle" in text_lower or "center" in text_lower:
            embedding[32] = 1.0
        
        # Vertical/Horizontal encoding (dimensions 40-47)
        if "above" in text_lower or "on top" in text_lower:
            embedding[40:48] = 1.0
        elif "below" in text_lower or "under" in text_lower:
            embedding[40:48] = -1.0
        
        # Adjacency encoding (dimensions 48-55)
        if "next to" in text_lower or "beside" in text_lower:
            embedding[48:56] = 0.5
        
        # Normalize
        norm = torch.norm(embedding)
        if norm > 0:
            embedding = embedding / (norm + 1e-8)
        
        return embedding


class RelationEmbeddingEncoder:
    """
    Generates relation embeddings for comparative/relation phrases.
    Creates 64-dimensional relation token embeddings based on relation keywords.
    """
    
    RELATION_DIM = 64
    
    RELATION_VOCAB = {
        "bigger": 0,
        "smaller": 1,
        "similar": 2,
        "same": 3,
        "null": 4,
    }
    
    @classmethod
    def encode(cls, phrase_text: str, phrase_type: str) -> torch.Tensor:
        """
        Create a relation embedding for a phrase.
        
        Args:
            phrase_text: The phrase text
            phrase_type: One of 'spatial', 'relation', 'attribute', 'object'
            
        Returns:
            Tensor of shape [64] containing relation encoding
        """
        embedding = torch.zeros(cls.RELATION_DIM, dtype=torch.float32)
        
        if phrase_type != "relation":
            # For non-relation phrases, return zeros
            return embedding
        
        text_lower = phrase_text.lower()
        
        # Size comparison encoding (dimensions 0-15)
        if "bigger" in text_lower or "larger" in text_lower:
            embedding[0:8] = 1.0  # Positive for bigger
        elif "smaller" in text_lower:
            embedding[0:8] = -1.0  # Negative for smaller
        
        # Similarity encoding (dimensions 16-23)
        if "similar" in text_lower:
            embedding[16:24] = 1.0
        
        # Equality encoding (dimensions 24-31)
        if "same" in text_lower:
            embedding[24:32] = 1.0
        
        # Magnitude encoding (dimensions 32-39)
        if "much" in text_lower:
            embedding[32:40] = 2.0  # Large magnitude
        elif "little" in text_lower or "a little" in text_lower:
            embedding[32:40] = 0.5  # Small magnitude
        else:
            embedding[32:40] = 1.0  # Normal magnitude
        
        # Normalize
        norm = torch.norm(embedding)
        if norm > 0:
            embedding = embedding / (norm + 1e-8)
        
        return embedding


def create_augmented_embeddings(
    text_embeddings: torch.Tensor,  # [N, 768]
    phrase_types: List[str],
    device: torch.device = None,
) -> torch.Tensor:
    """
    Augment text embeddings with spatial/relation encodings.
    
    Args:
        text_embeddings: Text embeddings [N, 768]
        phrase_types: List of phrase types for each embedding
        device: Device to place tensors on
        
    Returns:
        Augmented embeddings [N, 832] = [N, 768 + 64]
    """
    if device is None:
        device = text_embeddings.device
    
    n_phrases = text_embeddings.shape[0]
    spatial_embeddings = torch.zeros((n_phrases, 64), dtype=torch.float32).to(device)
    
    for i, phrase_type in enumerate(phrase_types):
        if phrase_type == "spatial":
            # Will be filled with actual spatial data if available
            spatial_embeddings[i] = SpatialEmbeddingEncoder.encode("", phrase_type).to(device)
        elif phrase_type == "relation":
            # Will be filled with actual relation data if available
            spatial_embeddings[i] = RelationEmbeddingEncoder.encode("", phrase_type).to(device)
        # For "attribute" and "object", keep zeros
    
    # Concatenate text embeddings with spatial embeddings
    augmented = torch.cat([text_embeddings, spatial_embeddings], dim=1)
    return augmented


class LearnableSpatialEncoder(nn.Module):
    """
    Learnable spatial encoder that can be trained.
    """
    
    def __init__(self, input_dim: int = 768, spatial_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.spatial_dim = spatial_dim
        
        # Learnable spatial projection
        self.spatial_proj = nn.Linear(input_dim, spatial_dim, bias=False)
        
        # Type-specific spatial encoders
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim * 2),
            nn.GELU(),
            nn.Linear(spatial_dim * 2, spatial_dim),
        )
        
        self.relation_encoder = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim * 2),
            nn.GELU(),
            nn.Linear(spatial_dim * 2, spatial_dim),
        )
    
    def forward(self, text_embeddings: torch.Tensor, phrase_types: List[str]) -> torch.Tensor:
        """
        Augment text embeddings with learned spatial encodings.
        
        Args:
            text_embeddings: [B, T, 768]
            phrase_types: List of phrase types
            
        Returns:
            Augmented embeddings [B, T, 832]
        """
        batch_size, num_tokens, embed_dim = text_embeddings.shape
        
        # Project text embeddings to spatial dimension
        spatial_base = self.spatial_proj(text_embeddings)  # [B, T, 64]
        
        # Apply type-specific processing
        spatial_encodings = torch.zeros_like(spatial_base)
        
        for i, ptype in enumerate(phrase_types):
            if ptype == "spatial":
                # Apply spatial encoder
                spatial_encodings[:, i] = self.spatial_encoder(spatial_base[:, i])
            elif ptype == "relation":
                # Apply relation encoder
                spatial_encodings[:, i] = self.relation_encoder(spatial_base[:, i])
            # For other types, keep zeros
        
        # Concatenate with original embeddings
        augmented = torch.cat([text_embeddings, spatial_encodings], dim=-1)
        return augmented
