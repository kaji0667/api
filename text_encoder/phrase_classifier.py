"""
Phrase type classification for text-guided detection.
Classifies phrases into: spatial, attribute, relation, and object types.
"""

from typing import Dict, List, Tuple, Set
import re


# Spatial keywords for location-based phrases
SPATIAL_KEYWORDS = {
    # Cardinal directions
    "left", "right", "top", "bottom", "middle", "center",
    # Relative positions
    "upper", "lower", "top-left", "top-right", "bottom-left", "bottom-right",
    "northeast", "northwest", "southeast", "southwest",
    # Prepositions
    "on", "at", "above", "below", "next to", "beside", "behind", "in front of",
    "inside", "outside", "between", "around",
    # Directional phrases
    "north", "south", "east", "west",
}

# Relation/Comparative keywords
RELATION_KEYWORDS = {
    "bigger", "smaller", "larger", "larger", "bigger than", "smaller than",
    "similar", "similar in size", "similar in color",
    "much", "much bigger", "much smaller", "much larger",
    "little", "a little", "a little bigger", "a little smaller",
    "same as", "same size as", "same color as",
    "is a", "is similar", "is bigger", "is smaller",
}

# Attribute keywords (colors, sizes, types, etc.)
ATTRIBUTE_KEYWORDS = {
    # Colors
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "gray", "white", "black",
    "light", "dark", "bright", "pale", "vibrant",
    # Sizes
    "tiny", "small", "medium", "large", "big", "huge", "long", "short", "wide", "narrow", "thick", "thin",
    # Textures/Materials
    "smooth", "rough", "shiny", "matte", "metallic", "wooden", "concrete", "asphalt",
    # Shapes
    "round", "square", "rectangular", "circular", "triangular", "curved", "straight",
}


def classify_phrase(text: str) -> str:
    """
    Classify a phrase into one of: 'spatial', 'relation', 'attribute', 'object'.
    
    Args:
        text: The phrase text to classify
        
    Returns:
        One of: 'spatial', 'relation', 'attribute', 'object'
    """
    text_lower = text.lower().strip()
    
    # Check for spatial keywords first
    for keyword in SPATIAL_KEYWORDS:
        if keyword in text_lower:
            return "spatial"
    
    # Check for relation keywords
    for keyword in RELATION_KEYWORDS:
        if keyword in text_lower:
            return "relation"
    
    # Check for attribute keywords
    for keyword in ATTRIBUTE_KEYWORDS:
        if keyword in text_lower:
            return "attribute"
    
    # Default to object
    return "object"


def classify_phrases_batch(texts: List[str]) -> List[str]:
    """
    Classify multiple phrases.
    
    Args:
        texts: List of phrase texts
        
    Returns:
        List of phrase types
    """
    return [classify_phrase(text) for text in texts]


def extract_spatial_tokens(text: str) -> List[str]:
    """
    Extract spatial tokens from a phrase.
    Returns a list of spatial token identifiers (e.g., ['left', 'upper']).
    """
    text_lower = text.lower()
    tokens = []
    
    # Check for cardinal directions
    if any(kw in text_lower for kw in ["left"]):
        tokens.append("left")
    if any(kw in text_lower for kw in ["right"]):
        tokens.append("right")
    if any(kw in text_lower for kw in ["top", "upper"]):
        tokens.append("top")
    if any(kw in text_lower for kw in ["bottom", "lower"]):
        tokens.append("bottom")
    if any(kw in text_lower for kw in ["middle", "center"]):
        tokens.append("center")
    
    # Check for relative positions
    if "above" in text_lower or "on top of" in text_lower:
        tokens.append("above")
    if "below" in text_lower or "under" in text_lower:
        tokens.append("below")
    if "next to" in text_lower or "beside" in text_lower:
        tokens.append("beside")
    
    return tokens


def extract_relation_tokens(text: str) -> List[str]:
    """
    Extract relation/comparative tokens from a phrase.
    Returns a list of relation token identifiers.
    """
    text_lower = text.lower()
    tokens = []
    
    # Size comparisons
    if "bigger" in text_lower or "larger" in text_lower:
        tokens.append("bigger")
    if "smaller" in text_lower:
        tokens.append("smaller")
    
    # Similarity
    if "similar" in text_lower:
        tokens.append("similar")
    
    # Magnitude
    if "much" in text_lower:
        tokens.append("much")
    if "little" in text_lower:
        tokens.append("little")
    
    # Equality
    if "same" in text_lower:
        tokens.append("same")
    
    return tokens


def get_phrase_type_weight(phrase_type: str, custom_weights: Dict[str, float] = None) -> float:
    """
    Get the weight for a phrase type.
    
    Args:
        phrase_type: One of 'spatial', 'relation', 'attribute', 'object'
        custom_weights: Optional custom weight dictionary
        
    Returns:
        Weight value (default 1.0)
    """
    default_weights = {
        "spatial": 2.0,
        "relation": 1.8,
        "attribute": 1.0,
        "object": 1.0,
    }
    
    if custom_weights and phrase_type in custom_weights:
        return custom_weights[phrase_type]
    
    return default_weights.get(phrase_type, 1.0)
