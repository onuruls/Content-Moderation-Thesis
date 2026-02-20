from typing import List, Protocol, runtime_checkable
from PIL import Image
import torch
import numpy as np

@runtime_checkable
class Model(Protocol):
    """
    Protocol for all content moderation models in this thesis.
    """
    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode images into feature vectors.
        Returns:
            Tensor of shape [B, D]
        """
        ...

    def logits(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Get raw logits from the model.
        Returns:
            Tensor of shape [B, C] (or [B] if binary)
        """
        ...

    def prob(self, images: List[Image.Image]) -> np.ndarray:
        """
        Get probabilities (sigmoid/softmax applied).
        Returns:
            Numpy array of shape [B, C] (or [B] if binary)
        """
        ...
