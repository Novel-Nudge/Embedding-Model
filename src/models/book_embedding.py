from typing import Union, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel


class BookEmbeddingModel(nn.Module):

    def __init__(self, dropout_rate=0.1, max_length=380):
        super().__init__()
        self.max_length = max_length

        # Load and freeze MiniLM
        self.encoder = AutoModel.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2')
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Attention block
        self.attention = nn.MultiheadAttention(384,
                                               8,
                                               dropout=dropout_rate,
                                               batch_first=True)
        self.norm = nn.LayerNorm(384)

        # Feed-forward network
        self.ff_network = nn.Sequential(nn.Linear(384, 1024), nn.GELU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(1024, 384))

        # Final projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Linear(128, 20),
            nn.LayerNorm(20)
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only embeddings for external loss computation.

        Args:
            input_ids: Tensor of shape [batch_size, seq_length]
            attention_mask: Tensor of shape [batch_size, seq_length]

        Returns:
            normalized_embeddings: Tensor of shape [batch_size, 20]
        """

        # Get base embeddings
        with torch.no_grad():
            base_embeddings = self.encoder(input_ids,
                                           attention_mask).last_hidden_state

        # Self-attention block
        attn_output, _ = self.attention(
            base_embeddings,  # q
            base_embeddings,  # k
            base_embeddings,  # v
            key_padding_mask=~attention_mask.bool(),
            need_weights=False)

        # Residual connection and layer norm
        residual = self.norm(attn_output + base_embeddings)

        # Feed-forward network
        ff_output = self.ff_network(residual)

        # Global average pooling
        seq_lengths = attention_mask.sum(dim=1, keepdim=True)
        pooled = (ff_output *
                  attention_mask.unsqueeze(-1)).sum(dim=1) / seq_lengths

        # Project to final embedding dimension
        outputs = self.projection(pooled)

        # L2 normalize
        return torch.nn.functional.normalize(outputs, p=2, dim=1)
