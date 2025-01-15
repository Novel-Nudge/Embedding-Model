import torch
import torch.nn as nn


class BookFeatureProcessor(nn.Module):

    def __init__(self):
        super().__init__()

        # Constants for feature dimensions
        self.n_emotional_tones = 10  # uplifting through peaceful
        self.n_metrics = 3  # reading_level, vocabulary_level, experimental_style
        self.n_genres = 7  # literary_fiction through historical_fiction

        # Emotional tone processing
        self.emotion_encoder = nn.Sequential(
            nn.Linear(self.n_emotional_tones, 12), nn.LayerNorm(16), nn.GELU())

        # Metric processing
        self.metric_encoder = nn.Sequential(nn.Linear(self.n_metrics, 8),
                                            nn.LayerNorm(8), nn.GELU())

        # Genre processing with attention for overlap
        self.genre_attention = nn.MultiheadAttention(embed_dim=self.n_genres,
                                                     num_heads=1,
                                                     batch_first=True)
        self.genre_encoder = nn.Sequential(nn.Linear(self.n_genres, 12),
                                           nn.LayerNorm(16), nn.GELU())

    def forward(self, emotional_tones, metrics, genres):
        # Process emotional tones
        emotion_features = self.emotion_encoder(emotional_tones)

        # Process metrics
        metric_features = self.metric_encoder(metrics)

        # Process genres with self-attention for overlap
        genre_attn_out, _ = self.genre_attention(genres.unsqueeze(1),
                                                 genres.unsqueeze(1),
                                                 genres.unsqueeze(1))
        genre_features = self.genre_encoder(genre_attn_out.squeeze(1))

        # Combine all features
        return torch.cat([emotion_features, metric_features, genre_features],
                         dim=1)
