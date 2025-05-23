import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, ff_dim=32, dropout_rate=0.1, num_classes=2):
        super(TransformerModel, self).__init__()
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, y, z):
        x, attn_weights = self.transformer_block(x, y, z)
        x = x.permute(0, 2, 1)  # Permute for global average pooling (batch, embed_dim, seq_len)
        x = self.global_avg_pool(x).squeeze(-1) # x: aggregated embedding
        x = self.dropout(x)

        return self.classifier(x), attn_weights

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim),
                                 nn.ReLU(),
                                 nn.Linear(ff_dim, embed_dim))
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, y, z):
        attn_output, attn_weights = self.attention(x, y, z)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output)), attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, key_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim // num_heads

        # PyTorch multi-head attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=key_dim, num_heads=num_heads)

    def forward(self, x, y, z, mask=None):
        # PyTorch multi-head attention expects (seq_len, batch, embedding_dim)
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        z = z.permute(1, 0, 2)
        attn_output, attn_weights = self.multihead_attn(x, y, z, attn_mask=mask)
        return attn_output.permute(1, 0, 2), attn_weights  # Transpose back to (batch, seq_len, embedding_dim)

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.position_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.position_emb(positions)
        return x



