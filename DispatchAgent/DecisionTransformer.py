import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(
            config["embed_dim"], config["head_size"], config["use_bias"]
        )
        self.K_weights = nn.Linear(
            config["embed_dim"], config["head_size"], config["use_bias"]
        )
        self.V_weights = nn.Linear(
            config["embed_dim"], config["head_size"], config["use_bias"]
        )

        self.dropout = nn.Dropout(config["dropout_rate"])

        casual_attention_mask = torch.tril(
            torch.ones(config["context_size"], config["context_size"])
        )
        self.register_buffer("casual_attention_mask", casual_attention_mask)

    def forward(self, input):  # (B, C, embedding_dim)
        batch_size, tokens_num, embedding_dim = input.shape
        Q = self.Q_weights(input)  # (B, C, head_size)
        K = self.K_weights(input)  # (B, C, head_size)
        V = self.V_weights(input)  # (B, C, head_size)

        # Matrix Multiplay Q x K transpose to get the dot product of the query vectors with the key vectors
        attention_scores = Q @ K.transpose(1, 2)  # (B, C, C)

        # scale attention scores, scalled by square root of the dimensionality of the key vectors
        attention_scores = attention_scores / (K.shape[-1] ** 0.5)
        
        # mask the attention scores
        attention_scores = attention_scores.masked_fill(
            self.casual_attention_mask[:tokens_num, :tokens_num] == 0, -torch.inf
        )

        # calculate softmax values
        attention_scores = torch.softmax(attention_scores, dim=-1)

        # apply dropout for regularization
        attention_scores = self.dropout(attention_scores)

        # multiply attention scores by the value function.
        return attention_scores @ V  # (B, C, head_size)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # initialize the individual AttentionHead objects
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)

        # Feedforward connection for after the attention heads
        self.linear = nn.Linear(config["embed_dim"], config["embed_dim"])

        # Dropout regularization.
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input):
        # execute heads in ||
        heads_outputs = [head(input) for head in self.heads]

        # concatenate the outputs into a single tensor
        scores_change = torch.cat(heads_outputs, dim=-1)

        # run the results through a feed forward network.
        scores_change = self.linear(scores_change)

        # regularization and return results
        return self.dropout(scores_change)

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.linear_layers = nn.Sequential(
            nn.Linear(config["embed_dim"], config["embed_dim"] * 4),
            nn.GELU(),
            nn.Linear(config["embed_dim"] * 4, config["embed_dim"]),
            nn.Dropout(config["dropout_rate"]),
        )

    def forward(self, input):
        return self.linear_layers(input)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["embed_dim"])

        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["embed_dim"])

    def forward(self, input):
        residual = input
        x = self.multi_head(self.layer_norm_1(input))
        x = x + residual

        residual = x
        x = self.feed_forward(self.layer_norm_2(x))
        return x + residual

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = [Block(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(config["embed_dim"])


    def forward(self, input_embeddings):
        """
        Forward step for the transformer model. The DecisionTransformer already handles embeddings and positional encodings.
        We are simply making predictions using the AttentionHeads and returning the final hidden layer.
        """
        
        # Pass the embeddings through the stacked Transformer blocks
        x = self.layers(input_embeddings)
        
        # Apply the final layer normalization
        return self.layer_norm(x)

class DecisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # size of the hidden layer
        self.hidden_size = config["embed_dim"]
        self.vocab_size = config["vocab_size"]
        self.context_size = config.get("context_size", config.get("max_ep_length", 1024))

        # Discrete embedding and position embedding layers
        self.embed_token = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embed_pos = nn.Embedding(self.context_size, self.hidden_size)

        # Normalization of the embedding layers
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        
        # Token prediction head
        self.predict_token = nn.Linear(self.hidden_size, self.vocab_size)

        # create an instance of the transformer model
        # this will be used to generate the next steps in the sequence.
        self.transformer = TransformerModel(config)

    def forward(self, token_ids):
        """
        Generate predictions for the next token in the sequence.
        """
        batch_size, seq_length = token_ids.shape
        
        # Position indices: [0, 1, ..., seq_length - 1]
        device = token_ids.device
        positions = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0) # (1, seq_length)
        
        # Embed tokens and positions
        token_embeddings = self.embed_token(token_ids) # (B, C, embed_dim)
        position_embeddings = self.embed_pos(positions) # (1, C, embed_dim)
        
        input_embeddings = self.embed_ln(token_embeddings + position_embeddings)
        
        # Pass to the transformer blocks
        hidden_states = self.transformer(input_embeddings) # (B, C, embed_dim)
        
        # Predict logits over vocabulary for each position
        logits = self.predict_token(hidden_states) # (B, C, vocab_size)
        
        return logits