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
        
        # maximum number of timesteps in an episode
        self.max_ep_length = config.get("max_ep_length", 1024)
        
        # dimensions for state and action spaces (sequence lengths)
        self.L_s = config.get("max_state_len", 20)
        self.L_a = config.get("max_action_len", 10)

        # embedding layers for timesteps, returns, states, and actions (t,r,s,a)
        self.embed_timestep = nn.Embedding(self.max_ep_length, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        
        # We use a single shared token embedding layer for discrete tokens in states and actions
        self.embed_token = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Segment embeddings to distinguish between Return (0), State (1), and Action (2)
        self.embed_segment = nn.Embedding(3, self.hidden_size)

        # Normalization of the embedding layers
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        
        # prediction heads
        self.predict_token = nn.Linear(self.hidden_size, self.vocab_size)
        self.predict_reward = nn.Linear(self.hidden_size, 1)

        # Ensure the transformer's internal causal mask is large enough for flat sequence
        step_len = 1 + self.L_s + self.L_a
        transformer_config = config.copy()
        transformer_config["context_size"] = transformer_config.get(
            "context_size", self.max_ep_length * step_len
        )

        # create an instance of the transformer model
        # this will be used to generate the next steps in the sequence.
        self.transformer = TransformerModel(transformer_config)

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        Generate the next action token predictions and reward predictions.
        Based on the DecisionTransformer algorithm with fine-grained token-level interleaving.
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        device = states.device
        
        # 1. Embed individual components
        # Timesteps embedding: (B, seq_len, hidden_size)
        pos_embedding = self.embed_timestep(timesteps)
        
        # Returns embedding: (B, seq_len, 1, hidden_size)
        returns_embeddings = self.embed_return(returns_to_go).unsqueeze(2)
        segment_r = self.embed_segment(torch.tensor([0], device=device)).unsqueeze(0).unsqueeze(0)
        returns_embeddings = returns_embeddings + segment_r + pos_embedding.unsqueeze(2)
        
        # States embedding: (B, seq_len, L_s, hidden_size)
        state_embeddings = self.embed_token(states)
        segment_s = self.embed_segment(torch.tensor([1], device=device)).unsqueeze(0).unsqueeze(0)
        state_embeddings = state_embeddings + segment_s + pos_embedding.unsqueeze(2)
        
        # Actions embedding: (B, seq_len, L_a, hidden_size)
        action_embeddings = self.embed_token(actions)
        segment_a = self.embed_segment(torch.tensor([2], device=device)).unsqueeze(0).unsqueeze(0)
        action_embeddings = action_embeddings + segment_a + pos_embedding.unsqueeze(2)
        
        # 2. Interleave step blocks: R_t, s_t, a_t
        # Concatenate along the token dimension: shape (B, seq_len, 1 + L_s + L_a, hidden_size)
        step_len = 1 + self.L_s + self.L_a
        stacked_inputs = torch.cat((returns_embeddings, state_embeddings, action_embeddings), dim=2)
        
        # Flatten time and token dimensions: shape (B, seq_len * step_len, hidden_size)
        flat_inputs = stacked_inputs.view(batch_size, seq_len * step_len, self.hidden_size)
        
        input_embeddings = self.embed_ln(flat_inputs)
        
        # Pass to the transformer blocks
        hidden_states = self.transformer(input_embeddings) # (B, seq_len * step_len, hidden_size)
        
        # 3. Slice hidden states to predict actions and rewards
        # We predict action token a_{t, j} using the preceding hidden state at offset (L_s + j)
        indices = torch.arange(seq_len, device=device).unsqueeze(1) * step_len + self.L_s + torch.arange(self.L_a, device=device).unsqueeze(0)
        indices = indices.view(-1) # (seq_len * L_a)
        action_hidden = hidden_states[:, indices, :] # (B, seq_len * L_a, hidden_size)
        
        # We predict the reward from the last state token's hidden state at offset L_s
        reward_indices = torch.arange(seq_len, device=device) * step_len + self.L_s
        reward_hidden = hidden_states[:, reward_indices, :] # (B, seq_len, hidden_size)
        
        # Compute predictions
        action_preds = self.predict_token(action_hidden).view(batch_size, seq_len, self.L_a, self.vocab_size)
        reward_preds = self.predict_reward(reward_hidden) # (B, seq_len, 1)
        
        return action_preds, reward_preds